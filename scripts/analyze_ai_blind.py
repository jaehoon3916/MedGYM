#!/usr/bin/env python3
"""
AI-blind case analysis.

"AI-blind" = cases the medical AI gets wrong in ALL n_trials repeated solo attempts.

Cache stores {case_id: [trial0, trial1, ...]}. On re-runs only missing trials are called.
Existing per-case ai_alone_correct from results.json records counts as trial 0 (free).

Usage:
    python scripts/analyze_ai_blind.py \\
      --results \\
        outputs/poc_0630_naive/veteran_attending_full/results.json \\
        outputs/poc_0630_medcobe_feedback/veteran_attending_full/results.json \\
        outputs/poc_0630_deliberation_llm/veteran_attending_full/results.json \\
      --config configs/poc_0630_medcobe_feedback.yaml \\
      --n_trials 3 \\
      --cache  outputs/ai_blind_cache.json \\
      --out    outputs/ai_blind_analysis.json
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.json_utils import safe_json_load
from scripts.run_dialogue import load_dotenv

load_dotenv()

_VALID_LETTERS = ("A", "B", "C", "D")

_AI_ALONE_SYSTEM = (
    "You are a medical AI. Given a clinical case, answer the multiple-choice question "
    "by selecting the single best option."
)
_AI_ALONE_USER = """\
[Clinical Case]
{scenario}

[Image]
{image_caption}

[Question]
{question}

[Options]
{options_text}

Respond with JSON only: {{"selected_option": "<single letter A-D>", "reasoning": "<1-2 sentences>"}}"""


def _call_ai_alone(case_info, cfg: dict) -> bool:
    client = OpenAI(
        base_url=cfg.get("base_url", "http://localhost:8001/v1"),
        api_key=cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "EMPTY"),
    )
    options_text = "\n".join(f"  {k}. {v}" for k, v in case_info.options.items())
    messages = [
        {"role": "system", "content": _AI_ALONE_SYSTEM},
        {"role": "user", "content": _AI_ALONE_USER.format(
            scenario=case_info.scenario,
            image_caption=case_info.caption or "N/A",
            question=case_info.question,
            options_text=options_text,
        )},
    ]
    resp = client.chat.completions.create(
        model=cfg["model"], messages=messages, temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    data = safe_json_load(text)
    selected = str(data.get("selected_option", "")).strip().upper()
    return selected in _VALID_LETTERS and selected == str(case_info.correct_option).strip().upper()


def load_ai_alone_trials(
    cases,
    result_paths: list[Path],
    cache_path: Path,
    medical_llm_cfg: dict,
    n_trials: int,
    concurrency: int,
) -> dict[str, list[bool]]:
    """Returns {case_id: [trial0, trial1, ...]} with n_trials entries per case.

    Priority:
      1. Cache file — already-done trials reused, only gaps are filled.
      2. Per-case ai_alone_correct in results.json records — counts as trial 0.
      3. Fresh API calls for remaining gaps.
    """
    # 1. Load cache: {case_id: [bool, ...]}
    cache: dict[str, list[bool]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"  Cache: {sum(len(v) for v in cache.values())} trial(s) for {len(cache)} case(s)")

    # 2. Seed from results.json per-case field (each run = one free trial)
    seeded = 0
    for rp in result_paths:
        data = json.loads(rp.read_text())
        for r in data.get("records", []):
            cid = r.get("case_id")
            if cid and "ai_alone_correct" in r and cid not in cache:
                cache[cid] = [bool(r["ai_alone_correct"])]
                seeded += 1
    if seeded:
        print(f"  Seeded {seeded} case(s) from results.json records (trial 0).")

    # 3. API calls for missing trials
    pending: list[tuple] = []
    for c in cases:
        have = len(cache.get(c.case_id, []))
        for t in range(have, n_trials):
            pending.append((c, t))

    if pending:
        n_cases_pending = len({c.case_id for c, _ in pending})
        print(f"  {len(pending)} trial call(s) needed across {n_cases_pending} case(s)...")
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_call_ai_alone, c, medical_llm_cfg): (c.case_id, t) for c, t in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="ai_alone", unit="call"):
                cid, t = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"  WARNING: {cid} trial {t} failed — {e}")
                    result = False
                if cid not in cache:
                    cache[cid] = []
                while len(cache[cid]) <= t:
                    cache[cid].append(False)
                cache[cid][t] = result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        print(f"  Saved to {cache_path}")
    else:
        print(f"  All {n_trials} trial(s) done for all {len(cases)} case(s). No API calls.")

    return cache


def compute_curve(records: list[dict], blind_ids: set[str], also_doctor_wrong: bool = False) -> dict:
    """Per-checkpoint accuracy on AI-blind subset.

    also_doctor_wrong=True: further filters to cases where doctor was also wrong at turn-0
    (alone_correct=False). This is the "both wrong" / double-blind subset.
    """
    subset = [r for r in records if r["case_id"] in blind_ids]
    if also_doctor_wrong:
        subset = [r for r in subset if not r.get("alone_correct", True)]
    if not subset:
        return {"n": 0, "checkpoints": {}}
    cp_keys: set[int] = set()
    for r in subset:
        cp_keys.update(int(k) for k in (r.get("checkpoints") or r.get("checkpoint_results", {})))
    curve: dict[int, dict] = {}
    for c in sorted(cp_keys):
        corrects = []
        for r in subset:
            entry = (r.get("checkpoints") or r.get("checkpoint_results", {})).get(str(c))
            if entry:
                corrects.append(bool(entry["is_correct"]))
        if corrects:
            curve[c] = {"accuracy": round(sum(corrects) / len(corrects), 4), "n": len(corrects)}
    return {"n": len(subset), "checkpoints": curve}


def plot_results(result: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    by_policy = result["by_policy"]
    policies = list(by_policy.items())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"AI-blind analysis  (ai_blind n={result['n_ai_blind']}/{result['n_cases']}, "
        f"ai_alone_acc={result['ai_alone_accuracy']:.3f}, {result['n_trials']} trials)",
        fontsize=11,
    )

    def _draw_curve(ax, curve_key, title, ylabel):
        ax.set_title(title)
        for i, (label, info) in enumerate(policies):
            curve = info.get(curve_key, {})
            if not curve:
                continue
            cps = sorted(int(k) for k in curve)
            ys = [curve[str(c)]["accuracy"] for c in cps]
            ax.plot(cps, ys, marker="o", color=colors[i % len(colors)], label=label)
        ax.set_xlabel("checkpoint (turns)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    def _draw_delta(ax, curve_key, title):
        ax.set_title(title)
        ax.axhline(0.0, color="grey", linestyle=":", alpha=0.7, label="doctor alone (baseline)")
        for i, (label, info) in enumerate(policies):
            curve = info.get(curve_key, {})
            if not curve:
                continue
            cps = sorted(int(k) for k in curve)
            baseline = curve[str(cps[0])]["accuracy"] if str(cps[0]) in curve else 0.0
            ys = [curve[str(c)]["accuracy"] - baseline for c in cps]
            if cps[0] == 0:
                ys[0] = 0.0
            ax.plot(cps, ys, marker="o", color=colors[i % len(colors)], label=label)
        ax.set_xlabel("checkpoint (turns)")
        ax.set_ylabel("Δ accuracy vs. turn-0")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # ── Panel 1: AI-blind accuracy ───────────────────────────────────────────
    _draw_curve(axes[0, 0], "ai_blind_curve",
                f"AI-blind accuracy  (n={result['n_ai_blind']})", "accuracy")

    # ── Panel 2: AI-blind delta ──────────────────────────────────────────────
    _draw_delta(axes[0, 1], "ai_blind_curve", "AI-blind Δ accuracy")

    # ── Panel 3: Double-blind accuracy (AI wrong + doctor initially wrong) ───
    dbl_n = max((info.get("double_blind_n", 0) for _, info in policies), default=0)
    _draw_curve(axes[1, 0], "double_blind_curve",
                f"Double-blind accuracy  (n={dbl_n})\n[AI wrong AND doctor initially wrong]",
                "accuracy")

    # ── Panel 4: Double-blind delta ──────────────────────────────────────────
    _draw_delta(axes[1, 1], "double_blind_curve",
                "Double-blind Δ accuracy\n[AI wrong AND doctor initially wrong]")

    plt.tight_layout()
    plot_path = out_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--from-json", default=None,
                        help="Skip analysis; load existing ai_blind_analysis.json and plot it.")
    parser.add_argument("--n_trials", type=int, default=3,
                        help="Number of AI-alone trials per case. A case is AI-blind if wrong in ALL trials.")
    parser.add_argument("--cache", default="outputs/ai_blind_trials_cache.json",
                        help="Persistent cache: {case_id: [bool, ...]}. Re-runs skip already-done trials.")
    parser.add_argument("--out", default="outputs/ai_blind_analysis.json")
    parser.add_argument("--plot", action="store_true", help="Also save a PNG plot.")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    # Fast path: just plot an existing JSON.
    if args.from_json:
        result = json.loads(Path(args.from_json).read_text())
        plot_results(result, Path(args.from_json))
        return

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp = cfg["experiment"]
    medical_llm_cfg = cfg["plugins"]["medical_llm"]

    # Load cases
    data_dir = exp.get("data_dir")
    data_path = exp.get("data_path")
    n_cases = exp.get("n_cases")
    if data_path:
        raw = json.loads((_ROOT / data_path).read_text())
        raw_cases = raw if isinstance(raw, list) else [raw]
        if n_cases:
            raw_cases = raw_cases[:int(n_cases)]
    else:
        from scripts.run_scaling_poc import _load_balanced_cases
        raw_cases, sc = _load_balanced_cases(_ROOT / data_dir, int(n_cases))
        print(f"Specialties: {sc}")
    from core.schemas import CaseInfo
    cases = [CaseInfo(**c) for c in raw_cases]
    print(f"Loaded {len(cases)} cases.\n")

    result_paths = [Path(r) for r in args.results]

    trials = load_ai_alone_trials(
        cases, result_paths, Path(args.cache), medical_llm_cfg, args.n_trials, args.concurrency
    )

    # AI-blind = wrong in ALL n_trials trials
    blind_ids = {
        cid for cid, t in trials.items()
        if len(t) >= args.n_trials and not any(t[:args.n_trials])
    }
    total = len([c for c in cases if c.case_id in trials])
    ai_alone_acc = sum(
        1 for c in cases if any(trials.get(c.case_id, [False])[:args.n_trials])
    ) / total if total else 0.0

    print(f"\nai_alone_accuracy (≥1 correct in {args.n_trials} trials): {ai_alone_acc:.4f}")
    print(f"AI-blind (wrong in ALL {args.n_trials} trials): {len(blind_ids)}/{total} ({100*len(blind_ids)/total:.1f}%)\n")

    analysis: dict[str, dict] = {}
    for rp in result_paths:
        data = json.loads(rp.read_text())
        records = data.get("records", [])
        run_meta = data.get("run_meta", {})
        label = run_meta.get("policy_type") or rp.parent.name

        overall_acc = round(sum(r["is_correct"] for r in records) / len(records), 4) if records else 0.0
        curve = compute_curve(records, blind_ids)
        curve_double = compute_curve(records, blind_ids, also_doctor_wrong=True)

        analysis[label] = {
            "results_path": str(rp),
            "n_total_records": len(records),
            "overall_accuracy": overall_acc,
            "ai_blind_n": curve["n"],
            "ai_blind_curve": {str(k): v for k, v in curve["checkpoints"].items()},
            "double_blind_n": curve_double["n"],
            "double_blind_curve": {str(k): v for k, v in curve_double["checkpoints"].items()},
        }

        print(f"  [{label}]  overall={overall_acc:.4f}  ai_blind_n={curve['n']}  double_blind_n={curve_double['n']}")
        for tag, cv in [("ai_blind", curve), ("double_blind", curve_double)]:
            if cv["checkpoints"]:
                cps = sorted(cv["checkpoints"].keys())
                print(f"    {tag}")
                print("      cp:  " + "  ".join(f"{cp:>5}" for cp in cps))
                print("      acc: " + "  ".join(f"{cv['checkpoints'][cp]['accuracy']:>5.3f}" for cp in cps))
        print()

    result = {
        "n_trials": args.n_trials,
        "ai_alone_accuracy": round(ai_alone_acc, 4),
        "n_cases": len(cases),
        "n_ai_blind": len(blind_ids),
        "ai_blind_rate": round(len(blind_ids) / total, 4) if total else 0.0,
        "ai_blind_case_ids": sorted(blind_ids),
        "by_policy": analysis,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to {out_path}")

    if args.plot:
        plot_results(result, out_path)


if __name__ == "__main__":
    main()
