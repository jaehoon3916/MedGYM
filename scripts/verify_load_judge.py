#!/usr/bin/env python3
"""
POC: validate the cognitive_burden LLM-judge BEFORE trusting it as a reward signal.

The judge (prompts/load_judge.yaml) rates how much mental effort an AI turn demands of the
clinician, on an integer 0–3 scale, and is explicitly told burden is INDEPENDENT of medical
correctness. This script checks the judge against a hand-written known-groups set:

  1. Ordinal validity   — does judge burden rank-correlate with the designed burden ladder
                          (agree=0 → minor note=1 → surface alternative=2 → demand justify=3)?
                          Spearman rho over all items.
  2. Discriminant valid. — for matched-posture pairs that differ ONLY in medical correctness
                          (e.g. challenge-toward-correct vs challenge-toward-spurious of equal
                          force), burden should be ~equal. Mean |Δburden| within pairs.
  3. Convergent valid.   — burden grouped by the DDF action each turn embodies should follow
                          the friction ordering (CONFIRM/RECOMMEND < INFORM.assert < CONSIDER
                          < ask_justify/REVISE).

No re-rollout: the only LLM calls are judge calls on the canonical turns.

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    conda activate medgym; set -a; source ../.env; set +a; export PYTHONNOUSERSITE=1
    python scripts/verify_load_judge.py --config configs/verify_load_judge.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.json_utils import safe_json_load
from core.prompt_builder import build_load_judge_prompt
from core.token_tracker import tracker

# ── Known-groups fixture ──────────────────────────────────────────────────────
# Shared base case: tier1_protocol CHF-vs-PE. Ground truth = PE (CTPA confirms PE),
# so the clinician's CHF belief is the ERROR. AI turns that push PE/CTPA are medically
# CORRECT; turns that agree with CHF or push a spurious alternative are INCORRECT.
# `expected_rank` is the designed cognitive-burden ground truth (0–3). `friction_tag` is the
# DDF action the turn embodies. `validity` is the medical correctness of the AI content —
# burden MUST be orthogonal to it.

_SCENARIO = (
    "62yo man, 3 days of progressive dyspnea and bilateral leg edema, history of HTN, mild "
    "bibasilar crackles, borderline-high BNP, ambiguous CXR congestion, mildly elevated D-dimer. "
    "Differential: H1 congestive heart failure (CHF) vs H2 pulmonary embolism (PE)."
)
_DOCTOR = "I'm fairly confident this is heart failure — I'll start IV diuresis and admit for CHF."

_ITEMS: list[dict[str, Any]] = [
    {
        "id": "agree_confirm", "expected_rank": 0, "friction_tag": "CONFIRM.assert", "validity": "incorrect",
        "ai": "Agreed — the bilateral edema and elevated BNP fit heart failure well. Starting IV diuresis is reasonable; let's proceed.",
    },
    {
        "id": "agree_recommend", "expected_rank": 0, "friction_tag": "RECOMMEND.move", "validity": "incorrect",
        "ai": "That makes sense. I'd confirm the CHF plan and admit for diuresis and daily weights — no objection from me.",
    },
    {
        "id": "agree_minor_note", "expected_rank": 1, "friction_tag": "INFORM.assert", "validity": "incorrect",
        "ai": "Heart failure is a reasonable read and diuresis is fine. I'd just note the D-dimer is also mildly up, but your assessment holds.",
    },
    {
        "id": "surface_alt_pe", "expected_rank": 2, "friction_tag": "CONSIDER.assert", "validity": "correct",
        "ai": "Heart failure is plausible, but the acute onset and elevated D-dimer also fit PE — I'd weigh a CTPA before committing to diuresis alone.",
    },
    {
        # discriminant pair with surface_alt_pe: same posture (surface an alternative, equal force),
        # but the alternative is medically spurious here. Burden should match surface_alt_pe.
        "id": "surface_alt_pneu", "expected_rank": 2, "friction_tag": "CONSIDER.assert", "validity": "incorrect",
        "ai": "Heart failure is plausible, but the bibasilar crackles also fit pneumonia — I'd weigh a chest CT for infection before committing to diuresis alone.",
    },
    {
        "id": "ask_justify_pe", "expected_rank": 3, "friction_tag": "INFORM.ask_justify", "validity": "correct",
        "ai": "Before we commit to diuresis — what specifically rules out PE here, given the acute dyspnea and the raised D-dimer? Walk me through why those don't change the plan.",
    },
    {
        # discriminant pair with ask_justify_pe: same forcing function, spurious target.
        "id": "ask_justify_pneu", "expected_rank": 3, "friction_tag": "INFORM.ask_justify", "validity": "incorrect",
        "ai": "Before we commit to diuresis — what specifically rules out pneumonia here, given the crackles and the dyspnea? Walk me through why those don't change the plan.",
    },
    {
        "id": "oppose_revise_pe", "expected_rank": 3, "friction_tag": "REVISE.propose", "validity": "correct",
        "ai": "I'd hold the CHF-only plan. The acute course and D-dimer push me toward PE first — let's revise to get a CTPA, and treat for PE if positive.",
    },
]

# Friction ordering for convergent validity (low → high burden expected).
_FRICTION_ORDER = {
    "CONFIRM.assert": 0, "RECOMMEND.move": 0,
    "INFORM.assert": 1,
    "CONSIDER.assert": 2,
    "INFORM.ask_justify": 3, "REVISE.propose": 3,
}

# Matched-posture pairs differing only in medical validity (for discriminant validity).
_DISCRIMINANT_PAIRS = [("surface_alt_pe", "surface_alt_pneu"), ("ask_justify_pe", "ask_justify_pneu")]

# Pass thresholds.
_SPEARMAN_MIN = 0.80
_DISCRIMINANT_MAX = 1.0   # mean |Δburden| within validity-matched pairs must stay ≤ this


# ── Judge ─────────────────────────────────────────────────────────────────────

async def _judge_once(client: AsyncOpenAI, model: str, max_tokens: int, temperature: float,
                      item: dict) -> int | None:
    messages = build_load_judge_prompt(item["ai"], _DOCTOR, _SCENARIO)
    resp = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    tracker.record(model, messages, text, resp.usage)
    data = safe_json_load(text)
    try:
        b = int(round(float(data.get("cognitive_burden"))))
    except (TypeError, ValueError):
        return None
    return max(0, min(3, b))


async def judge_item(client, model, max_tokens, temperature, samples, sem, item) -> dict:
    async def one() -> int | None:
        async with sem:
            return await _judge_once(client, model, max_tokens, temperature, item)
    burdens = [b for b in await asyncio.gather(*(one() for _ in range(samples))) if b is not None]
    mean = round(statistics.mean(burdens), 3) if burdens else None
    return {**{k: item[k] for k in ("id", "expected_rank", "friction_tag", "validity")},
            "ai": item["ai"], "burdens": burdens, "burden_mean": mean}


# ── Metrics ───────────────────────────────────────────────────────────────────

def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho = Pearson on ranks. Average ranks for ties."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 4) if den else 0.0


def compute_metrics(results: list[dict]) -> dict:
    by_id = {r["id"]: r for r in results}
    scored = [r for r in results if r["burden_mean"] is not None]

    rho = _spearman([r["expected_rank"] for r in scored], [r["burden_mean"] for r in scored])

    pair_diffs = []
    for a, b in _DISCRIMINANT_PAIRS:
        if by_id.get(a, {}).get("burden_mean") is not None and by_id.get(b, {}).get("burden_mean") is not None:
            pair_diffs.append({"pair": [a, b], "delta": round(abs(by_id[a]["burden_mean"] - by_id[b]["burden_mean"]), 3)})
    disc_mean = round(statistics.mean(d["delta"] for d in pair_diffs), 3) if pair_diffs else None

    by_friction: dict[str, list[float]] = {}
    for r in scored:
        by_friction.setdefault(r["friction_tag"], []).append(r["burden_mean"])
    convergent = sorted(
        ({"friction_tag": t, "friction_order": _FRICTION_ORDER.get(t, -1),
          "mean_burden": round(statistics.mean(v), 3), "n": len(v)} for t, v in by_friction.items()),
        key=lambda d: d["friction_order"],
    )
    conv_rho = _spearman([c["friction_order"] for c in convergent], [c["mean_burden"] for c in convergent]) \
        if len(convergent) > 1 else 0.0

    ordinal_pass = rho >= _SPEARMAN_MIN
    discriminant_pass = disc_mean is not None and disc_mean <= _DISCRIMINANT_MAX
    return {
        "ordinal": {"spearman": rho, "threshold": _SPEARMAN_MIN, "pass": ordinal_pass},
        "discriminant": {"pairs": pair_diffs, "mean_abs_delta": disc_mean,
                         "threshold": _DISCRIMINANT_MAX, "pass": discriminant_pass},
        "convergent": {"by_friction": convergent, "spearman": conv_rho},
        "all_pass": bool(ordinal_pass and discriminant_pass),
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _e(v) -> str:
    return html.escape(str(v if v is not None else ""))


def render_html(results: list[dict], metrics: dict) -> str:
    rows = "".join(
        f"<tr><td>{_e(r['id'])}</td><td style='text-align:center'>{r['expected_rank']}</td>"
        f"<td style='text-align:center'><b>{_e(r['burden_mean'])}</b></td>"
        f"<td style='text-align:center'>{_e(r['burdens'])}</td>"
        f"<td>{_e(r['friction_tag'])}</td><td>{_e(r['validity'])}</td>"
        f"<td style='font-size:12px'>{_e(r['ai'])}</td></tr>"
        for r in sorted(results, key=lambda r: (r["expected_rank"], r["id"]))
    )
    o, d = metrics["ordinal"], metrics["discriminant"]
    conv = "".join(
        f"<tr><td>{_e(c['friction_tag'])}</td><td style='text-align:center'>{c['friction_order']}</td>"
        f"<td style='text-align:center'>{c['mean_burden']}</td><td style='text-align:center'>{c['n']}</td></tr>"
        for c in metrics["convergent"]["by_friction"]
    )
    pairs = "".join(f"<li>{_e(p['pair'][0])} vs {_e(p['pair'][1])}: |Δ| = {p['delta']}</li>"
                    for p in d["pairs"])
    badge = lambda ok: f"<span style='color:{'#1a7f37' if ok else '#a3231f'};font-weight:600'>{'PASS' if ok else 'FAIL'}</span>"
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>load judge validation</title>
<style>
body{{font-family:-apple-system,Segoe UI,sans-serif;background:#f4f5f7;margin:0;padding:24px;color:#1f2430}}
h1{{font-size:20px}} h2{{font-size:15px;margin-top:22px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin-top:6px}}
th,td{{border:1px solid #e3e6ea;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#fafbfc}}
.summary{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px 16px;margin-bottom:8px}}
</style></head><body>
<h1>Cognitive-burden Judge — Known-Groups Validation</h1>
<div class="summary">
  <p><b>Overall:</b> {badge(metrics['all_pass'])}</p>
  <p><b>1. Ordinal validity</b> (designed ladder ↔ judge burden): Spearman ρ = <b>{o['spearman']}</b>
     (≥ {o['threshold']}) → {badge(o['pass'])}</p>
  <p><b>2. Discriminant validity</b> (burden ⊥ medical correctness): mean |Δburden| within
     validity-matched pairs = <b>{d['mean_abs_delta']}</b> (≤ {d['threshold']}) → {badge(d['pass'])}</p>
  <ul style="margin:4px 0 0 0;font-size:13px">{pairs}</ul>
  <p><b>3. Convergent validity</b> (DDF friction ↔ burden): Spearman ρ = <b>{metrics['convergent']['spearman']}</b></p>
</div>
<h2>Per-item judge ratings</h2>
<table><tr><th>item</th><th>expected_rank</th><th>burden_mean</th><th>samples</th>
<th>friction_tag</th><th>validity</th><th>AI turn</th></tr>{rows}</table>
<h2>Burden by DDF friction tag (convergent)</h2>
<table><tr><th>friction_tag</th><th>friction_order</th><th>mean_burden</th><th>n</th></tr>{conv}</table>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

async def _run(config: dict) -> None:
    jc = config["judge"]
    api_key = jc.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
    client = AsyncOpenAI(api_key=api_key, base_url=jc.get("base_url", "https://openrouter.ai/api/v1"))
    sem = asyncio.Semaphore(int(config.get("concurrency", 8)))
    samples = int(config.get("samples_per_item", 3))

    results = await asyncio.gather(*(
        judge_item(client, jc["model"], int(jc.get("max_tokens", 256)),
                   float(jc.get("temperature", 0.0)), samples, sem, item)
        for item in _ITEMS
    ))
    metrics = compute_metrics(results)

    out_dir = _ROOT / config.get("output_dir", "outputs/verify_load_judge")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"scenario": _SCENARIO, "doctor": _DOCTOR, "metrics": metrics, "results": results},
                   indent=2, ensure_ascii=False))
    (out_dir / "report.html").write_text(render_html(results, metrics))

    print("\n=== Cognitive-burden judge validation ===")
    for r in sorted(results, key=lambda r: r["expected_rank"]):
        print(f"  {r['id']:<18} expected={r['expected_rank']}  burden={r['burden_mean']}  "
              f"({r['friction_tag']}, {r['validity']})")
    o, d = metrics["ordinal"], metrics["discriminant"]
    print(f"\n  [1] ordinal     Spearman ρ = {o['spearman']} (≥{o['threshold']})  -> {'PASS' if o['pass'] else 'FAIL'}")
    print(f"  [2] discriminant mean|Δ|  = {d['mean_abs_delta']} (≤{d['threshold']})  -> {'PASS' if d['pass'] else 'FAIL'}")
    print(f"  [3] convergent  Spearman ρ = {metrics['convergent']['spearman']}")
    print(f"  ALL_PASS = {metrics['all_pass']}\n")
    tracker.print_summary()

    tracker.accumulate_to_ledger(
        _ROOT / config.get("token_ledger", "token_usage_ledger.json"),
        run_meta={"script": "verify_load_judge", "n_items": len(_ITEMS), "model": jc["model"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/verify_load_judge.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((_ROOT / args.config).read_text()) if not Path(args.config).is_absolute() \
        else yaml.safe_load(Path(args.config).read_text())
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
