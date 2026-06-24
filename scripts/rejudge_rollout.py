#!/usr/bin/env python3
"""
Re-judge a REAL multi-turn rollout (not a synthetic fixture) with the v3 NASA-TLX burden
judge, turn by turn, and compare the resulting cumulative-burden trajectory against the OLD
burden values already embedded in that rollout (from whichever v1.py judge produced it).

No regeneration: the AI utterances are read verbatim from the rollout file. Only judge calls
are made (turns x samples).

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    conda activate medgym; set -a; source ../.env; set +a; export PYTHONNOUSERSITE=1
    python scripts/rejudge_rollout.py \
        --rollout outputs/scaling_poc_persona_heuristic_delib/burned_out_resident_full/rollouts/6001.jsonl \
        --samples 3
"""
from __future__ import annotations

import argparse
import asyncio
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

from core.token_tracker import tracker
from plugins.user_llm.user_simulator.v3_burden import (
    _TLX_DIMS,
    _normalize_overall_workload,
    _parse_burden_judge,
    build_burden_judge_prompt_v3,
)

_VALID_PERSONAS = ("veteran_attending", "exhausted_attending", "eager_resident", "burned_out_resident")
# Old v1 burden is on a 0-3 raw scale (pre-dating today's r*q_t_prior effective-burden
# formula in some older runs, or already-effective in newer ones -- either way the ceiling is
# the same /3.0 normalization v1.py has always used for "burden contributed this turn").
_OLD_BURDEN_SCALE_MAX = 3.0


def _load_personas() -> dict:
    personas: dict = {}
    for path in sorted((_ROOT / "source" / "persona").glob("persona_*.yaml")):
        with open(path) as f:
            personas.update(yaml.safe_load(f))
    return personas


def _persona_from_folder(rollout_path: Path) -> str:
    """The rollout's grandparent folder is named like "burned_out_resident_full" -- match
    against the 4 known persona names by prefix (folder name = persona + suffix)."""
    folder = rollout_path.parent.parent.name
    for p in _VALID_PERSONAS:
        if folder.startswith(p):
            return p
    raise ValueError(f"could not derive a known persona from folder name '{folder}' "
                      f"(expected one of {_VALID_PERSONAS} as a prefix)")


def _fmt_history(turns: list[dict[str, Any]]) -> str:
    """Same line-prefix convention as plugins/user_llm/user_simulator/v3.py's _fmt_history."""
    if not turns:
        return "(No conversation yet)"
    lines = []
    for t in turns:
        role = "AI Assistant" if t["speaker"] == "medical" else "Doctor"
        lines.append(f"[{role}]: {t['text']}")
    return "\n".join(lines)


def load_rollout_dialogue(rollout_path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Reads a scripts/run_scaling_poc.py-style rollout JSONL (one line per step). The LAST
    line's `dialogue_history` is the fullest accumulated transcript for the whole episode --
    earlier lines are just prefixes of it. Returns (scenario, full_dialogue_history)."""
    lines = [json.loads(l) for l in rollout_path.read_text().splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"{rollout_path} has no turns")
    scenario = lines[0]["case_info"]["scenario"]
    full_history = lines[-1]["dialogue_history"]
    return scenario, full_history


def extract_ai_turns(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each `speaker == "medical"` entry, pair it with (a) the prior turns as judge
    context and (b) the OLD burden value v1.py recorded in the immediately FOLLOWING "user"
    turn's user_state (that's where v1's _followup_turn puts the burden it just scored for
    THIS AI turn -- see plugins/user_llm/user_simulator/v1.py's _score_burden docstring).
    Turns with no following user turn (episode ended right after this AI turn) or no
    recorded cognitive_burden (e.g. the very first AI turn if something went wrong) get
    old_burden_raw=None."""
    out = []
    for i, turn in enumerate(history):
        if turn["speaker"] != "medical":
            continue
        old_burden_raw = None
        if i + 1 < len(history) and history[i + 1]["speaker"] == "user":
            us = history[i + 1].get("user_state") or {}
            old_burden_raw = us.get("cognitive_burden")
        out.append({
            "ai_turn_index": len(out),  # 0-based index among AI turns only
            "dialogue_text_before": _fmt_history(history[:i]),
            "ai_utterance": turn["text"],
            "old_burden_raw": old_burden_raw,
        })
    return out


async def judge_turn(
    client: AsyncOpenAI, model: str, max_tokens: int, temperature: float, samples: int,
    sem: asyncio.Semaphore, tmpl: dict, scenario: str, persona_instruction: str,
    dialogue_text: str, ai_utterance: str, cumulative_burden: float,
) -> dict:
    messages = build_burden_judge_prompt_v3(
        tmpl=tmpl, scenario=scenario, persona_instruction=persona_instruction,
        dialogue_text=dialogue_text, ai_utterance=ai_utterance, cumulative_burden=cumulative_burden,
    )

    async def one() -> dict | None:
        async with sem:
            resp = await client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        text = resp.choices[0].message.content or ""
        tracker.record(model, messages, text, resp.usage)
        return _parse_burden_judge(text)

    parsed = [p for p in await asyncio.gather(*(one() for _ in range(samples))) if p is not None]
    if not parsed:
        return {"overall_workload": None, "dims": None, "short_rationale": None, "n_ok": 0, "n_attempted": samples}
    overall = round(statistics.mean(p["overall_workload"] for p in parsed), 3)
    dims = {d: round(statistics.mean(p[d] for p in parsed), 3) for d in _TLX_DIMS}
    return {
        "overall_workload": overall, "dims": dims,
        "short_rationale": parsed[0]["short_rationale"],  # representative sample, illustrative only
        "n_ok": len(parsed), "n_attempted": samples,
    }


def _load_v3_prompts() -> dict:
    with open(_ROOT / "prompts" / "user_simulator_v3.yaml") as f:
        return yaml.safe_load(f)


async def _run(args) -> None:
    rollout_path = Path(args.rollout) if Path(args.rollout).is_absolute() else _ROOT / args.rollout
    persona = args.persona or _persona_from_folder(rollout_path)
    scenario, history = load_rollout_dialogue(rollout_path)
    ai_turns = extract_ai_turns(history)

    print(f"=== Run parameters ===")
    print(f"  rollout      = {rollout_path}")
    print(f"  persona      = {persona}")
    print(f"  judge model  = {args.model} (temperature={args.temperature}, max_tokens={args.max_tokens})")
    print(f"  samples/turn = {args.samples}")
    print(f"  {len(ai_turns)} AI turns found -> {len(ai_turns) * args.samples} new judge calls")

    persona_instruction = _load_personas()[persona]
    tmpl = _load_v3_prompts()
    api_key = os.environ.get("OPENROUTER_API_KEY", "EMPTY")
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.concurrency)

    rows = []
    new_cumulative = 0.0
    old_cumulative = 0.0
    for at in ai_turns:
        judged = await judge_turn(
            client, args.model, args.max_tokens, args.temperature, args.samples, sem, tmpl,
            scenario, persona_instruction, at["dialogue_text_before"], at["ai_utterance"], new_cumulative,
        )
        new_norm = _normalize_overall_workload(judged["overall_workload"]) if judged["overall_workload"] is not None else None
        old_raw = at["old_burden_raw"]
        old_norm = min(1.0, max(0.0, old_raw / _OLD_BURDEN_SCALE_MAX)) if old_raw is not None else None

        if new_norm is not None:
            new_cumulative += new_norm
        if old_norm is not None:
            old_cumulative += old_norm

        rows.append({
            "ai_turn_index": at["ai_turn_index"], "ai_utterance": at["ai_utterance"],
            "old_burden_raw": old_raw, "old_burden_norm": old_norm, "old_cumulative": round(old_cumulative, 4) if old_norm is not None else None,
            "new_overall_workload": judged["overall_workload"], "new_dims": judged["dims"],
            "new_burden_norm": new_norm, "new_cumulative": round(new_cumulative, 4) if new_norm is not None else None,
            "new_short_rationale": judged["short_rationale"],
            "n_judge_ok": judged["n_ok"], "n_judge_attempted": judged["n_attempted"],
        })

    case_id_match = rollout_path.stem
    out_dir = rollout_path.parent.parent
    out_json = out_dir / f"rejudge_{case_id_match}.json"
    out_json.write_text(json.dumps({
        "rollout": str(rollout_path), "persona": persona, "scenario": scenario,
        "judge_model": args.model, "samples_per_turn": args.samples, "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_json}")

    print("\n=== Per-turn burden (old raw -> [0,1]) vs (new NASA-TLX -> [0,1]) ===")
    for r in rows:
        print(f"  turn {r['ai_turn_index']:<2} old_norm={r['old_burden_norm']}  new_norm={r['new_burden_norm']}  "
              f"old_cum={r['old_cumulative']}  new_cum={r['new_cumulative']}")
    tracker.print_summary()
    tracker.accumulate_to_ledger(
        _ROOT / "token_usage_ledger.json",
        run_meta={"script": "rejudge_rollout", "rollout": str(rollout_path), "n_ai_turns": len(ai_turns), "model": args.model},
    )

    _plot(rows, out_dir / f"rejudge_{case_id_match}.png", case_id_match, persona)


def _plot(rows: list[dict], out_path: Path, case_id: str, persona: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["ai_turn_index"] + 1 for r in rows]  # 1-indexed turn count, matches plot_burden's "checkpoint (turns)"
    old_ys = [r["old_cumulative"] for r in rows]
    new_ys = [r["new_cumulative"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(y is not None for y in old_ys):
        ax.plot(xs, old_ys, marker="o", linestyle="--", label="old (v1 judge, 0-3 raw -> [0,1])")
    if any(y is not None for y in new_ys):
        ax.plot(xs, new_ys, marker="o", linestyle="-", label="new (v3 NASA-TLX judge -> [0,1])")
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("cumulative burden (normalized [0,1])")
    ax.set_title(f"Cognitive burden vs. turn checkpoint -- single-episode rejudge\n"
                 f"case_id={case_id}, persona={persona}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--persona", default=None, help="Override auto-derived persona")
    parser.add_argument("--model", default="deepseek/deepseek-v3.2")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
