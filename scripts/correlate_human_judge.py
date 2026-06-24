#!/usr/bin/env python3
"""
Compares the v3 NASA-TLX judge's per-turn scores (from scripts/rejudge_rollout.py's
rejudge_<case_id>.json) against a human's own per-turn Likert scores (from app.py's
/annotate UI, human_annotations_<case_id>.json) -- Spearman rho per dimension + overall.

No LLM calls -- pure computation over two already-saved JSON files. Only uses turns the human
has actually scored so far (partial annotation is fine, reports how many that is).

Usage:
    python scripts/correlate_human_judge.py \
        --rollout outputs/scaling_poc_persona_heuristic_delib/burned_out_resident_full/rollouts/6001.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from plugins.user_llm.user_simulator.v3_burden import _TLX_DIMS


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho = Pearson on ranks. Average ranks for ties. Same implementation as
    scripts/verify_load_judge_v3.py's _spearman -- duplicated, not shared, per this
    codebase's existing per-script convention for this small helper."""
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


def compute_correlation(rejudge: dict, annotations: dict) -> dict:
    """rejudge: scripts/rejudge_rollout.py's saved dict (has "rows", each with
    "ai_turn_index"/"new_dims"/"new_overall_workload"). annotations: app.py's saved dict
    {"<ai_turn_index>": {dim: 1-5, ...}}."""
    rows_by_idx = {r["ai_turn_index"]: r for r in rejudge["rows"]}
    scored_idxs = sorted(int(k) for k in annotations.keys())
    matched_idxs = [i for i in scored_idxs if i in rows_by_idx and rows_by_idx[i]["new_dims"] is not None]

    n_total_ai_turns = len(rejudge["rows"])
    n_human_scored = len(scored_idxs)
    n_matched = len(matched_idxs)

    per_dim_rho = {}
    for dim in _TLX_DIMS:
        judge_vals = [rows_by_idx[i]["new_dims"][dim] for i in matched_idxs]
        human_vals = [annotations[str(i)][dim] for i in matched_idxs]
        per_dim_rho[dim] = _spearman(judge_vals, human_vals) if n_matched > 1 else None

    judge_overall = [rows_by_idx[i]["new_overall_workload"] for i in matched_idxs]
    human_overall = [
        sum(annotations[str(i)][d] for d in _TLX_DIMS) / len(_TLX_DIMS) for i in matched_idxs
    ]
    overall_rho = _spearman(judge_overall, human_overall) if n_matched > 1 else None

    return {
        "n_total_ai_turns": n_total_ai_turns, "n_human_scored": n_human_scored, "n_matched": n_matched,
        "per_dim_spearman": per_dim_rho, "overall_spearman": overall_rho,
        "matched_turn_indices": matched_idxs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True)
    args = parser.parse_args()

    rollout_path = Path(args.rollout) if Path(args.rollout).is_absolute() else _ROOT / args.rollout
    case_id = rollout_path.stem
    exp_dir = rollout_path.parent.parent

    rejudge_path = exp_dir / f"rejudge_{case_id}.json"
    annotations_path = exp_dir / f"human_annotations_{case_id}.json"

    if not rejudge_path.exists():
        print(f"No {rejudge_path} -- run scripts/rejudge_rollout.py first.")
        return
    if not annotations_path.exists():
        print(f"No {annotations_path} -- score at least one turn via the /annotate UI first "
              f"(python app.py, then open /annotate/{rollout_path.relative_to(_ROOT / 'outputs')}).")
        return

    rejudge = json.loads(rejudge_path.read_text())
    annotations = json.loads(annotations_path.read_text())
    result = compute_correlation(rejudge, annotations)

    print(f"=== Human vs judge correlation -- case {case_id} ===")
    print(f"  {result['n_human_scored']}/{result['n_total_ai_turns']} AI turns scored by human "
          f"({result['n_matched']} matched against a successfully-judged turn)")
    if result["n_matched"] < 2:
        print("  Not enough matched turns yet for a meaningful Spearman rho (need >= 2). "
              "Score more turns via /annotate and re-run.")
    for dim, rho in result["per_dim_spearman"].items():
        print(f"  {dim:<16} rho = {rho}")
    print(f"  {'overall_workload':<16} rho = {result['overall_spearman']}")

    out_path = exp_dir / f"human_judge_correlation_{case_id}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
