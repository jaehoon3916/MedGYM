"""Empirically calibrate cumulative-burden dropout thresholds from rollout logs.

The v4 user simulator accumulates raw NASA-TLX overall_workload scores (1-5/turn)
and can terminate with closed_by="burden_dropout" when the cumulative sum crosses
plugins.user_llm.burden_dropout_threshold. This script estimates where that
threshold should sit by replaying existing full-length rollout logs and measuring
each episode's max cognitive_burden_cumulative.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = [
    "outputs/poc_0705_react_control_v3",
    "outputs/poc_0704_oracle_a3",
    "outputs/poc_0704_user_state_oracle_a3",
    "outputs/poc_0704_naive_a3",
]


def _infer_persona(path: Path, rows: list[dict[str, Any]]) -> str:
    for row in rows:
        cfg = row.get("episode_config") or {}
        persona = cfg.get("persona")
        if persona:
            return str(persona)
    for part in path.parts:
        if part.endswith("_full"):
            return part.removesuffix("_full")
    return "unknown"


def _rollout_record(path: Path) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return None
    if not rows:
        return None

    cumulative: list[float] = []
    per_turn: list[float] = []
    final_correct = None
    closed_by = None
    for row in rows:
        closed_by = row.get("closed_by") or closed_by
        fj = row.get("final_judgement") or {}
        if "is_correct" in fj:
            final_correct = bool(fj["is_correct"])
        for turn in row.get("dialogue_history", []):
            state = turn.get("user_state")
            if not state:
                continue
            if state.get("cognitive_burden") is not None:
                per_turn.append(float(state["cognitive_burden"]))
            if state.get("cognitive_burden_cumulative") is not None:
                cumulative.append(float(state["cognitive_burden_cumulative"]))

    if not cumulative:
        return None

    policy_root = str(path.parts[1]) if len(path.parts) > 2 and path.parts[0] == "outputs" else str(path.parent)
    return {
        "path": str(path),
        "policy": policy_root,
        "persona": _infer_persona(path, rows),
        "turns": len(rows),
        "max_cumulative": max(cumulative),
        "final_cumulative": cumulative[-1],
        "mean_turn_burden": stats.mean(per_turn) if per_turn else None,
        "closed_by": closed_by,
        "is_correct": final_correct,
    }


def _quantile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[round((len(ys) - 1) * p)]


def _drop_rate(xs: list[float], threshold: float) -> float:
    return sum(x >= threshold for x in xs) / len(xs) if xs else float("nan")


def _bootstrap_quantile_ci(xs: list[float], p: float, n: int, seed: int) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [rng.choice(xs) for _ in xs]
        vals.append(_quantile(sample, p))
    vals.sort()
    return vals[round((len(vals) - 1) * 0.025)], vals[round((len(vals) - 1) * 0.975)]


def _summarize_group(
    label: str,
    records: list[dict[str, Any]],
    target_dropout: float,
    grid: list[float],
    bootstrap: int,
) -> dict[str, Any]:
    xs = sorted(float(r["max_cumulative"]) for r in records)
    q_target = 1.0 - target_dropout
    threshold = _quantile(xs, q_target)
    ci_low, ci_high = _bootstrap_quantile_ci(xs, q_target, bootstrap, seed=17 + len(label))
    return {
        "group": label,
        "n": len(xs),
        "mean": stats.mean(xs),
        "stdev": stats.pstdev(xs) if len(xs) > 1 else 0.0,
        "p50": _quantile(xs, 0.50),
        "p75": _quantile(xs, 0.75),
        "p80": _quantile(xs, 0.80),
        "p85": _quantile(xs, 0.85),
        "p90": _quantile(xs, 0.90),
        "p95": _quantile(xs, 0.95),
        "max": max(xs),
        "recommended_threshold": threshold,
        "recommended_ci_low": ci_low,
        "recommended_ci_high": ci_high,
        "target_dropout": target_dropout,
        "dropout_at_recommended": _drop_rate(xs, threshold),
        **{f"dropout_at_{g:g}": _drop_rate(xs, g) for g in grid},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--target-dropout", type=float, default=0.15)
    ap.add_argument("--grid", nargs="*", type=float, default=[18, 20, 21, 22, 23, 24, 25, 26])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out-dir", default="analysis/burden_dropout_thresholds")
    args = ap.parse_args()

    records: list[dict[str, Any]] = []
    for root in args.roots:
        for path in Path(root).glob("*_full/rollouts/*.jsonl"):
            rec = _rollout_record(path)
            if rec is not None:
                records.append(rec)

    if not records:
        raise SystemExit("No rollout records with cognitive_burden_cumulative found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_persona_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_persona[rec["persona"]].append(rec)
        by_policy[rec["policy"]].append(rec)
        by_persona_policy[(rec["persona"], rec["policy"])].append(rec)

    summary_rows = [_summarize_group("ALL", records, args.target_dropout, args.grid, args.bootstrap)]
    for persona in sorted(by_persona):
        summary_rows.append(_summarize_group(f"persona={persona}", by_persona[persona], args.target_dropout, args.grid, args.bootstrap))
    for policy in sorted(by_policy):
        summary_rows.append(_summarize_group(f"policy={policy}", by_policy[policy], args.target_dropout, args.grid, args.bootstrap))

    cross_rows = []
    persona_thresholds = {
        persona: _quantile([float(r["max_cumulative"]) for r in rs], 1.0 - args.target_dropout)
        for persona, rs in by_persona.items()
    }
    for (persona, policy), rs in sorted(by_persona_policy.items()):
        xs = [float(r["max_cumulative"]) for r in rs]
        th = persona_thresholds[persona]
        cross_rows.append({
            "persona": persona,
            "policy": policy,
            "n": len(xs),
            "mean": stats.mean(xs),
            "p75": _quantile(xs, 0.75),
            "p90": _quantile(xs, 0.90),
            "max": max(xs),
            "persona_recommended_threshold": th,
            "dropout_at_persona_threshold": _drop_rate(xs, th),
        })

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (out_dir / "persona_policy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cross_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cross_rows)

    with (out_dir / "records.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"records={len(records)} target_dropout={args.target_dropout:.0%} out={out_dir}")
    print("\nRecommended persona thresholds:")
    for persona in sorted(by_persona):
        row = next(r for r in summary_rows if r["group"] == f"persona={persona}")
        print(
            f"  {persona:<20} threshold={row['recommended_threshold']:.1f} "
            f"95%CI=[{row['recommended_ci_low']:.1f},{row['recommended_ci_high']:.1f}] "
            f"drop@22={row.get('dropout_at_22', float('nan')):.1%} "
            f"p75={row['p75']:.1f} p90={row['p90']:.1f}"
        )

    all_row = summary_rows[0]
    print(
        f"\nPooled threshold={all_row['recommended_threshold']:.1f} "
        f"95%CI=[{all_row['recommended_ci_low']:.1f},{all_row['recommended_ci_high']:.1f}]"
    )
    print("\nPolicy stress test using persona-specific thresholds:")
    for row in cross_rows:
        print(
            f"  {row['policy']:<30} {row['persona']:<20} "
            f"drop={row['dropout_at_persona_threshold']:.1%} "
            f"mean={row['mean']:.1f} p90={row['p90']:.1f}"
        )


if __name__ == "__main__":
    main()
