#!/usr/bin/env python3
"""
Sanity check: does r_align (the hand-crafted process-shaping reward the oracle policy
argmaxes -- core/reward_align.py) actually correlate with final correctness (final_judge
vs. gold), or could "oracle didn't converge" in the scaling POC just mean r_align is a bad
proxy for the real objective?

r_align.txt is explicit that r_final (whether the user reaches gold) is the actual goal,
and r_align is only a dense process-shaping signal for HOW to get there:

    "핵심 목표(정답)는 r_final이 담당. r_align은 '어떻게 그 정답으로 가느냐'의 dense 신호."

That design intent was never empirically checked against outcomes -- scripts/verify_reward.py
only verifies the *implementation* matches the written BASE table (r_align.txt), not that
the table is actually a good proxy for reaching the correct answer. Since the oracle policy
(plugins/policy/oracle_policy.py) argmaxes r_align rather than final correctness, "the oracle
didn't converge" in scripts/run_scaling_poc.py could mean either (a) longer dialogue doesn't
help, or (b) r_align just doesn't track the thing we actually care about -- this script can't
distinguish those without checking r_align's own validity first.

This checks that validity directly: across recorded episodes, does higher accumulated
r_align correlate with the dialogue having reached the correct answer (per final_judge) at
that point? No LLM calls -- reads an existing scripts/run_scaling_poc.py results.json
(requires its r_align/mean_r_align per-checkpoint fields, added alongside this script).

Usage:
    python scripts/analyze_r_align_validity.py --results outputs/scaling_poc/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho = Pearson on ranks. Average ranks for ties.

    Same helper as scripts/verify_load_judge.py's _spearman -- duplicated rather than
    imported since it's a tiny, self-contained, project-wide convention (no scipy/numpy dep).
    """
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


def load_pairs(data: dict) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """Returns (pooled, final_only) lists of (mean_r_align, is_correct as 0/1).

    Checkpoint 0 is excluded -- r_align is trivially 0 there (no AI turn happened yet),
    which isn't informative about the proxy's validity.
    """
    pooled: list[tuple[float, int]] = []
    final_only: list[tuple[float, int]] = []
    checkpoints = data["run_meta"]["checkpoints"]
    last_cp = str(checkpoints[-1])
    for rec in data["records"]:
        cps = rec["checkpoints"]
        for c_str, cp in cps.items():
            if c_str == "0":
                continue
            pooled.append((cp["mean_r_align"], int(cp["is_correct"])))
        if last_cp in cps:
            final_only.append((cps[last_cp]["mean_r_align"], int(cps[last_cp]["is_correct"])))
    return pooled, final_only


def main(results_path: str) -> None:
    data = json.loads(Path(results_path).read_text())
    pooled, final_only = load_pairs(data)

    print(f"Loaded {len(data['records'])} episode(s) from {results_path}")
    print()

    for label, pairs in (("pooled (all checkpoints x cases)", pooled), ("final checkpoint only", final_only)):
        if len(pairs) < 3:
            print(f"  [{label}] n={len(pairs)} -- too few points to compute a correlation meaningfully")
            continue
        xs = [p[0] for p in pairs]
        ys = [float(p[1]) for p in pairs]
        rho = _spearman(xs, ys)
        n_correct = sum(p[1] for p in pairs)
        print(f"  [{label}] n={len(pairs)}  correct={n_correct}/{len(pairs)}  "
              f"Spearman(mean_r_align, is_correct) = {rho}")

    print()
    print("  Interpretation:")
    print("  - rho near 0 or negative: r_align is a WEAK/bad proxy for final correctness --")
    print("    an oracle that perfectly maximizes r_align is not necessarily converging on")
    print("    truth, so 'oracle didn't converge' in the scaling POC would NOT cleanly mean")
    print("    'dialogue length doesn't help' -- it could just mean the reward table is")
    print("    miscalibrated as a proxy.")
    print("  - rho clearly positive: maximizing r_align is at least directionally aligned")
    print("    with reaching the right answer, so treating the oracle's behavior as a")
    print("    trustworthy ceiling for the scaling question is more defensible.")
    print("  - With this few episodes this is a coarse diagnostic, not a significance test --")
    print("    re-run with a larger n_cases before trusting the sign/magnitude strongly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", default="outputs/scaling_poc/results.json",
        help="Path to run_scaling_poc.py's results.json",
    )
    args = parser.parse_args()
    main(args.results)
