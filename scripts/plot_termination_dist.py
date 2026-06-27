#!/usr/bin/env python3
"""
종료 turn 분포 시각화 — termination_poc / termination_poc_medcobe_feedback

Usage:
    python scripts/plot_termination_dist.py
    python scripts/plot_termination_dist.py --exp termination_poc --out plot/result/termination_dist.png
"""
from __future__ import annotations

import argparse
import json
import glob
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PERSONAS = [
    "veteran_attending_full",
    "exhausted_attending_full",
    "burned_out_resident_full",
    "eager_resident_full",
]
PERSONA_LABELS = {
    "veteran_attending_full": "Veteran Attending",
    "exhausted_attending_full": "Exhausted Attending",
    "burned_out_resident_full": "Burned-out Resident",
    "eager_resident_full": "Eager Resident",
    "_overall_": "Overall",
}
COLORS = {
    "veteran_attending_full": "#4393c3",
    "exhausted_attending_full": "#d6604d",
    "burned_out_resident_full": "#878787",
    "eager_resident_full": "#4dac26",
    "_overall_": "#1a1a1a",
}


def load_turn_dist(base: str) -> dict[str, Counter]:
    """각 persona(+ overall)별 최종 turn_id Counter 반환."""
    result: dict[str, Counter] = {}
    overall: Counter = Counter()

    for persona in PERSONAS:
        files = sorted(glob.glob(os.path.join(base, persona, "rollouts", "*.jsonl")))
        if not files:
            continue
        c: Counter = Counter()
        for f in files:
            with open(f) as fp:
                lines = [json.loads(l) for l in fp if l.strip()]
            if lines:
                t = lines[-1].get("turn_id", 0)
                c[t] += 1
                overall[t] += 1
        result[persona] = c

    if overall:
        result["_overall_"] = overall
    return result


def _draw_bar(ax: plt.Axes, c: Counter, turns: list[int], color: str, label: str) -> None:
    total = sum(c.values())
    freqs = [c.get(t, 0) / total * 100 for t in turns]
    avg = sum(t * c[t] for t in c) / total if total else 0

    bars = ax.bar(turns, freqs, color=color, alpha=0.85, width=0.6, zorder=3)
    ax.axvline(avg, color="black", linestyle="--", linewidth=1.3, zorder=4, label=f"avg={avg:.2f}")

    for bar, freq in zip(bars, freqs):
        if freq > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{freq:.0f}%",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xlabel("Final Turn", fontsize=8)
    ax.set_xticks(turns)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, max(freqs) * 1.25 + 5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(labelsize=8)


def plot(base: str, out_path: str) -> None:
    dist = load_turn_dist(base)
    if not dist:
        raise RuntimeError(f"rollout 파일을 찾을 수 없습니다: {base}")

    all_turns = sorted({t for c in dist.values() for t in c})
    turns = list(range(max(all_turns) + 1))
    persona_keys = [p for p in PERSONAS if p in dist]
    n = len(persona_keys)

    exp_label = Path(base).name.replace("_", " ")

    # 위: persona 4개 / 아래: overall (가운데 정렬)
    fig = plt.figure(figsize=(3.2 * n, 8))
    fig.suptitle(f"Termination Turn Distribution\n({exp_label})", fontsize=11, fontweight="bold")

    # 위쪽 row — persona
    top_axes = [fig.add_subplot(2, n, i + 1) for i in range(n)]
    for i, (ax, key) in enumerate(zip(top_axes, persona_keys)):
        _draw_bar(ax, dist[key], turns, COLORS[key], PERSONA_LABELS[key])
        if i == 0:
            ax.set_ylabel("% of Episodes", fontsize=8)

    # 아래쪽 row — overall (중앙 2칸 차지)
    mid = n // 2
    ax_overall = fig.add_subplot(2, n, n + mid)  # 중앙 셀
    if "_overall_" in dist:
        _draw_bar(ax_overall, dist["_overall_"], turns, COLORS["_overall_"], PERSONA_LABELS["_overall_"])
        ax_overall.set_ylabel("% of Episodes", fontsize=8)
        for spine in ax_overall.spines.values():
            spine.set_linewidth(1.8)
            spine.set_edgecolor("#111111")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="termination_poc",
        help="outputs/ 아래 실험 폴더명 (default: termination_poc)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="출력 PNG 경로 (default: plot/result/<exp>_turn_dist.png)",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="outputs 루트 경로 (default: 스크립트 기준 ../outputs)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    base_dir = Path(args.base_dir) if args.base_dir else script_dir.parent / "outputs"
    exp_base = base_dir / args.exp
    out_path = args.out or str(script_dir.parent / "plot" / "result" / f"{args.exp}_turn_dist.png")

    plot(str(exp_base), out_path)


if __name__ == "__main__":
    main()
