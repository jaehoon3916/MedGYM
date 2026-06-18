#!/usr/bin/env python3
"""
Plot poc_medcobe_hac.py results: grouped bar chart of
recall_correction / recall_confirmation / medcobe_score
across baseline / hac_command / hac_reference.

Usage:
    python scripts/plot_poc_medcobe_hac.py --results outputs/poc_medcobe_hac/results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_CONDITIONS = ("baseline", "hac_command", "hac_reference")
_LABELS = {"baseline": "Baseline", "hac_command": "HAC-command", "hac_reference": "HAC-reference"}
_COLORS = {"baseline": "#9aa5b1", "hac_command": "#E87040", "hac_reference": "#4393c3"}
_METRICS = ("recall_correction", "recall_confirmation", "medcobe_score")
_METRIC_LABELS = {"recall_correction": "recall_corr", "recall_confirmation": "recall_conf", "medcobe_score": "MedCOBE"}


def main(results_path: str, output_path: str | None) -> None:
    data = json.loads(Path(results_path).read_text())
    scores = data["scores"]

    x = np.arange(len(_METRICS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cond in enumerate(_CONDITIONS):
        values = [scores[cond][m] for m in _METRICS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=_LABELS[cond], color=_COLORS[cond])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS[m] for m in _METRICS])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    n_error = scores["baseline"]["n_error"]
    n_correct = scores["baseline"]["n_correct"]
    ax.set_title(f"MedCOBE POC: Baseline vs HAC (command vs reference)\n"
                 f"n={n_error + n_correct}  ({n_error} error + {n_correct} correct)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    out = Path(output_path) if output_path else Path(results_path).parent / "plot.png"
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/poc_medcobe_hac/results.json", help="Path to results.json")
    parser.add_argument("--output", default=None, help="Output image path (default: <results_dir>/plot.png)")
    args = parser.parse_args()
    main(args.results, args.output)
