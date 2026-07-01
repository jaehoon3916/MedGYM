#!/usr/bin/env python3
"""
Plot AI-knows vs double-blind accuracy curves for naive / medcobe_feedback / deliberation_llm.

Usage:
    python scripts/plot_ai_blind.py --out outputs/ai_blind_comparison.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

RUNS = {
    "naive":            "outputs/poc_0630_naive/veteran_attending_full/results.json",
    "medcobe_feedback": "outputs/poc_0630_medcobe_feedback/veteran_attending_full/results.json",
    "deliberation_llm": "outputs/poc_0630_deliberation_llm/veteran_attending_full/results.json",
}
CACHE = "outputs/ai_blind_trials_cache.json"
COLORS = {"naive": "C0", "medcobe_feedback": "C1", "deliberation_llm": "C2"}


def get_curve(records: list[dict], case_ids: set[str]) -> tuple[list[int], list[float], int]:
    subset = [r for r in records if r["case_id"] in case_ids]
    if not subset:
        return [], [], 0
    cps_ref = subset[0].get("checkpoints") or subset[0].get("checkpoint_results", {})
    cps = sorted(int(k) for k in cps_ref)
    accs = []
    for c in cps:
        corrects = [
            bool((r.get("checkpoints") or r.get("checkpoint_results", {})).get(str(c), {}).get("is_correct"))
            for r in subset
        ]
        accs.append(sum(corrects) / len(corrects))
    return cps, accs, len(subset)


def main(out: str) -> None:
    cache = json.loads(Path(CACHE).read_text())
    blind_ids  = {cid for cid, t in cache.items() if len(t) >= 3 and not any(t[:3])}
    ai_knows_ids = {cid for cid, t in cache.items() if len(t) >= 3 and any(t[:3])}

    data: dict[str, list[dict]] = {
        label: json.loads(Path(path).read_text())["records"]
        for label, path in RUNS.items()
    }
    shared = set.intersection(*(set(r["case_id"] for r in recs) for recs in data.values()))

    # double_blind: AI blind AND doctor initially wrong
    double_blind_ids = {
        cid for cid in shared & blind_ids
        if all(
            not next((r for r in recs if r["case_id"] == cid), {}).get("alone_correct", True)
            for recs in data.values()
        )
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"AI-knows vs Double-blind  (shared n={len(shared)}, "
        f"ai_knows={len(shared & ai_knows_ids)}, double_blind={len(double_blind_ids)})",
        fontsize=11,
    )

    panels = [
        (axes[0], ai_knows_ids & shared,  "AI-knows accuracy\n(AI ≥1/3 trials correct)"),
        (axes[1], double_blind_ids,        "Double-blind accuracy\n(AI wrong AND doctor initially wrong)"),
    ]

    for ax, id_set, title in panels:
        for label, records in data.items():
            cps, accs, n = get_curve(records, id_set)
            if not cps:
                continue
            color = COLORS[label]
            ax.plot(cps, accs, marker="o", color=color, label=f"{label} (n={n})")
        ax.axhline(0, color="grey", linestyle=":", alpha=0.5)
        ax.set_xlabel("checkpoint (turns)")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel 3: delta bar (turn-0 → final turn)
    ax = axes[2]
    labels = list(RUNS.keys())
    x = range(len(labels))
    w = 0.35

    knows_deltas, blind_deltas = [], []
    for label, records in data.items():
        for id_set, lst in [(ai_knows_ids & shared, knows_deltas), (double_blind_ids, blind_deltas)]:
            cps, accs, n = get_curve(records, id_set)
            if cps:
                start = accs[0] if cps[0] != 0 else accs[0]
                delta = accs[-1] - start
                lst.append(delta)
            else:
                lst.append(0.0)

    bars1 = ax.bar([xi - w / 2 for xi in x], knows_deltas, w, label="AI-knows Δ", color=[COLORS[l] for l in labels], alpha=0.9)
    bars2 = ax.bar([xi + w / 2 for xi in x], blind_deltas, w, label="double-blind Δ", color=[COLORS[l] for l in labels], alpha=0.5, hatch="//")
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 if h >= 0 else h - 0.02,
                f"{h:+.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Δ accuracy (final − turn-0)")
    ax.set_title("Gain: AI-knows (solid) vs double-blind (hatched)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/ai_blind_comparison.png")
    main(p.parse_args().out)
