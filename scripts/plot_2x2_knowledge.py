#!/usr/bin/env python3
"""
2×2 knowledge quadrant analysis.

Axes:
  Doctor knows  = alone_correct=True in ALL 3 runs (naive/medcobe/deliberation_llm)
  Doctor blind  = alone_correct=False in ALL 3 runs
  AI knows      = ≥1 correct in 3 ai_alone trials (cache)
  AI blind      = wrong in ALL 3 ai_alone trials (cache)

4 panels, one per quadrant. Each panel: per-checkpoint accuracy curves for
naive / medcobe_feedback / deliberation_llm.

Usage:
    python scripts/plot_2x2_knowledge.py --out outputs/2x2_knowledge.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

RUNS = {
    "naive":            "outputs/poc_0630_naive/veteran_attending_full/results.json",
    "medcobe_feedback": "outputs/poc_0630_medcobe_feedback/veteran_attending_full/results.json",
    "deliberation_llm": "outputs/poc_0630_deliberation_llm/veteran_attending_full/results.json",
}
CACHE = "outputs/ai_blind_trials_cache.json"
COLORS = {"naive": "C0", "medcobe_feedback": "C1", "deliberation_llm": "C2"}


def get_curve(records_by_id: dict, case_ids: set[str]) -> tuple[list[int], list[float], int]:
    subset = [records_by_id[cid] for cid in case_ids if cid in records_by_id]
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


def draw_panel(ax, label: str, records_by_id: dict[str, dict[str, list]], case_ids: set[str]) -> None:
    has_data = False
    for policy, recs in records_by_id.items():
        cps, accs, n = get_curve(recs, case_ids)
        if not cps:
            continue
        has_data = True
        ax.plot(cps, accs, marker="o", color=COLORS[policy], label=f"{policy} (n={n})", linewidth=1.5)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.set_xlabel("checkpoint (turns)", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=7)
    if not has_data:
        ax.text(0.5, 0.5, "no cases", ha="center", va="center", transform=ax.transAxes, color="grey")


def main(out: str) -> None:
    # ── Load AI-alone trial results ──────────────────────────────────────────
    cache = json.loads(Path(CACHE).read_text())
    ai_knows_ids = {cid for cid, t in cache.items() if len(t) >= 3 and any(t[:3])}
    ai_blind_ids  = {cid for cid, t in cache.items() if len(t) >= 3 and not any(t[:3])}

    # ── Load run records ─────────────────────────────────────────────────────
    data: dict[str, dict[str, dict]] = {}
    for policy, path in RUNS.items():
        recs = json.loads(Path(path).read_text())["records"]
        data[policy] = {r["case_id"]: r for r in recs}

    shared = set.intersection(*(set(d.keys()) for d in data.values()))

    # ── Doctor alone_correct: use 3 runs as 3 independent trials ────────────
    # doctor_knows = True in ALL 3 runs; doctor_blind = False in ALL 3 runs
    doctor_knows_ids: set[str] = set()
    doctor_blind_ids: set[str] = set()
    for cid in shared:
        trials = [data[p][cid].get("alone_correct", False) for p in RUNS]
        if all(trials):
            doctor_knows_ids.add(cid)
        elif not any(trials):
            doctor_blind_ids.add(cid)
    # mixed doctor cases (not all-agree) are excluded from quadrant analysis

    # ── 4 quadrants ──────────────────────────────────────────────────────────
    quadrants = {
        ("Doctor knows", "AI knows"):   doctor_knows_ids & ai_knows_ids & shared,
        ("Doctor knows", "AI blind"):   doctor_knows_ids & ai_blind_ids & shared,
        ("Doctor blind", "AI knows"):   doctor_blind_ids & ai_knows_ids & shared,
        ("Doctor blind", "AI blind"):   doctor_blind_ids & ai_blind_ids & shared,
    }

    print("Quadrant sizes:")
    for (dr, ai), ids in quadrants.items():
        print(f"  {dr} × {ai}: n={len(ids)}")
    print(f"  (mixed-doctor excluded: {len(shared) - len(doctor_knows_ids) - len(doctor_blind_ids)} cases)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    positions = [
        (0, 0, ("Doctor knows", "AI knows")),
        (0, 1, ("Doctor knows", "AI blind")),
        (1, 0, ("Doctor blind", "AI knows")),
        (1, 1, ("Doctor blind", "AI blind")),
    ]

    for row, col, key in positions:
        ax = fig.add_subplot(gs[row, col])
        dr_label, ai_label = key
        ids = quadrants[key]
        panel_title = f"{dr_label}  ×  {ai_label}  (n={len(ids)})"
        draw_panel(ax, panel_title, data, ids)

    fig.suptitle(
        f"2×2 knowledge quadrants  (shared n={len(shared)}, "
        f"doctor_knows={len(doctor_knows_ids)}, doctor_blind={len(doctor_blind_ids)}, "
        f"ai_knows={len(ai_knows_ids & shared)}, ai_blind={len(ai_blind_ids & shared)})",
        fontsize=10,
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")
    plt.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/2x2_knowledge.png")
    main(p.parse_args().out)
