#!/usr/bin/env python3
"""
Plot scaling-POC results. One positional arg per experiment to look at -- a folder
(outputs/<experiment>/, containing summary.json) or a single condition's results.json
(outputs/<experiment>/<persona>_<info_condition>/). No LLM calls, pure matplotlib.

    python plot/code/plot_scaling_poc.py outputs/scaling_poc_persona_naive

Always writes ONE breakdown PNG (every condition in every given experiment, overlaid, plus
each multi-condition experiment's own pooled "ALL combined" curve). Give MORE than one
experiment to compare them, and a SECOND PNG is written automatically -- one pooled curve
per experiment, i.e. the by-policy comparison, persona detail stripped out:

    python plot/code/plot_scaling_poc.py outputs/scaling_poc_persona_naive outputs/scaling_poc_persona_heuristic_delib --labels naive oracle
    # -> ..._breakdown.png (8 persona curves + 2 pooled curves)
    # -> ..._by_policy.png (just the 2 pooled curves)

--labels is optional (one per positional path, auto-named from the folder otherwise).
--output-dir overrides where PNGs land (default: plot/result/plot_scaling_poc/), filenames
are always derived from the input paths so different experiments never collide on one name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# json.dump stringifies dict keys: bool -> "true"/"false", int -> "0".."8".
_BUCKETS = ("true", "false")
_BUCKET_LABELS = {"true": "alone_correct [solid]", "false": "alone_incorrect [dashed]"}
_BUCKET_LINESTYLE = {"true": "-", "false": "--"}
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# This script lives at plot/code/plot_scaling_poc.py -- default output goes to the sibling
# plot/result/<script_stem>/ directory, not outputs/, per project convention.
_RESULT_DIR = Path(__file__).parent.parent / "result" / Path(__file__).stem


# ── Loading ──────────────────────────────────────────────────────────────────────────────

def _resolve_results_file(path: str) -> Path:
    """A --results path may be the file itself, OR the folder containing it -- the
    experiment root outputs/<experiment>/ (has summary.json) or a single condition folder
    outputs/<experiment>/<persona>_<info_condition>/ (has results.json)."""
    p = Path(path)
    if p.is_dir():
        for candidate in ("summary.json", "results.json"):
            if (p / candidate).exists():
                return p / candidate
        raise FileNotFoundError(f"{p} is a directory but contains neither summary.json nor results.json")
    return p


def _default_name(path: str) -> str:
    """A name for this path that's unique across DIFFERENT EXPERIMENTS, not just across
    conditions within one -- a bare results.json's parent folder is just the condition name
    (e.g. "veteran_attending_full"), the SAME under every experiment that swept that persona."""
    resolved = _resolve_results_file(path)
    if resolved.name == "summary.json":
        return resolved.parent.name
    return f"{resolved.parent.parent.name}_{resolved.parent.name}"


def load_experiment(path: str, label: str | None) -> tuple[str, list[tuple[str, dict]], dict | None]:
    """Loads one experiment path. Returns (experiment_label, per_condition_runs, pooled_scores).
    pooled_scores is None for a single results.json (nothing to pool -- it already IS the one
    curve); per_condition_runs has exactly one entry in that case too."""
    resolved = _resolve_results_file(path)
    data = json.loads(resolved.read_text())
    exp_label = label if label is not None else _default_name(path)

    if "scores" in data and "run_meta" in data:
        return exp_label, [(exp_label, data["scores"])], None

    # summary.json: flat {condition_key: scores_dict}, no wrapper.
    runs = [(f"{exp_label}: {cond}", scores) for cond, scores in data.items()]
    pooled = _pool_scores(list(data.values())) if len(data) > 1 else next(iter(data.values()))
    return exp_label, runs, pooled


def _pool_scores(scores_list: list[dict]) -> dict:
    """Combine multiple conditions' scores dicts (e.g. all 4 personas in one sweep) into one
    synthetic scores dict of the same shape, as if every condition's cases had been run as a
    single pooled sample -- weighted by each condition's n, not a plain average of rates
    (conditions can have different alone_correct/alone_incorrect split sizes)."""
    all_cps = scores_list[0]["all_checkpoints"]

    total_n_overall = sum(
        scores["curve"][b][str(all_cps[0])]["n"] for scores in scores_list for b in _BUCKETS
    )
    doctor_alone_accuracy = (
        sum(s["doctor_alone_accuracy"] * sum(s["curve"][b][str(all_cps[0])]["n"] for b in _BUCKETS)
            for s in scores_list) / total_n_overall
    ) if total_n_overall else 0.0

    curve: dict[str, dict] = {b: {} for b in _BUCKETS}
    for b in _BUCKETS:
        for c in all_cps:
            cps_here = [s["curve"][b][str(c)] for s in scores_list]
            n = sum(cp["n"] for cp in cps_here)
            acc = sum(cp["accuracy"] * cp["n"] for cp in cps_here) / n if n else 0.0
            burden = sum(cp["avg_cumulative_burden"] * cp["n"] for cp in cps_here) / n if n else 0.0
            curve[b][str(c)] = {"accuracy": round(acc, 4), "avg_cumulative_burden": round(burden, 4), "n": n}

    trajectory_counts: dict[str, dict] = {b: {} for b in _BUCKETS}
    for b in _BUCKETS:
        classes = scores_list[0]["trajectory_counts"][b].keys()
        trajectory_counts[b] = {
            cls: sum(s["trajectory_counts"][b][cls] for s in scores_list) for cls in classes
        }

    burden_by_trajectory: dict[str, float | None] = {}
    for cls in scores_list[0]["burden_by_trajectory"].keys():
        weighted, total_count = 0.0, 0
        for s, b_counts in zip(scores_list, (s["trajectory_counts"] for s in scores_list)):
            count = sum(b_counts[b].get(cls, 0) for b in _BUCKETS)
            v = s["burden_by_trajectory"].get(cls)
            if v is not None and count:
                weighted += v * count
                total_count += count
        burden_by_trajectory[cls] = round(weighted / total_count, 4) if total_count else None

    return {
        "all_checkpoints": all_cps,
        "doctor_alone_accuracy": round(doctor_alone_accuracy, 4),
        "curve": curve,
        "trajectory_counts": trajectory_counts,
        "burden_by_trajectory": burden_by_trajectory,
        "burden_judge_calls_ok": sum(s.get("burden_judge_calls_ok", 0) for s in scores_list),
        "burden_judge_calls_attempted": sum(s.get("burden_judge_calls_attempted", 0) for s in scores_list),
    }


# ── Panels ───────────────────────────────────────────────────────────────────────────────

def _merged_accuracy(scores: dict, checkpoint: int) -> float:
    """Overall accuracy at a checkpoint, pooling the alone_correct/alone_incorrect buckets
    (weighted by each bucket's n) instead of reporting them as separate curves -- the bucket
    split is definitional at checkpoint 0 (that's literally how the buckets are assigned) and
    not informative as an accuracy comparison."""
    total_n, total_correct = 0, 0.0
    for bucket in _BUCKETS:
        cp = scores["curve"][bucket][str(checkpoint)]
        total_n += cp["n"]
        total_correct += cp["accuracy"] * cp["n"]
    return total_correct / total_n if total_n else 0.0


def plot_accuracy(ax, runs: list[tuple[str, dict]]) -> None:
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        ys = [_merged_accuracy(scores, c) for c in cps]
        ax.plot(cps, ys, color=color, linestyle="-", marker="o", label=f"{label} (overall accuracy)")
        ax.axhline(scores["doctor_alone_accuracy"], color=color, linestyle=":", alpha=0.6,
                    label=f"{label} (doctor alone, baseline)")
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Overall accuracy vs. turn checkpoint")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_accuracy_delta(ax, runs: list[tuple[str, dict]]) -> None:
    """Change from the doctor-alone baseline at each checkpoint -- positive means the AI
    dialogue up to that turn count helped relative to the doctor's solo judgment, negative
    means it hurt."""
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        baseline = scores["doctor_alone_accuracy"]
        ys = [_merged_accuracy(scores, c) - baseline for c in cps]
        ax.plot(cps, ys, color=color, linestyle="-", marker="o", label=label)
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("accuracy - doctor_alone_accuracy")
    ax.set_title("Accuracy delta vs. doctor-alone baseline")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_preserved_recovered(ax, runs: list[tuple[str, dict]]) -> None:
    """The un-merged per-bucket accuracy curve plot_accuracy's pooling hides -- and the same
    numbers under their complementarity-relevant names: the alone_correct bucket's accuracy
    at checkpoint c is the rate knowledge already held at turn 0 is still correct (PRESERVED,
    solid); the alone_incorrect bucket's accuracy at c is the rate an initial error was fixed
    by turn c (RECOVERED, dashed)."""
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        preserved = [scores["curve"]["true"][str(c)]["accuracy"] for c in cps]
        recovered = [scores["curve"]["false"][str(c)]["accuracy"] for c in cps]
        ax.plot(cps, preserved, color=color, linestyle="-", marker="o",
                 label=f"{label} (preserved -- alone_correct)")
        ax.plot(cps, recovered, color=color, linestyle="--", marker="s",
                 label=f"{label} (recovered -- alone_incorrect)")
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("accuracy within bucket")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Knowledge preserved (solid) vs. recovered (dashed) over turns")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_burden(ax, runs: list[tuple[str, dict]]) -> None:
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        for bucket in _BUCKETS:
            ys = [scores["curve"][bucket][str(c)]["avg_cumulative_burden"] for c in cps]
            ax.plot(cps, ys, color=color, linestyle=_BUCKET_LINESTYLE[bucket], marker="o",
                     label=f"{label} ({_BUCKET_LABELS[bucket]})")
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("avg cumulative burden")
    ax.set_title("Cognitive burden vs. turn checkpoint")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_trajectories(ax, runs: list[tuple[str, dict]]) -> None:
    # Rate within the relevant bucket, not raw count: bucket sizes differ run-to-run (doctor
    # alone-accuracy is stochastic), so counts alone aren't comparable across runs.
    metrics = [
        ("self_corrected", "false"),   # rate within alone_incorrect bucket = recovered (first->last)
        ("locked_wrong", "false"),
        ("regressed", "true"),         # rate within alone_correct bucket
        ("always_correct", "true"),    # = preserved (first->last)
    ]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(runs), 1)
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        values = []
        for cls, bucket in metrics:
            counts = scores["trajectory_counts"][bucket]
            n = sum(counts.values())
            values.append(counts[cls] / n if n else 0.0)
        offset = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cls}\n({_BUCKET_LABELS[b]})" for cls, b in metrics], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate within bucket")
    ax.set_title("Trajectory classification rates (first checkpoint -> last)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def render(runs: list[tuple[str, dict]], out: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(11, 21))
    plot_accuracy(axes[0], runs)
    plot_accuracy_delta(axes[1], runs)
    plot_preserved_recovered(axes[2], runs)
    plot_burden(axes[3], runs)
    plot_trajectories(axes[4], runs)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} ({len(runs)} curve(s))")


# ── Main ─────────────────────────────────────────────────────────────────────────────────

def main(results_paths: list[str], labels: list[str | None] | None, output_dir: str | None) -> None:
    if labels is not None and len(results_paths) != len(labels):
        raise ValueError(f"paths ({len(results_paths)}) and --labels ({len(labels)}) must have the same length")
    resolved_labels: list[str | None] = labels if labels is not None else [None] * len(results_paths)
    out_dir = Path(output_dir) if output_dir else _RESULT_DIR
    base_name = "_vs_".join(_default_name(p) for p in results_paths)

    breakdown_runs: list[tuple[str, dict]] = []
    pooled_runs: list[tuple[str, dict]] = []
    for path, label in zip(results_paths, resolved_labels):
        exp_label, runs, pooled = load_experiment(path, label)
        breakdown_runs.extend(runs)
        if pooled is not None:
            breakdown_runs.append((f"{exp_label}: ALL combined", pooled))
        pooled_runs.append((exp_label, pooled if pooled is not None else runs[0][1]))

    render(breakdown_runs, out_dir / f"{base_name}_breakdown.png")
    if len(results_paths) > 1:
        render(pooled_runs, out_dir / f"{base_name}_by_policy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+",
                         help="One or more experiment folders or results.json/summary.json files")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Name per path, same order (default: derived from the folder name)")
    parser.add_argument("--output-dir", default=None,
                         help="Where to write PNGs (default: plot/result/plot_scaling_poc/)")
    args = parser.parse_args()
    main(args.results, args.labels, args.output_dir)
