#!/usr/bin/env python3
"""
Plot scaling-POC / termination-POC results.

  --mode compare  — 서로 다른 baseline 간 비교 (실험당 pooled 커브 1개씩)
  --mode persona  — 단일 baseline 내 persona별 비교 (condition별 커브 각각)

scaling_poc / termination_poc 모두 동일한 포맷 (all_checkpoints, curve, trajectory_counts 등).
Labels are always the folder name — no override option.

Examples:

    python plot/code/plot_scaling_poc.py --mode persona outputs/scaling_poc_persona_naive
    python plot/code/plot_scaling_poc.py --mode persona outputs/termination_poc

    python plot/code/plot_scaling_poc.py --mode compare \\
        outputs/termination_poc \\
        outputs/termination_poc_medcobe_feedback

--output-dir overrides where PNGs land (default: plot/result/plot_scaling_poc/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_BUCKETS = ("true", "false")
_BUCKET_LABELS = {"true": "alone_correct [solid]", "false": "alone_incorrect [dashed]"}
_BUCKET_LINESTYLE = {"true": "-", "false": "--"}
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

_RESULT_DIR = Path(__file__).parent.parent / "result" / Path(__file__).stem


# ── Format check ─────────────────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"all_checkpoints", "curve", "trajectory_counts", "doctor_alone_accuracy"}

def _check_format(scores: dict, source) -> None:
    missing = _REQUIRED_KEYS - scores.keys()
    if missing:
        raise ValueError(
            f"{source}: scores 포맷 불일치 (없는 키: {sorted(missing)}).\n"
            f"  있는 키: {sorted(scores.keys())}\n"
            f"  run_poc.py를 재실행해서 결과를 새로 뽑아야 합니다 (구버전 results.json)."
        )


# ── Loading ──────────────────────────────────────────────────────────────────────────────

def _resolve_results_file(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        for candidate in ("summary.json", "results.json"):
            if (p / candidate).exists():
                return p / candidate
        raise FileNotFoundError(f"{p} is a directory but contains neither summary.json nor results.json")
    return p


def _inject_terminal_stats(scores: dict, data: dict) -> dict:
    """records 필드에서 terminal burden/accuracy를 계산해 scores에 주입."""
    records = data.get("records", [])
    if not records:
        return scores
    burdens = [r["final_cumulative_burden"] for r in records if "final_cumulative_burden" in r]
    accs = [1.0 if r.get("is_correct") else 0.0 for r in records]
    if not burdens:
        return scores
    scores = dict(scores)
    scores["avg_terminal_burden"] = round(sum(burdens) / len(burdens), 4)
    scores["terminal_accuracy"] = round(sum(accs) / len(accs), 4)
    scores["_terminal_burden_per_ep"] = burdens
    scores["_terminal_acc_per_ep"] = accs
    return scores


def load_conditions(path: str) -> list[tuple[str, dict]]:
    """summary.json → condition 키를 라벨로 그대로 써서 (label, scores) 리스트 반환."""
    resolved = _resolve_results_file(path)
    data = json.loads(resolved.read_text())
    if "scores" in data and "run_meta" in data:
        _check_format(data["scores"], resolved)
        return [(Path(path).name, _inject_terminal_stats(data["scores"], data))]
    pairs = list(data.items())
    for cond, scores in pairs:
        _check_format(scores, f"{resolved}[{cond}]")
    return pairs


def load_pooled(path: str, label: str) -> tuple[str, dict]:
    """summary.json → 전체 conditions pool해서 (label, pooled_scores) 반환."""
    resolved = _resolve_results_file(path)
    data = json.loads(resolved.read_text())
    if "scores" in data and "run_meta" in data:
        _check_format(data["scores"], resolved)
        return (label, _inject_terminal_stats(data["scores"], data))
    scores_list = list(data.values())
    for i, s in enumerate(scores_list):
        _check_format(s, f"{resolved}[{i}]")
    # summary.json 경우: 조건(persona/info) 이름으로 results.json 매칭해서 terminal stats 주입
    _terminal_by_condition: dict[str, dict] = {}
    for cond_name, scores in zip(data.keys(), scores_list):
        cond_dir = resolved.parent / cond_name.replace("/", "_")
        cond_results = cond_dir / "results.json"
        if cond_results.exists():
            cond_data = json.loads(cond_results.read_text())
            injected = _inject_terminal_stats({}, cond_data)
            scores.update({k: v for k, v in injected.items()
                           if k.startswith("_terminal") or k in ("avg_terminal_burden", "terminal_accuracy")})
            if "avg_terminal_burden" in injected:
                _terminal_by_condition[cond_name] = {
                    "avg_terminal_burden": injected["avg_terminal_burden"],
                    "terminal_accuracy": injected["terminal_accuracy"],
                    "n": len(cond_data.get("records", [])),
                }
    pooled = _pool_scaling_poc(scores_list) if len(scores_list) > 1 else scores_list[0]
    if _terminal_by_condition:
        pooled["_terminal_by_condition"] = _terminal_by_condition
    return (label, pooled)


# ── Pooling: scaling_poc ─────────────────────────────────────────────────────────────────

def _pool_scaling_poc(scores_list: list[dict]) -> dict:
    all_cps = scores_list[0]["all_checkpoints"]
    total_n_overall = sum(
        s["curve"][b][str(all_cps[0])]["n"] for s in scores_list for b in _BUCKETS
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

    # pool by_closed_by (optional field — skip if absent in all sources)
    _reasons = ["agreement", "burden_dropout", "max_turns"]
    sources_with_closed = [s for s in scores_list if "by_closed_by" in s]
    if sources_with_closed:
        by_closed_by: dict[str, dict] = {}
        for reason in _reasons:
            entries = [s["by_closed_by"][reason] for s in sources_with_closed]
            total_n = sum(e["n"] for e in entries)
            weighted = [(e["avg_n_turns"], e["n"]) for e in entries if e.get("avg_n_turns") is not None and e["n"] > 0]
            avg_n_turns = round(sum(v * n for v, n in weighted) / sum(n for _, n in weighted), 2) if weighted else None
            by_closed_by[reason] = {
                "n": total_n,
                "rate": round(total_n / total_n_overall, 4) if total_n_overall else 0.0,
                "avg_n_turns": avg_n_turns,
            }
        sources_with_avg = [(s["avg_n_turns"], sum(s["curve"][b][str(all_cps[0])]["n"] for b in _BUCKETS))
                            for s in scores_list if "avg_n_turns" in s]
        avg_n_turns_overall_val: float | None = (
            round(sum(v * n for v, n in sources_with_avg) / sum(n for _, n in sources_with_avg), 2)
            if sources_with_avg else None
        )
    else:
        by_closed_by = {}
        avg_n_turns_overall_val = None

    pooled: dict = {
        "all_checkpoints": all_cps,
        "doctor_alone_accuracy": round(doctor_alone_accuracy, 4),
        "curve": curve,
        "trajectory_counts": trajectory_counts,
        "burden_by_trajectory": burden_by_trajectory,
        "burden_judge_calls_ok": sum(s.get("burden_judge_calls_ok", 0) for s in scores_list),
        "burden_judge_calls_attempted": sum(s.get("burden_judge_calls_attempted", 0) for s in scores_list),
    }
    if by_closed_by:
        pooled["by_closed_by"] = by_closed_by
    if avg_n_turns_overall_val is not None:
        pooled["avg_n_turns"] = avg_n_turns_overall_val

    # pool terminal stats (injected by _inject_terminal_stats; absent in summary.json sources)
    all_ep_burden = [v for s in scores_list for v in s.get("_terminal_burden_per_ep", [])]
    all_ep_acc = [v for s in scores_list for v in s.get("_terminal_acc_per_ep", [])]
    if all_ep_burden:
        pooled["avg_terminal_burden"] = round(sum(all_ep_burden) / len(all_ep_burden), 4)
        pooled["terminal_accuracy"] = round(sum(all_ep_acc) / len(all_ep_acc), 4)
        pooled["_terminal_burden_per_ep"] = all_ep_burden
        pooled["_terminal_acc_per_ep"] = all_ep_acc

    return pooled



# ── Panels: scaling_poc ──────────────────────────────────────────────────────────────────

def _merged_accuracy(scores: dict, checkpoint: int) -> float:
    total_n, total_correct = 0, 0.0
    for bucket in _BUCKETS:
        cp = scores["curve"][bucket][str(checkpoint)]
        total_n += cp["n"]
        total_correct += cp["accuracy"] * cp["n"]
    return total_correct / total_n if total_n else 0.0


def _merged_burden(scores: dict, checkpoint: int) -> float:
    total_n, total_burden = 0, 0.0
    for bucket in _BUCKETS:
        cp = scores["curve"][bucket][str(checkpoint)]
        total_n += cp["n"]
        total_burden += cp["avg_cumulative_burden"] * cp["n"]
    return total_burden / total_n if total_n else 0.0


def plot_accuracy(ax, runs: list[tuple[str, dict]]) -> None:
    """Overall accuracy — one pooled curve per run (compare mode) or pooled across all personas (persona mode)."""
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        ys = [_merged_accuracy(scores, c) for c in cps]
        ax.plot(cps, ys, color=color, linestyle="-", marker="o", label=f"{label}")
        ax.axhline(scores["doctor_alone_accuracy"], color=color, linestyle=":", alpha=0.6,
                    label=f"{label} (doctor alone)")
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Overall accuracy vs. turn checkpoint")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_accuracy_per_persona(ax, runs: list[tuple[str, dict]]) -> None:
    """Per-persona accuracy curves (persona mode only)."""
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        ys = [_merged_accuracy(scores, c) for c in cps]
        ax.plot(cps, ys, color=color, linestyle="-", marker="o", label=label)
        ax.axhline(scores["doctor_alone_accuracy"], color=color, linestyle=":", alpha=0.6)
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accuracy by persona vs. turn checkpoint")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_accuracy_delta(ax, runs: list[tuple[str, dict]]) -> None:
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


def plot_burden_accuracy_pareto(ax, runs: list[tuple[str, dict]]) -> None:
    all_points: list[tuple[float, float]] = []
    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        xs = [_merged_burden(scores, c) for c in cps]
        ys = [_merged_accuracy(scores, c) for c in cps]
        ax.plot(xs, ys, color=color, linestyle="-", marker="o", alpha=0.85, label=label)
        for x, y, c in zip(xs, ys, cps):
            ax.annotate(str(c), (x, y), fontsize=6, color=color,
                        textcoords="offset points", xytext=(3, 3))
        all_points.extend(zip(xs, ys))
    best_per_x: dict[float, float] = {}
    for x, y in all_points:
        best_per_x[x] = max(best_per_x.get(x, -1.0), y)
    pareto: list[tuple[float, float]] = []
    best_acc = -1.0
    for x, y in sorted(best_per_x.items()):
        if y > best_acc:
            pareto.append((x, y))
            best_acc = y
    if pareto:
        fx, fy = zip(*pareto)
        ax.plot(fx, fy, color="black", linestyle="--", marker="x", linewidth=1.5,
                 label="Pareto frontier (all runs/checkpoints)")
    ax.set_xlabel("avg cumulative burden")
    ax.set_ylabel("overall accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accuracy vs. burden Pareto frontier (point labels = checkpoint)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


_CONDITION_MARKERS = {
    "veteran_attending":   "o",
    "exhausted_attending": "s",
    "eager_resident":      "^",
    "burned_out_resident": "D",
}
_DEFAULT_CONDITION_MARKER = "P"


def _condition_marker(cond_name: str) -> str:
    persona = cond_name.split("/")[0]
    return _CONDITION_MARKERS.get(persona, _DEFAULT_CONDITION_MARKER)


def plot_terminal_pareto(ax, runs: list[tuple[str, dict]]) -> None:
    """Terminal outcome Pareto: 각 policy의 실제 종료 시점 (avg terminal burden, terminal accuracy).

    _terminal_by_condition이 있으면 persona/condition별 개별 포인트를 표시하고 같은 policy끼리
    선으로 연결함. 없으면 policy 평균 1점으로 fallback. Pareto frontier는 condition-level 포인트
    (policy×persona) 기준으로 계산함.
    """
    # (x, y) pairs for Pareto computation — condition-level means
    frontier_points: list[tuple[float, float]] = []
    condition_legend_done: set[str] = set()

    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        by_cond: dict = scores.get("_terminal_by_condition", {})

        # per-episode scatter (low alpha background)
        ep_burden = scores.get("_terminal_burden_per_ep", [])
        ep_acc = scores.get("_terminal_acc_per_ep", [])
        if ep_burden:
            ax.scatter(ep_burden, ep_acc, color=color, alpha=0.10, s=10, zorder=2)

        if by_cond:
            # condition-level points (per persona)
            cxs, cys = [], []
            for cond_name, cstats in by_cond.items():
                cx, cy = cstats["avg_terminal_burden"], cstats["terminal_accuracy"]
                marker = _condition_marker(cond_name)
                persona = cond_name.split("/")[0]
                # legend entry for marker shape — shown once globally
                mkr_label = persona if persona not in condition_legend_done else "_nolegend_"
                condition_legend_done.add(persona)
                ax.scatter([cx], [cy], color=color, marker=marker, s=70, zorder=5,
                           edgecolors="white", linewidths=0.5, label=mkr_label)
                ax.annotate(persona.replace("_", " "), (cx, cy), fontsize=6, color=color,
                            textcoords="offset points", xytext=(4, 3))
                cxs.append(cx); cys.append(cy)
                frontier_points.append((cx, cy))
            # connect same-policy conditions with thin line
            if len(cxs) > 1:
                order = sorted(range(len(cxs)), key=lambda k: cxs[k])
                ax.plot([cxs[k] for k in order], [cys[k] for k in order],
                        color=color, linestyle="-", linewidth=0.8, alpha=0.5)
            # policy mean (larger marker)
            px = scores.get("avg_terminal_burden")
            py = scores.get("terminal_accuracy")
            if px is not None:
                ax.scatter([px], [py], color=color, marker="*", s=180, zorder=6, label=label)
                ax.annotate(label, (px, py), fontsize=7, color=color,
                            textcoords="offset points", xytext=(5, 4))
        else:
            # fallback: policy mean only
            px = scores.get("avg_terminal_burden")
            py = scores.get("terminal_accuracy")
            if px is None or py is None:
                continue
            ax.scatter([px], [py], color=color, s=80, zorder=5, label=label)
            ax.annotate(label, (px, py), fontsize=7, color=color,
                        textcoords="offset points", xytext=(5, 4))
            frontier_points.append((px, py))

    # Pareto frontier over condition-level (or policy-mean) points
    if len(frontier_points) >= 2:
        best_per_x: dict[float, float] = {}
        for x, y in frontier_points:
            best_per_x[x] = max(best_per_x.get(x, -1.0), y)
        pareto: list[tuple[float, float]] = []
        best_acc = -1.0
        for x, y in sorted(best_per_x.items()):
            if y > best_acc:
                pareto.append((x, y))
                best_acc = y
        if len(pareto) >= 2:
            fx, fy = zip(*pareto)
            ax.plot(fx, fy, color="black", linestyle="--", linewidth=1.5,
                    label="Pareto frontier (terminal)")

    ax.set_xlabel("avg terminal cumulative burden")
    ax.set_ylabel("terminal accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Terminal outcome Pareto (policy × persona operating points)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="both", linestyle="--", alpha=0.3)


_TRAJECTORY_DISPLAY = {
    "self_corrected": "recovered",
    "locked_wrong":   "error remains",
    "regressed":      "regressed",
    "always_correct": "preserved",
}


def plot_trajectories(ax, runs: list[tuple[str, dict]]) -> None:
    metrics = [
        ("self_corrected", "false"),
        ("locked_wrong", "false"),
        ("regressed", "true"),
        ("always_correct", "true"),
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
    ax.set_xticklabels([f"{_TRAJECTORY_DISPLAY[cls]}\n({_BUCKET_LABELS[b]})" for cls, b in metrics], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate within bucket")
    ax.set_title("Trajectory classification rates (first checkpoint -> last)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_end_turn(ax, runs: list[tuple[str, dict]]) -> None:
    """avg n_turns at episode end, closed_by reason별 breakdown."""
    _reasons = ["agreement", "burden_dropout", "max_turns"]
    has_data = any("by_closed_by" in s and any(s["by_closed_by"][r]["avg_n_turns"] is not None for r in _reasons) for _, s in runs)
    if not has_data:
        ax.set_visible(False)
        return
    x = np.arange(len(_reasons))
    width = 0.8 / max(len(runs), 1)
    for i, (label, scores) in enumerate(runs):
        if "by_closed_by" not in scores:
            continue
        color = _COLORS[i % len(_COLORS)]
        values = [scores["by_closed_by"][r].get("avg_n_turns") or 0.0 for r in _reasons]
        ns = [scores["by_closed_by"][r]["n"] for r in _reasons]
        offset = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"n={n}", ha="center", va="bottom", fontsize=6, color=color)
    # overall avg_n_turns as horizontal lines
    for i, (label, scores) in enumerate(runs):
        if "avg_n_turns" not in scores:
            continue
        color = _COLORS[i % len(_COLORS)]
        ax.axhline(scores["avg_n_turns"], color=color, linestyle=":", alpha=0.7,
                   label=f"{label} (overall avg={scores['avg_n_turns']})")
    ax.set_xticks(x)
    ax.set_xticklabels(_reasons, fontsize=9)
    ax.set_ylabel("avg n_turns at end")
    ax.set_title("Average episode length by termination reason")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_by_closed_by(ax, runs: list[tuple[str, dict]]) -> None:
    """termination_poc 전용: by_closed_by breakdown. 해당 키 없으면 패널 비움."""
    has_data = any("by_closed_by" in scores for _, scores in runs)
    if not has_data:
        ax.set_visible(False)
        return
    reasons = list(next(s for _, s in runs if "by_closed_by" in s)["by_closed_by"].keys())
    x = np.arange(len(reasons))
    width = 0.8 / max(len(runs), 1)
    for i, (label, scores) in enumerate(runs):
        if "by_closed_by" not in scores:
            continue
        color = _COLORS[i % len(_COLORS)]
        total = sum(scores["by_closed_by"][r]["n"] for r in reasons)
        values = [scores["by_closed_by"][r]["n"] / total if total else 0.0 for r in reasons]
        offset = (i - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate of cases")
    ax.set_title("Termination reason distribution (agreement / burden_dropout / max_turns)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def _save(fig, out: Path, n_runs: int) -> None:
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out} ({n_runs} curve(s))")


def _panel_list(runs: list[tuple[str, dict]], extra_first: list | None = None) -> list:
    """패널 순서 리스트 반환. has_termination/has_end_turn 여부에 따라 마지막에 추가."""
    panels = list(extra_first or [])
    panels += [
        (plot_accuracy_delta, runs),
        (plot_preserved_recovered, runs),
        (plot_burden, runs),
        (plot_burden_accuracy_pareto, runs),
        (plot_trajectories, runs),
    ]
    if any("avg_n_turns" in s or "by_closed_by" in s for _, s in runs):
        panels.append((plot_end_turn, runs))
    if any("by_closed_by" in s for _, s in runs):
        panels.append((plot_by_closed_by, runs))
    if any("avg_terminal_burden" in s for _, s in runs):
        panels.append((plot_terminal_pareto, runs))
    return panels


def render_compare(runs: list[tuple[str, dict]], out: Path) -> None:
    """compare 모드: 실험별로 pool된 커브 비교."""
    panels = _panel_list(runs, extra_first=[(plot_accuracy, runs)])
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 4 * len(panels) + 1))
    for ax, (fn, data) in zip(axes, panels):
        fn(ax, data)
    _save(fig, out, len(runs))


def render_persona(runs: list[tuple[str, dict]], out: Path) -> None:
    """persona 모드: overall(pool) 1커브 + 페르소나별 커브 각각."""
    scores_list = [s for _, s in runs]
    pooled = _pool_scaling_poc(scores_list) if len(scores_list) > 1 else scores_list[0]
    pooled_run = [("all personas (pooled)", pooled)]

    panels = _panel_list(runs, extra_first=[
        (plot_accuracy, pooled_run),
        (plot_accuracy_per_persona, runs),
    ])
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 4 * len(panels) + 1))
    for ax, (fn, data) in zip(axes, panels):
        fn(ax, data)
    _save(fig, out, len(runs))


# ── Subcommand handlers ───────────────────────────────────────────────────────────────────

def cmd_overall(args: argparse.Namespace) -> None:
    """모든 condition을 pool해서 패널 전체를 overall 단일 커브로 그린다."""
    out_dir = Path(args.output_dir) if args.output_dir else _RESULT_DIR
    path = args.results[0]
    label, scores = load_pooled(path, Path(path).name)
    render_compare([(label, scores)], out_dir / f"overall_{Path(path).name}.png")


def plot_overall(result_path: str | Path, output_dir: str | Path | None = None) -> None:
    """실험 스크립트에서 직접 호출할 수 있는 인터페이스."""
    out_dir = Path(output_dir) if output_dir else _RESULT_DIR
    path = Path(result_path)
    label, scores = load_pooled(str(path), path.name)
    render_compare([(label, scores)], out_dir / f"overall_{path.name}.png")


def cmd_compare(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else _RESULT_DIR
    pooled_runs = [load_pooled(p, Path(p).name) for p in args.results]
    out_name = "compare_" + "_vs_".join(lb for lb, _ in pooled_runs)
    render_compare(pooled_runs, out_dir / f"{out_name}.png")


def cmd_persona(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else _RESULT_DIR
    path = args.results[0]
    runs = load_conditions(path)
    render_persona(runs, out_dir / f"persona_{Path(path).name}.png")


# ── Main ─────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["overall", "compare", "persona"], default="overall",
                        help="overall(기본): 전체 pool → 단일 커브 | compare: baseline 간 비교 | persona: persona별 커브")
    parser.add_argument("results", nargs="+", help="실험 폴더 또는 summary.json/results.json 경로")
    parser.add_argument("--output-dir", default=None, help="출력 디렉터리 (기본: plot/result/plot_scaling_poc/)")
    args = parser.parse_args()

    if args.mode == "compare":
        cmd_compare(args)
    elif args.mode == "persona":
        if len(args.results) != 1:
            parser.error("--mode persona 는 경로를 정확히 1개만 받습니다")
        cmd_persona(args)
    else:  # overall (default)
        if len(args.results) != 1:
            parser.error("--mode overall 은 경로를 정확히 1개만 받습니다")
        cmd_overall(args)
