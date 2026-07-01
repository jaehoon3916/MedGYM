#!/usr/bin/env python3
"""
Plot scaling-POC / termination-POC results.

  --mode compare  — 서로 다른 baseline 간 비교 (실험당 pooled 커브 1개씩)
  --mode persona  — 단일 baseline 내 persona별 비교 (condition별 커브 각각)
  --mode quadrant — 2×2 knowledge quadrant (doctor knows/blind × AI knows/blind), run당 커브 1개씩

scaling_poc / termination_poc 모두 동일한 포맷 (all_checkpoints, curve, trajectory_counts 등).
Labels are always the folder name — no override option.

Examples:

    python plot/code/plot_scaling_poc.py --mode persona outputs/scaling_poc_persona_naive
    python plot/code/plot_scaling_poc.py --mode persona outputs/termination_poc

    python plot/code/plot_scaling_poc.py --mode compare \\
        outputs/termination_poc \\
        outputs/termination_poc_medcobe_feedback

    python plot/code/plot_scaling_poc.py --mode quadrant \\
        outputs/poc_0630_naive \\
        outputs/poc_0630_medcobe_feedback \\
        outputs/poc_0630_deliberation_llm

--output-dir overrides where PNGs land (default: plot/result/plot_scaling_poc/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
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
    valid = [(r["final_cumulative_burden"], 1.0 if r.get("is_correct") else 0.0)
             for r in records if r.get("final_cumulative_burden") is not None]
    accs = [1.0 if r.get("is_correct") else 0.0 for r in records]
    if not valid:
        return scores
    burdens, burden_accs = zip(*valid)
    burdens, burden_accs = list(burdens), list(burden_accs)
    scores = dict(scores)
    scores["avg_terminal_burden"] = round(sum(burdens) / len(burdens), 4)
    scores["terminal_accuracy"] = round(sum(accs) / len(accs), 4)
    scores["_terminal_burden_per_ep"] = burdens
    scores["_terminal_acc_per_ep"] = burden_accs
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

    # pool natural_end_agreement (first-END agreement view; n-weighted like by_closed_by)
    nea_sources = [s["natural_end_agreement"] for s in scores_list if s.get("natural_end_agreement")]
    if nea_sources:
        nea_total = sum(e["n"] for e in nea_sources)
        nea_turns = [(e["avg_n_turns"], e["n"]) for e in nea_sources if e.get("avg_n_turns") is not None and e["n"] > 0]
        nea_acc = [(e["accuracy"], e["n"]) for e in nea_sources if e.get("accuracy") is not None and e["n"] > 0]
        pooled["natural_end_agreement"] = {
            "n": nea_total,
            "rate": round(nea_total / total_n_overall, 4) if total_n_overall else 0.0,
            "avg_n_turns": round(sum(v * n for v, n in nea_turns) / sum(n for _, n in nea_turns), 2) if nea_turns else None,
            "accuracy": round(sum(v * n for v, n in nea_acc) / sum(n for _, n in nea_acc), 4) if nea_acc else None,
        }

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
    # Compute pooled baselines first so the 0th point on every curve can be unified.
    all_scores = [s for _, s in runs]
    total_n = sum(sum(s["curve"][b][str(s["all_checkpoints"][0])]["n"] for b in _BUCKETS) for s in all_scores)
    pooled_doc_alone: float | None = None
    pooled_ai_alone: float | None = None
    if total_n:
        pooled_doc_alone = sum(
            s["doctor_alone_accuracy"] * sum(s["curve"][b][str(s["all_checkpoints"][0])]["n"] for b in _BUCKETS)
            for s in all_scores
        ) / total_n
        ai_vals = [(s["ai_alone_accuracy"],
                    sum(s["curve"][b][str(s["all_checkpoints"][0])]["n"] for b in _BUCKETS))
                   for s in all_scores if s.get("ai_alone_accuracy") is not None]
        if ai_vals:
            pooled_ai_alone = sum(v * n for v, n in ai_vals) / sum(n for _, n in ai_vals)

    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        ys = [_merged_accuracy(scores, c) for c in cps]
        # Unify the 0th point so all curves share the same doctor-alone starting value.
        if pooled_doc_alone is not None and cps and cps[0] == 0:
            ys[0] = pooled_doc_alone
        ax.plot(cps, ys, color=color, linestyle="-", marker="o", label=f"{label}")

    # Draw unified solo baselines as single grey lines.
    if pooled_doc_alone is not None:
        ax.axhline(pooled_doc_alone, color="grey", linestyle=":", alpha=0.7,
                   label=f"doctor alone ({pooled_doc_alone:.3f})")
    if pooled_ai_alone is not None:
        ax.axhline(pooled_ai_alone, color="grey", linestyle="-.", alpha=0.7,
                   label=f"AI alone ({pooled_ai_alone:.3f})")

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
        if scores.get("ai_alone_accuracy") is not None:
            ax.axhline(scores["ai_alone_accuracy"], color=color, linestyle="-.", alpha=0.6)
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Accuracy by persona vs. turn checkpoint")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_accuracy_delta(ax, runs: list[tuple[str, dict]]) -> None:
    # Pooled doctor-alone baseline (same weight logic as plot_accuracy).
    all_scores = [s for _, s in runs]
    total_n = sum(sum(s["curve"][b][str(s["all_checkpoints"][0])]["n"] for b in _BUCKETS) for s in all_scores)
    if total_n:
        pooled_baseline = sum(
            s["doctor_alone_accuracy"] * sum(s["curve"][b][str(s["all_checkpoints"][0])]["n"] for b in _BUCKETS)
            for s in all_scores
        ) / total_n
    else:
        pooled_baseline = 0.0

    ai_vals = [(s["ai_alone_accuracy"],
                sum(s["curve"][b][str(s["all_checkpoints"][0])]["n"] for b in _BUCKETS))
               for s in all_scores if s.get("ai_alone_accuracy") is not None]
    pooled_ai_alone = (
        sum(v * n for v, n in ai_vals) / sum(n for _, n in ai_vals) if ai_vals else None
    )

    for i, (label, scores) in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        cps = scores["all_checkpoints"]
        ys = [_merged_accuracy(scores, c) - pooled_baseline for c in cps]
        # Unify the 0th point (doctor-alone starting point) to exactly 0.
        if cps and cps[0] == 0:
            ys[0] = 0.0
        ax.plot(cps, ys, color=color, linestyle="-", marker="o", label=label)

    ax.axhline(0.0, color="grey", linestyle=":", alpha=0.7,
               label=f"doctor alone (baseline = {pooled_baseline:.3f})")
    if pooled_ai_alone is not None:
        ai_delta = pooled_ai_alone - pooled_baseline
        ax.axhline(ai_delta, color="grey", linestyle="-.", alpha=0.7,
                   label=f"AI alone (Δ = {ai_delta:+.3f})")
    ax.set_xlabel("checkpoint (turns)")
    ax.set_ylabel("Δ accuracy vs. pooled doctor-alone")
    ax.set_title("Accuracy gain over doctor-alone baseline")
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


def _termination_bars(scores: dict) -> dict[str, tuple[float, int]]:
    """{reason: (avg_n_turns, n)} for the termination panel, as a TRUE partition.

    When force_full_turns suppressed real agreement (closed_by 'agreement' empty) but a first-END
    signal exists, reinterpret via natural_end: those cases ONLY landed in max_turns because the END
    was forced-ignored, so move them out of max_turns into agreement. Otherwise the n's double-count
    (e.g. agreement=211 AND max_turns=250). Non-forced runs are untouched (agreement bucket non-empty).
    """
    cb = scores.get("by_closed_by", {})
    reasons = ["agreement", "burden_dropout", "max_turns"]
    out = {r: ((cb.get(r, {}).get("avg_n_turns") or 0.0), cb.get(r, {}).get("n", 0)) for r in reasons}
    nat = scores.get("natural_end_agreement")
    if nat and not cb.get("agreement", {}).get("n"):
        out["agreement"] = (nat["avg_n_turns"] or 0.0, nat["n"])
        mt_v, mt_n = out["max_turns"]
        out["max_turns"] = (mt_v, max(0, mt_n - nat["n"]))  # remove the naturally-agreed cases
    return out


def plot_end_turn(ax, runs: list[tuple[str, dict]]) -> None:
    """avg n_turns at episode end, termination reason별 breakdown (true partition).
    'agreement' uses the first END token when force_full_turns suppressed real agreement,
    and those cases are removed from max_turns so the n's sum to the case count -- see
    _termination_bars."""
    _reasons = ["agreement", "burden_dropout", "max_turns"]
    has_data = any(
        ("by_closed_by" in s and any(s["by_closed_by"][r]["avg_n_turns"] is not None for r in _reasons))
        or s.get("natural_end_agreement")
        for _, s in runs
    )
    if not has_data:
        ax.set_visible(False)
        return
    x = np.arange(len(_reasons))
    width = 0.8 / max(len(runs), 1)
    for i, (label, scores) in enumerate(runs):
        if "by_closed_by" not in scores:
            continue
        color = _COLORS[i % len(_COLORS)]
        bars_data = _termination_bars(scores)
        values = [bars_data[r][0] for r in _reasons]
        ns = [bars_data[r][1] for r in _reasons]
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
    """termination_poc 전용: termination reason 분포. force_full_turns로 agreement가 억제된
    경우 _termination_bars의 natural_end 재해석(agreement<-first END, max_turns에서 제외)을
    그대로 써서 rate가 진짜 partition(합=1)이 되게 한다."""
    has_data = any("by_closed_by" in scores for _, scores in runs)
    if not has_data:
        ax.set_visible(False)
        return
    reasons = ["agreement", "burden_dropout", "max_turns"]
    x = np.arange(len(reasons))
    width = 0.8 / max(len(runs), 1)
    for i, (label, scores) in enumerate(runs):
        if "by_closed_by" not in scores:
            continue
        color = _COLORS[i % len(_COLORS)]
        ns = {r: n for r, (_, n) in _termination_bars(scores).items()}
        total = sum(ns[r] for r in reasons)
        values = [ns[r] / total if total else 0.0 for r in reasons]
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


# ── Panels: 2×2 knowledge quadrant (doctor knows/blind × AI knows/blind) ────────────────────
#
# Unlike every other mode above, this works directly off per-case `records` (each with its own
# `alone_correct` + per-checkpoint `is_correct`), not the pooled `scores["curve"]` aggregate --
# the quadrant partition is a per-case boolean AND across runs, which pooling would destroy.
# Ported from scripts/plot_2x2_knowledge.py (generalized from a hardcoded 3-run dict to N runs).

_DEFAULT_AI_BLIND_CACHE = "outputs/ai_blind_trials_cache.json"


def _resolve_case_records_file(path: str) -> Path:
    """Like _resolve_results_file, but quadrant mode needs a results.json with real per-case
    `records` -- summary.json (condition-pooled, no per-case records) is NOT usable here, unlike
    overall/compare/persona mode. poc_0630_* runs keep results.json one level down, under a
    persona-named subdir (e.g. veteran_attending_full/results.json) alongside a top-level
    summary.json -- passing the top-level dir used to silently resolve to summary.json and
    produce near-empty/garbage quadrants (case_ids became condition-name strings). If the given
    dir has no results.json but exactly one subdir does, auto-descend into it; if more than one
    subdir qualifies, fail loudly instead of guessing which persona/condition was meant."""
    p = Path(path)
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"{p} does not exist")
    direct = p / "results.json"
    if direct.exists():
        return direct
    candidates = sorted(
        d for d in p.iterdir() if d.is_dir() and (d / "results.json").exists()
    )
    if len(candidates) == 1:
        resolved = candidates[0] / "results.json"
        print(f"[quadrant] {p} has no results.json (only summary.json) -- "
              f"using {resolved} instead.")
        return resolved
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise FileNotFoundError(
            f"{p} has no results.json and multiple persona/condition subdirs with one "
            f"({names}) -- quadrant mode needs per-case records, not the pooled summary.json, "
            f"so pass one explicitly, e.g. {candidates[0]}"
        )
    raise FileNotFoundError(
        f"{p} has neither results.json nor a subdir containing one "
        f"(only summary.json, which lacks per-case records -- unusable for quadrant mode)."
    )


def _load_case_records(path: str) -> dict[str, dict]:
    """results.json (or a dir containing/leading to it) -> {case_id: record}."""
    resolved = _resolve_case_records_file(path)
    data = json.loads(resolved.read_text())
    records = data.get("records", data)  # tolerate a bare list-of-records file too
    if isinstance(records, dict):
        return records
    return {r["case_id"]: r for r in records}


def _quadrant_curve(records_by_id: dict[str, dict], case_ids: set[str]) -> tuple[list[int], list[float], int]:
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


def _draw_quadrant_panel(ax, title: str, data: dict[str, dict[str, dict]], case_ids: set[str]) -> None:
    has_data = False
    for i, (label, records_by_id) in enumerate(data.items()):
        color = _COLORS[i % len(_COLORS)]
        cps, accs, n = _quadrant_curve(records_by_id, case_ids)
        if not cps:
            continue
        has_data = True
        ax.plot(cps, accs, marker="o", color=color, label=f"{label} (n={n})", linewidth=1.5)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("checkpoint (turns)", fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=7)
    if not has_data:
        ax.text(0.5, 0.5, "no cases", ha="center", va="center", transform=ax.transAxes, color="grey")


def _dedupe_labels(paths: list[str]) -> list[str]:
    """Path(p).name, qualified with the parent dir when leaf names collide -- e.g. every
    poc_0630_* run's results.json lives under a persona-named leaf like
    'veteran_attending_full', so plain .name would silently collapse distinct runs into one
    dict key (mixed-doctor detection would then compare a run against itself)."""
    labels = [Path(p).name for p in paths]
    if len(set(labels)) == len(labels):
        return labels
    return [f"{Path(p).parent.name}/{Path(p).name}" for p in paths]


def render_quadrant(run_paths: list[str], ai_blind_cache: str, out: Path) -> None:
    """2×2 figure: rows = doctor knows/blind, cols = AI knows/blind. One curve per run per panel.

    doctor knows/blind = alone_correct True/False in ALL provided runs (mixed-agreement cases,
    i.e. runs disagree on whether the doctor started out correct, are excluded -- they aren't a
    property of the doctor alone). AI knows/blind = majority (>=2/3) over the ai_blind trials
    cache, keyed by case_id, independent of which runs are being compared.
    """
    cache = json.loads(Path(ai_blind_cache).read_text())
    ai_knows_ids = {cid for cid, t in cache.items() if len(t) >= 3 and any(t[:3])}
    ai_blind_ids = {cid for cid, t in cache.items() if len(t) >= 3 and not any(t[:3])}

    labels = _dedupe_labels(run_paths)
    data: dict[str, dict[str, dict]] = {
        label: _load_case_records(p) for label, p in zip(labels, run_paths)
    }
    shared = set.intersection(*(set(d.keys()) for d in data.values()))

    doctor_knows_ids: set[str] = set()
    doctor_blind_ids: set[str] = set()
    for cid in shared:
        trials = [data[label][cid].get("alone_correct", False) for label in data]
        if all(trials):
            doctor_knows_ids.add(cid)
        elif not any(trials):
            doctor_blind_ids.add(cid)
    # mixed-doctor cases (runs disagree) are excluded from the quadrant analysis

    quadrants = {
        ("Doctor knows", "AI knows"): doctor_knows_ids & ai_knows_ids & shared,
        ("Doctor knows", "AI blind"): doctor_knows_ids & ai_blind_ids & shared,
        ("Doctor blind", "AI knows"): doctor_blind_ids & ai_knows_ids & shared,
        ("Doctor blind", "AI blind"): doctor_blind_ids & ai_blind_ids & shared,
    }

    print("Quadrant sizes:")
    for (dr, ai), ids in quadrants.items():
        print(f"  {dr} × {ai}: n={len(ids)}")
    print(f"  (mixed-doctor excluded: {len(shared) - len(doctor_knows_ids) - len(doctor_blind_ids)} cases)")

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
        _draw_quadrant_panel(ax, f"{dr_label}  ×  {ai_label}  (n={len(ids)})", data, ids)

    fig.suptitle(
        f"2×2 knowledge quadrants  (shared n={len(shared)}, "
        f"doctor_knows={len(doctor_knows_ids)}, doctor_blind={len(doctor_blind_ids)}, "
        f"ai_knows={len(ai_knows_ids & shared)}, ai_blind={len(ai_blind_ids & shared)})",
        fontsize=10,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


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


def cmd_quadrant(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else _RESULT_DIR
    labels = [lbl.replace("/", "-") for lbl in _dedupe_labels(args.results)]
    out_name = "quadrant_" + "_vs_".join(labels)
    render_quadrant(args.results, args.ai_blind_cache, out_dir / f"{out_name}.png")


# ── Main ─────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["overall", "compare", "persona", "quadrant"], default="overall",
                        help="overall(기본): 전체 pool → 단일 커브 | compare: baseline 간 비교 | "
                             "persona: persona별 커브 | quadrant: 2×2 knowledge quadrant")
    parser.add_argument("results", nargs="+", help="실험 폴더 또는 summary.json/results.json 경로")
    parser.add_argument("--output-dir", default=None, help="출력 디렉터리 (기본: plot/result/plot_scaling_poc/)")
    parser.add_argument("--ai-blind-cache", default=_DEFAULT_AI_BLIND_CACHE,
                        help="--mode quadrant 전용: ai_blind trials cache 경로 "
                             f"(기본: {_DEFAULT_AI_BLIND_CACHE})")
    args = parser.parse_args()

    if args.mode == "compare":
        cmd_compare(args)
    elif args.mode == "persona":
        if len(args.results) != 1:
            parser.error("--mode persona 는 경로를 정확히 1개만 받습니다")
        cmd_persona(args)
    elif args.mode == "quadrant":
        if len(args.results) < 2:
            parser.error("--mode quadrant 는 경로를 2개 이상 받습니다")
        cmd_quadrant(args)
    else:  # overall (default)
        if len(args.results) != 1:
            parser.error("--mode overall 은 경로를 정확히 1개만 받습니다")
        cmd_overall(args)
