"""GRPO trajectory reward aggregation (reward.txt formula).

R(τ) = λ_align·Σ_t r_align + λ_final·r_final − λ_burden·Σ_t burden_t + λ_fmt·Σ_t r_fmt
A_i  = (R_i − μ_G) / (σ_G + ε)

The interaction cost is the clinician's cumulative cognitive_burden (per-turn NASA-TLX
overall_workload, 1-5), NOT raw turn count: turns are not equally expensive. Per-step burden
also gives turn-level credit assignment and subsumes the old length penalty (more turns =>
more accumulated burden), so λ_len / τ* are removed.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


DEFAULT_WEIGHTS = {
    # Lowered from 1.0: r_align is hand-crafted shaping, not validated ground truth.
    "lambda_align": 0.3,
    "lambda_final": 1.0,
    # Per-burden-unit cost. burden_t is 1-5/turn => Σ burden_t ~ 5-25 over an episode.
    # 0.02 keeps the penalty in the ~0.1-0.5 range (nudge vs r_final=1.0, not a rival).
    # CALIBRATION PLACEHOLDER — tune against observed cumulative-burden distributions.
    "lambda_burden": 0.02,
    "lambda_fmt": 0.1,
}


@dataclass
class Trajectory:
    """One sampled rollout: per-step align/format/burden signals + terminal outcome + action tokens."""
    step_align: list[float] = field(default_factory=list)
    step_fmt: list[float] = field(default_factory=list)
    # per-turn cognitive_burden (NASA-TLX overall_workload, 1-5) induced by each AI turn
    step_burden: list[float] = field(default_factory=list)
    # per step: (prompt_ids, action_ids) for logprob recompute in the update phase
    steps: list[tuple[Any, Any]] = field(default_factory=list)
    is_correct: bool = False
    num_turns: int = 0

    @property
    def R(self) -> float:  # noqa: N802 — keep paper notation
        raise NotImplementedError  # use trajectory_return(weights)


def trajectory_return(traj: Trajectory, weights: dict[str, Any]) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    r_align = sum(traj.step_align)
    r_fmt = sum(traj.step_fmt)
    r_final = 1.0 if traj.is_correct else 0.0
    burden_cost = sum(traj.step_burden)
    return (
        w["lambda_align"] * r_align
        + w["lambda_final"] * r_final
        + w["lambda_fmt"] * r_fmt
        - w["lambda_burden"] * burden_cost
    )


def group_advantages(returns: list[float], eps: float = 1e-8) -> list[float]:
    """Group-relative advantage A_i = (R_i − μ) / (σ + ε)."""
    if not returns:
        return []
    mu = sum(returns) / len(returns)
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return [(r - mu) / (sd + eps) for r in returns]
