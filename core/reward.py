"""GRPO trajectory reward aggregation (reward.txt formula).

R(τ) = λ_align·Σ_t r_align + λ_final·r_final − λ_len·max(0, |τ|−τ*) + λ_fmt·Σ_t r_fmt
A_i  = (R_i − μ_G) / (σ_G + ε)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


DEFAULT_WEIGHTS = {
    # Lowered from 1.0 to match configs/config_grpo.yaml: r_align's BASE table is hand-crafted
    # shaping, not validated ground truth -- it should nudge r_final, not rival it.
    "lambda_align": 0.3,
    "lambda_final": 1.0,
    "lambda_len": 0.05,
    "lambda_fmt": 0.1,
    "tau_star": 6,
}


@dataclass
class Trajectory:
    """One sampled rollout: per-step align/format rewards + terminal outcome + recorded action tokens."""
    step_align: list[float] = field(default_factory=list)
    step_fmt: list[float] = field(default_factory=list)
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
    length_pen = max(0, traj.num_turns - int(w["tau_star"]))
    return (
        w["lambda_align"] * r_align
        + w["lambda_final"] * r_final
        + w["lambda_fmt"] * r_fmt
        - w["lambda_len"] * length_pen
    )


def group_advantages(returns: list[float], eps: float = 1e-8) -> list[float]:
    """Group-relative advantage A_i = (R_i − μ) / (σ + ε)."""
    if not returns:
        return []
    mu = sum(returns) / len(returns)
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return [(r - mu) / (sd + eps) for r in returns]
