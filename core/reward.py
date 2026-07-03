"""GRPO trajectory reward aggregation (docs/problem_formulation.md §5).

R(τ) = λ_final·1[correct]
     − λ_turn·num_turns          (Phase 1 cost: turns-to-close; 0 by default)
     − λ_burden·Σ_t burden_t     (Phase 2 cost: NASA-TLX cumulative burden; 0 in Phase 0/1)
     − λ_leak·Σ_t leak_t         (O2 reward-hacking guard for generative guidance; ON by default)
     + λ_align·Σ_t r_align + λ_fmt·Σ_t r_fmt      (shaping; 0 in the Phase-0 smoke)
A_i  = (R_i − μ_G) / (σ_G + ε)

Phased cost schedule (senior's turns-first plan, problem_formulation §5): Phase 0 accuracy-only
(pipeline check) → Phase 1 λ_turn (clean zero-noise scaling baseline) → Phase 2 λ_burden (ours;
turns are not equally expensive — a correcting CHALLENGE turn costs ~2.8 vs ~1.2 for CONVERGE).

Lessons wired in from the first smoke run (grpo_ours_v2_smoke, 2026-07-01):
- r_fmt is structurally 1.0 every turn for ours_v2 (control→McBurney mapping is always valid),
  so any λ_fmt>0 silently becomes a +λ_fmt·turns LENGTH reward (observed: meanTurns crept
  2.7→3.5 while all-correct groups had constant is_correct). Keep λ_fmt=0 for ours_v2.
- All-correct groups get advantage≈0 under accuracy-only R — correct GRPO behavior; training
  signal must come from mixed-outcome (wrong-anchored) cases, not from spurious tie-breakers.
- num_turns == turns-to-close only when user_llm.force_full_turns=false (agreement ends the
  episode). With force_full_turns=true every episode pads to max_turns and λ_turn is degenerate.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any


DEFAULT_WEIGHTS = {
    # Lowered from 1.0: r_align is hand-crafted shaping, not validated ground truth.
    "lambda_align": 0.3,
    "lambda_final": 1.0,
    # Phase-1 cost: per-turn penalty on turns-to-close. 0 by default; enable for the turn
    # frontier. Keep SMALL (≤0.05): early-CONVERGE collapse (recommend-first anchoring, Tier-1
    # B2) is the failure mode — watch `regressed` when raising it.
    "lambda_turn": 0.0,
    # Phase-2 cost: per-burden-unit. burden_t is 1-5/turn => Σ ~ 5-25 over an episode.
    # 0.02 keeps the penalty in the ~0.1-0.5 range (nudge vs r_final=1.0, not a rival).
    "lambda_burden": 0.02,
    "lambda_fmt": 0.1,
    # O2 guard: −1 per turn whose generated guidance names the correct option (letter or text).
    # ON by default the moment guidance is generative — without it, accuracy-only R is maximized
    # by whispering the answer (cost 0, acc 1): the policy would learn leakage, not deliberation.
    "lambda_leak": 1.0,
}


@dataclass
class Trajectory:
    """One sampled rollout: per-step align/format/burden/leak signals + terminal outcome + action tokens."""
    step_align: list[float] = field(default_factory=list)
    step_fmt: list[float] = field(default_factory=list)
    # per-turn cognitive_burden (NASA-TLX overall_workload, 1-5) induced by each AI turn
    step_burden: list[float] = field(default_factory=list)
    # per-turn leak flag (1.0 = the generated guidance named the correct option; see leak_score)
    step_leak: list[float] = field(default_factory=list)
    # per step: (prompt_ids, action_ids) for logprob recompute in the update phase
    steps: list[tuple[Any, Any]] = field(default_factory=list)
    is_correct: bool = False
    num_turns: int = 0

    @property
    def R(self) -> float:  # noqa: N802 — keep paper notation
        raise NotImplementedError  # use trajectory_return(weights)


def trajectory_return(traj: Trajectory, weights: dict[str, Any]) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return (
        w["lambda_final"] * (1.0 if traj.is_correct else 0.0)
        - w["lambda_turn"] * traj.num_turns
        - w["lambda_burden"] * sum(traj.step_burden)
        - w["lambda_leak"] * sum(traj.step_leak)
        + w["lambda_align"] * sum(traj.step_align)
        + w["lambda_fmt"] * sum(traj.step_fmt)
    )


def group_advantages(returns: list[float], eps: float = 1e-8) -> list[float]:
    """Group-relative advantage A_i = (R_i − μ) / (σ + ε)."""
    if not returns:
        return []
    mu = sum(returns) / len(returns)
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return [(r - mu) / (sd + eps) for r in returns]


# ── O2 leak guard ────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def leak_score(guidance: str, correct_option: str, answer_text: str) -> float:
    """1.0 if the policy's generated guidance names the correct option; else 0.0.

    Two channels (how-not-what invariant, problem_formulation §5 O2):
      letter — 'option C' / 'choice C' / 'answer is C' / '(C)' for the correct letter. Bare 'C'
               alone is NOT flagged (too many false positives: 'CT', vitamin C, ...).
      text   — the full normalized answer text appears verbatim in the guidance. Only checked
               when the normalized answer is ≥10 chars, so short generic answers can't
               false-positive on ordinary clinical vocabulary.
    Conservative by design: it catches copy-through leakage (e.g. the validator's reasoning
    naming the answer and the policy pasting it), not paraphrase. Tighten only with evidence.
    """
    if not guidance:
        return 0.0
    g = guidance.strip()
    letter = (correct_option or "").strip().upper()
    if letter:
        pat = rf"(?:\b(?:option|choice|answer)\s*(?:is\s*)?[(\[]?{re.escape(letter)}[)\]]?\b|\({re.escape(letter)}\))"
        if re.search(pat, g, flags=re.IGNORECASE):
            return 1.0
    ans = _norm(answer_text or "")
    if len(ans) >= 10 and ans in _norm(g):
        return 1.0
    return 0.0
