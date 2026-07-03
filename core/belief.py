"""Biased-Bayesian belief core for the v4 user simulator — pure math, zero LLM/IO deps.

Model (plan wise-greeting-lampson):
  b₀(k)      = c₀ if k == k₀ else (1−c₀)/(K−1)
  ℓ_{t+1}(k) = (1−λ)·log b_t(k) + λ·log b₀(k) + w_eff·e_t(k)
  b_{t+1}    = softmax(ℓ_{t+1})
  w_eff      = w · (1 + ρ · B_t / B*)          (burden-capitulation coupling; computed by caller)

Interpretation of the parameters (persona axes as explicit dials):
  c₀ ∈ (1/K, 1)  initial confidence in the persona's own opening pick
  λ  ∈ [0, 1)    anchoring to the initial belief (stubbornness). With zero evidence the update
                 contracts toward b₀ at rate (1−λ) per turn — the fixed point IS b₀, so a
                 stubborn doctor mathematically "drifts back to their first impression".
  w  ≥ 0         persuadability: how strongly one turn of argued evidence moves the belief.
  e_t(k) ∈ [−1,1] per-option evidence direction×strength tagged from the AI utterance TEXT
                 (see prompts/user_simulator_v4.yaml evidence_tagger_*) — pure assertions tag ≈0,
                 which is what makes the persona assertion-resistant but evidence-responsive.

Φ for reward shaping: phi(b, gold) = b[gold]. Per-turn deltas telescope to Φ_end − Φ₀.
"""
from __future__ import annotations

import math

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

_EPS = 1e-9  # probability floor before log (softmax output never hits 0, but guard anyway)


def init_belief(k0: str, c0: float, letters: tuple[str, ...] = LETTERS) -> dict[str, float]:
    """b₀: probability c₀ on the persona's own opening pick k0, uniform remainder elsewhere."""
    k0 = k0.strip().upper()
    if k0 not in letters:
        raise ValueError(f"k0 must be one of {letters}, got {k0!r}")
    n = len(letters)
    if not (1.0 / n < c0 < 1.0):
        raise ValueError(f"c0 must be in (1/{n}, 1), got {c0}")
    rest = (1.0 - c0) / (n - 1)
    return {k: (c0 if k == k0 else rest) for k in letters}


def update_belief(
    b: dict[str, float],
    b0: dict[str, float],
    evidence: dict[str, float],
    lam: float,
    w_eff: float,
) -> dict[str, float]:
    """One anchored/leaky-Bayes step in log space (see module docstring for the equation).

    evidence values are clamped to [−1, 1]; missing options count as 0 (no evidence).
    """
    if not (0.0 <= lam < 1.0):
        raise ValueError(f"lambda must be in [0,1), got {lam}")
    if w_eff < 0.0:
        raise ValueError(f"w_eff must be >= 0, got {w_eff}")
    scores = {}
    for k in b:
        e_k = max(-1.0, min(1.0, float(evidence.get(k, 0.0))))
        scores[k] = (
            (1.0 - lam) * math.log(max(b[k], _EPS))
            + lam * math.log(max(b0[k], _EPS))
            + w_eff * e_k
        )
    m = max(scores.values())  # softmax with max-shift for numerical stability
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def argmax_letter(b: dict[str, float]) -> str:
    """The persona's self-reported decision letter (deterministic tie-break by option order)."""
    return max(b, key=lambda k: (b[k], -LETTERS.index(k) if k in LETTERS else 0))


def confidence(b: dict[str, float]) -> float:
    """q_t = max_k b_t(k) — drives the agreement (END) termination check."""
    return max(b.values())


def phi(b: dict[str, float], gold: str) -> float:
    """Belief-potential Φ = b(gold). Reward-side only — the simulator itself never sees gold."""
    return b.get(str(gold).strip().upper(), 0.0)
