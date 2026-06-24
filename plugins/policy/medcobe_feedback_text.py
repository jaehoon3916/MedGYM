"""Behavior-guideline text assembly for the medcobe_feedback policy.

Ported from MedCOBE_EMNLP/src/medcobe_feedback/feedback_generator.py: picks one of
profile.{passive,aggressive,balanced} from prompts/medcobe_feedback_profiles.yaml based on a
model's measured sycophancy/obstruction rate, and fills in the {sycophancy*100:.0f}-style
placeholders.
"""
from __future__ import annotations

import re
from typing import Iterable

from core.prompt_builder import _load

DEFAULT_TAU = 0.20

# Matches placeholders like {sycophancy*100:.0f} and rewrites them to {sycophancy_pct:.0f} so
# str.format can handle them.
_MULT100_RE = re.compile(r"\{(\w+)\*100(:[^}]+)?\}")


def _normalize_template(template: str) -> str:
    return _MULT100_RE.sub(lambda m: "{%s_pct%s}" % (m.group(1), m.group(2) or ""), template)


def _build_context(metrics: dict[str, float]) -> dict[str, float]:
    ctx: dict[str, float] = dict(metrics)
    for name, value in metrics.items():
        ctx[f"{name}_pct"] = value * 100
    return ctx


def _format(template: str, metrics: dict[str, float]) -> str:
    if not template:
        return ""
    return _normalize_template(template).format(**_build_context(metrics)).strip()


def _select_profile_key(sycophancy: float, obstruction: float, tau: float) -> str:
    if sycophancy > tau:
        return "passive"
    if obstruction > tau:
        return "aggressive"
    return "balanced"


def _join_nonempty(parts: Iterable[str]) -> str:
    return "\n\n".join(p for p in parts if p)


def generate_feedback(sycophancy: float, obstruction: float, *, tau: float = DEFAULT_TAU) -> str:
    """Assemble the behavior guideline string for a model from prompts/medcobe_feedback_profiles.yaml."""
    profiles = _load("medcobe_feedback_profiles")
    metrics = {"sycophancy": sycophancy, "obstruction": obstruction}
    profile_section = profiles.get("profile") or {}
    profile_key = _select_profile_key(sycophancy, obstruction, tau)
    return _join_nonempty([_format(profile_section.get(profile_key, "") or "", metrics)])
