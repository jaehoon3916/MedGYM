from __future__ import annotations

from typing import Any

from core.json_utils import safe_json_load

# NASA-TLX-derived dimension order, fixed. Drives fallback construction, the "dominant
# driver" tie-break in _format_burden_breakdown, and JSON-shape validation. Two of the
# original six NASA-TLX dimensions are deliberately excluded from this tuple, not just
# null-filled -- the judge is never even asked about them:
#   physical_demand -- doesn't apply to this text-only clinical dialogue (no interface to
#     operate); permanently excluded.
#   temporal_demand -- excluded for now; the plan is to inject it later as a fixed constant
#     rather than an LLM self-report, so it's out of this judge's schema until that lands.
_TLX_DIMS = ("mental_demand", "performance", "effort", "frustration")

# Neutral mid-scale fallback value (1-5 scale), used only when every judge sample fails to
# parse. Mirrors xinobi/code/generate/conversation.py's _build_turn_judge_fallback's "fill
# mid-scale 3 for every criterion" idea -- structural pattern only, not its actual criteria.
_TLX_MID_FALLBACK = 3


def _coerce_tlx_score(value: Any) -> int | None:
    """int 1-5, clamped (not rejected) the same way v1._confidence clamps -- LLMs drift
    slightly outside requested ranges rather than refusing outright. Returns None only when
    the value can't be coerced to a number at all (missing field, non-numeric junk)."""
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, v))


def _parse_burden_judge(raw: str) -> dict[str, Any] | None:
    """Parse Stage 1's JSON. Returns None (triggering a retry) if ANY of the 4 required
    dimensions is missing/unparseable -- a partial result is treated as a full failure
    rather than silently zero-filling individual dimensions, so a parse failure is never
    indistinguishable from a genuine low score (same "all-or-nothing" philosophy as v1's
    _parse_followup retry-trigger condition).

    On success, recomputes overall_workload itself from the 4 parsed dims (mean, rounded to
    1 decimal) rather than trusting the model's own arithmetic -- the model is told the
    formula but LLMs are not reliable calculators; recomputing deterministically in code is
    strictly more trustworthy and costs nothing extra.
    """
    data = safe_json_load(raw)
    dims: dict[str, int] = {}
    for dim in _TLX_DIMS:
        v = _coerce_tlx_score(data.get(dim))
        if v is None:
            return None
        dims[dim] = v
    overall = round(sum(dims.values()) / len(dims), 1)
    rationale_raw = data.get("short_rationale")
    rationale_raw = rationale_raw if isinstance(rationale_raw, dict) else {}
    rationale = {}
    for dim in _TLX_DIMS:
        text = rationale_raw.get(dim)
        rationale[dim] = text.strip() if isinstance(text, str) and text.strip() else ""
    return {
        **dims,
        "overall_workload": overall,
        "short_rationale": rationale,
    }


def _burden_judge_fallback() -> dict[str, Any]:
    """All retries exhausted -> neutral mid-scale fill (3 for every 1-5 dimension)."""
    dims = {dim: _TLX_MID_FALLBACK for dim in _TLX_DIMS}
    return {
        **dims,
        "overall_workload": float(_TLX_MID_FALLBACK),
        "short_rationale": {dim: "fallback: judge parse failed" for dim in _TLX_DIMS},
    }


def _normalize_overall_workload(overall_workload: float) -> float:
    """[1,5] NASA-TLX overall_workload -> [0,1], mirroring v1's judge_mean_0_3/3.0 mapping
    of [0,3] -> [0,1]. 1 = very low workload (floor of the scale, not zero workload in any
    absolute sense) maps to 0.0; 5 = very high workload maps to 1.0.

    NOT used by v3.py's runtime cumulative-burden accumulation -- that sums raw 1-5 scores
    directly (see UserSimulatorV3._burden_cumulative) rather than normalizing, since mapping
    an ordinal 1-5 judge score to [0,1] bakes in an interval-scale + zero-point assumption
    that has no consumer yet (no reward/dropout uses this burden signal currently). Still used
    by scripts/verify_load_judge_v3.py's known-groups validation, which legitimately needs a
    [0,1] scale to compare against other normalized metrics."""
    return (overall_workload - 1.0) / 4.0


def _format_burden_breakdown(tlx: dict[str, Any]) -> str:
    """One compact line for Stage 2's {burden_breakdown_line}. Dominant driver = the
    highest-scoring dimension, ties broken by _TLX_DIMS order (earlier dimension wins)."""
    parts = ", ".join(f"{d}={tlx[d]}" for d in _TLX_DIMS)
    dominant = max(_TLX_DIMS, key=lambda d: (tlx[d], -_TLX_DIMS.index(d)))
    return f"{parts} (overall={tlx['overall_workload']}/5). Dominant driver: {dominant}."


def build_burden_judge_prompt_v3(
    tmpl: dict[str, str],
    scenario: str,
    persona_instruction: str,
    dialogue_text: str,
    ai_utterance: str,
    cumulative_burden: float = 0.0,
) -> list[dict[str, str]]:
    """Stage 1 message builder. Pure function (no `self`) so both
    UserSimulatorV3._build_burden_judge_prompt (a thin wrapper supplying self-derived args)
    and scripts/verify_load_judge_v3.py (supplying fixture args directly) can call it without
    the verify script reaching into v3.py's instance internals.

    `tmpl` is the loaded prompts/user_simulator_v3.yaml dict (caller-supplied so this module
    has no file-loading/caching responsibility of its own)."""
    ctx = dict(
        scenario=scenario,
        persona_instruction=persona_instruction,
        dialogue_text=dialogue_text,
        ai_utterance=ai_utterance,
        cumulative_burden=round(cumulative_burden, 2),
    )
    return [
        {"role": "system", "content": tmpl["burden_judge_system"].format(**ctx)},
        {"role": "user", "content": tmpl["burden_judge_user"].format(**ctx)},
    ]
