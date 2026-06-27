from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.json_utils import safe_json_load
from core.schemas import CaseInfo, DialogueHistory, EpisodeConfig
from plugins.user_llm.base import UserLLMPlugin
from plugins.user_llm.user_simulator.v3_burden import (
    _TLX_DIMS,
    _burden_judge_fallback,
    _format_burden_breakdown,
    _parse_burden_judge,
    build_burden_judge_prompt_v3,
)
from plugins.vllm_base import VLLMBasePlugin

_ROOT = Path(__file__).parent.parent.parent.parent
_PROMPTS_DIR = _ROOT / "prompts"
_PERSONA_DIR = _ROOT / "persona"
_PERSONA_ROLE_DIR = _ROOT / "source" / "persona"

_MAX_FORMAT_RETRIES = 2
_VALID_LETTERS = ("A", "B", "C", "D")
_VALID_DECISIONS = ("CONTINUE", "END")
_VALID_PERSONAS = ("veteran_attending", "exhausted_attending", "eager_resident", "burned_out_resident")

# Injected into Stage 2's ctx as {forced_withdrawal_line} only when this turn's accumulated
# burden crosses self._burden_dropout_threshold -- empty string otherwise (see _followup_turn).
_FORCED_WITHDRAWAL_LINE = """
[Forced Withdrawal]
Your cognitive workload in this discussion has now exceeded what you can tolerate. Regardless
of where the discussion currently stands, you must end it now. Write a brief closing statement
that reflects you stepping back because you are overwhelmed (e.g. fatigue, frustration, feeling
unable to keep evaluating this) -- do not pretend the discussion reached a resolution. Set
"decision" to "END".
"""


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    """Merges user_simulator_v3.yaml (Stage 2: utterance_*) with burden_judge.yaml (Stage 1:
    burden_judge_*, split out for ease of editing) into one dict, so every existing
    tmpl["burden_judge_*"]/tmpl["utterance_*"] access site is unaffected by the split."""
    with open(_PROMPTS_DIR / "user_simulator_v3.yaml") as f:
        tmpl = yaml.safe_load(f)
    with open(_PROMPTS_DIR / "burden_judge.yaml") as f:
        tmpl.update(yaml.safe_load(f))
    return tmpl


@lru_cache(maxsize=1)
def _load_personas() -> dict:
    """Persona definitions are managed separately under source/persona/ -- one
    persona_{name}.yaml per named role, merged here into a single name -> text dict."""
    personas: dict = {}
    for path in sorted(_PERSONA_ROLE_DIR.glob("persona_*.yaml")):
        with open(path) as f:
            personas.update(yaml.safe_load(f))
    return personas


@lru_cache(maxsize=1)
def _load_burden_dropout_thresholds() -> dict:
    """Per-persona default burden_dropout_threshold -- see
    source/persona/burden_dropout_thresholds.yaml. A run config's explicit
    burden_dropout_threshold always overrides this."""
    with open(_PERSONA_ROLE_DIR / "burden_dropout_thresholds.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_sparsity_persona() -> dict:
    with open(_PERSONA_DIR / "information_sparsity.yaml") as f:
        return yaml.safe_load(f)


# ── generic helpers (intentionally duplicated from v1.py, not imported -- keeps v1 and v3
# fully independent modules; neither may depend on the other's internals) ──────────────────

def _fmt_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


def _fmt_history(history: DialogueHistory) -> str:
    if not history.turns:
        return "(No conversation yet)"
    lines = []
    for turn in history.turns:
        role = "AI Assistant" if turn.speaker == "medical" else "Doctor"
        lines.append(f"[{role}]: {turn.text}")
    return "\n".join(lines)


def _clean_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _letter(value: Any) -> str | None:
    v = str(value).strip().upper() if value is not None else ""
    return v if v in _VALID_LETTERS else None


def _confidence(value: Any) -> float | None:
    """q_t -- clamp to [0,1] rather than reject, since LLMs occasionally drift slightly
    outside the requested range instead of refusing outright."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, v))


# ── Stage 2 parsing ─────────────────────────────────────────────────────────────────────

def _parse_opening(raw: str, lettered: bool = True) -> tuple[str, str | None, str | None, float | None]:
    """turn0 -- (response, belief, reasoning, confidence). No AI utterance exists yet on turn
    0, so there is nothing for "argument_validity" to evaluate; turn0 keeps "reasoning" as its
    internal-justification field.

    `lettered` mirrors show_options: when True (options shown), "belief" is a single A-D
    letter, parsed via _letter(). When False (utterance_turn0_*_no_options' prompt asks for
    "the clinical decision you have independently made" in free text -- there is nothing
    lettered to point at), belief is kept as that free text directly -- NOT discarded by
    forcing it through _letter() (that was a real bug: the no-options prompt already asked
    for and received a free-text belief, but the old code threw it away as None because it
    never matched A/B/C/D)."""
    data = safe_json_load(raw)
    response = _clean_str(data.get("response")) or raw.strip()
    belief = _letter(data.get("belief")) if lettered else _clean_str(data.get("belief"))
    return (
        response, belief, _clean_str(data.get("reasoning")),
        _confidence(data.get("confidence")),
    )


def _parse_followup(raw: str) -> tuple[str, str | None, str | None, str | None, float | None]:
    """followup (options shown) -- (response, belief, decision, argument_validity,
    confidence). "argument_validity" REPLACES v1's "reasoning" field here (see
    prompts/user_simulator_v3.yaml's utterance_followup_system Output Format)."""
    data = safe_json_load(raw)
    response = _clean_str(data.get("response")) or raw.strip()
    decision_raw = str(data.get("decision") or "").strip().upper()
    decision = decision_raw if decision_raw in _VALID_DECISIONS else None
    return (
        response, _letter(data.get("belief")), decision,
        _clean_str(data.get("argument_validity")), _confidence(data.get("confidence")),
    )


def _parse_followup_no_options(raw: str) -> tuple[str, str | None, str | None, float | None, str | None]:
    """followup (no options) -- (response, belief, decision, confidence, argument_validity).
    "belief" here is free text (current diagnostic/management impression in the doctor's own
    words) -- there is no lettered option to point at, but the doctor's belief must still be
    captured every turn so it's never silently null in the saved transcript (see
    prompts/user_simulator_v3.yaml's utterance_followup_system_no_options Output Format)."""
    data = safe_json_load(raw)
    response = _clean_str(data.get("response")) or raw.strip()
    decision_raw = str(data.get("decision") or "").strip().upper()
    decision = decision_raw if decision_raw in _VALID_DECISIONS else None
    return (
        response, _clean_str(data.get("belief")), decision,
        _confidence(data.get("confidence")), _clean_str(data.get("argument_validity")),
    )


class UserSimulatorV3(VLLMBasePlugin, UserLLMPlugin):
    """Two-stage doctor simulator: (1) a first-person NASA-TLX self-report burden judge
    (persona+history+case+latest AI utterance aware, see prompts/user_simulator_v3.yaml's
    burden_judge_* templates, derived from /home/kjy/Jaehoon/nasa-tmx-scale.txt), feeding
    into (2) the same combined belief/confidence/decision/response doctor call v1 uses
    (utterance_* templates), now also receiving Stage 1's output as injected context and
    gaining an "argument_validity" field (forced to appear before "belief" in the JSON
    schema) that replaces v1's "reasoning" field in followup turns.

    v1's third-person 0-3 load_judge.yaml / _score_burden and its r*q_t_prior amplification
    (_effective_burden) are NOT reused -- v3's burden judge is persona- and history-aware by
    construction, so that amplification is dropped (see the plan's "open design question A"
    for the reasoning this was confirmed on, not silently assumed). v1.py is never imported
    from here and is never modified.
    """

    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        self._episode_cfg: EpisodeConfig = EpisodeConfig()
        self._force_close: bool = False
        self._last_belief: str | None = None
        self._burden_samples: int = int(config.get("burden_samples_per_item", 3))
        self._force_full_turns: bool = bool(config.get("force_full_turns", False))
        self._show_options: bool = bool(config.get("show_options", True))
        persona = str(config.get("persona", "burned_out_resident")).strip().lower()
        self._persona: str = persona if persona in _VALID_PERSONAS else "burned_out_resident"
        information_sparsity = str(config.get("information_sparsity", "dense")).strip().lower()
        self._information_sparsity: str = (
            information_sparsity if information_sparsity in ("dense", "sparse") else "dense"
        )
        # Running sum of raw 1-5 per-turn overall_workload (NOT normalized to [0,1] --
        # normalizing would impose an interval-scale + zero-point assumption on an ordinal
        # 1-5 judge score before any actual consumer needs that scale; deferred until a real
        # consumer (e.g. reward/dropout) requires it). Monotonically non-decreasing by
        # construction -- no decay/relief modeling for now (see plan baseline-snug-acorn.md).
        self._burden_cumulative: float = 0.0
        # Defaults to this persona's entry in source/persona/burden_dropout_thresholds.yaml
        # (currently the same uncalibrated placeholder for all 4 personas) unless a run config
        # sets plugins.user_llm.burden_dropout_threshold explicitly, which always wins. When
        # the running raw sum crosses this, the doctor is forced to withdraw this turn (see
        # _followup_turn).
        self._burden_dropout_threshold: float = float(
            config.get("burden_dropout_threshold", _load_burden_dropout_thresholds()[self._persona])
        )
        # Placeholder for a possible future "judge-only" vs "judge+formula" A/B comparison --
        # default False per the plan's recommendation to DROP authority amplification in v3
        # (Stage 1 is already persona-aware, so re-applying a hand-picked multiplier on top
        # would double-count the same effect). No multiplication logic is wired to this flag
        # yet; it exists purely so that comparison doesn't require re-deriving the formula.
        self._apply_authority_amplification: bool = bool(config.get("apply_authority_amplification", False))

    def reset_episode(self, episode_config: EpisodeConfig) -> None:
        self._episode_cfg = episode_config
        self._force_close = False
        self._last_belief = None
        self._burden_cumulative = 0.0

    def force_close(self) -> None:
        self._force_close = True

    def name(self) -> str:
        return f"user-simulator-v3-{self._model}"

    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> tuple[str, bool, dict[str, Any] | None]:
        if self._force_close:
            self._force_close = False
            return "I understand. Let's close the discussion here.", True, None

        if not dialogue_history.turns:
            return self._opening_turn(case_info)
        return self._followup_turn(case_info, dialogue_history)

    # ── Turn 0: Stage 2 only -- no AI utterance exists yet to judge ────────────────────

    def _opening_turn(self, case_info: CaseInfo) -> tuple[str, bool, dict[str, Any]]:
        tmpl = _load_prompts()
        ctx = dict(
            scenario=case_info.scenario,
            image_caption=case_info.caption or "N/A",
            question=case_info.question,
            persona_instruction=_load_personas()[self._persona],
            sparsity_instruction=_load_sparsity_persona()[self._information_sparsity],
        )
        if self._show_options:
            ctx["options_text"] = _fmt_options(case_info.options)
            sys_key, user_key = "utterance_turn0_system", "utterance_turn0_user"
        else:
            sys_key, user_key = "utterance_turn0_system_no_options", "utterance_turn0_user_no_options"
        messages = [
            {"role": "system", "content": tmpl[sys_key].format(**ctx)},
            {"role": "user", "content": tmpl[user_key].format(**ctx)},
        ]
        raw = self._chat(messages, response_format={"type": "json_object"})
        utterance, belief, reasoning, confidence = _parse_opening(raw, lettered=self._show_options)
        self._last_belief = belief or self._last_belief
        return utterance, False, {
            "belief": self._last_belief, "confidence": confidence,
            "decision": None, "reasoning": reasoning,
            **self._episode_cfg.model_dump(),
        }

    # ── Turn k>=1: Stage 1 (burden judge) then Stage 2 (utterance gen) ─────────────────

    def _followup_turn(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[str, bool, dict[str, Any]]:
        tlx, burden_n_ok, burden_n_attempted = self._score_burden_tlx(case_info, history)
        # Accumulate BEFORE building Stage 2's ctx, mirroring v1's ordering -- the doctor
        # reacts to the running total including the AI turn it's responding to right now.
        # Raw 1-5 sum, not normalized (see _burden_cumulative's docstring above) -- can exceed 5.
        self._burden_cumulative += tlx["overall_workload"]
        burden_exceeded = self._burden_cumulative >= self._burden_dropout_threshold

        tmpl = _load_prompts()
        ctx = dict(
            scenario=case_info.scenario,
            image_caption=case_info.caption or "N/A",
            question=case_info.question,
            dialogue_text=_fmt_history(history),
            persona_instruction=_load_personas()[self._persona],
            sparsity_instruction=_load_sparsity_persona()[self._information_sparsity],
            cumulative_burden=round(self._burden_cumulative, 2),
            burden_breakdown_line=_format_burden_breakdown(tlx),
            forced_withdrawal_line=_FORCED_WITHDRAWAL_LINE if burden_exceeded else "",
        )
        if self._show_options:
            ctx["options_text"] = _fmt_options(case_info.options)
            sys_key, user_key = "utterance_followup_system", "utterance_followup_user"
        else:
            sys_key, user_key = "utterance_followup_system_no_options", "utterance_followup_user_no_options"
        messages = [
            {"role": "system", "content": tmpl[sys_key].format(**ctx)},
            {"role": "user", "content": tmpl[user_key].format(**ctx)},
        ]

        utterance, belief, decision, argument_validity, confidence = "", None, None, None, None
        for _ in range(_MAX_FORMAT_RETRIES + 1):
            raw = self._chat(messages, response_format={"type": "json_object"})
            if self._show_options:
                utterance, belief, decision, argument_validity, confidence = _parse_followup(raw)
                if belief and decision and utterance and argument_validity and confidence is not None:
                    break
            else:
                utterance, belief, decision, confidence, argument_validity = _parse_followup_no_options(raw)
                if decision and utterance and argument_validity and confidence is not None:
                    break

        if not decision:
            decision = "CONTINUE"  # never force a premature close on a parse failure
        if belief:
            self._last_belief = belief

        done = (decision == "END" or burden_exceeded) and not self._force_full_turns
        if burden_exceeded:
            termination_reason = "burden_dropout"
        elif decision == "END":
            termination_reason = "agreement"
        else:
            termination_reason = None
        return (
            utterance,
            done,
            {
                "belief": self._last_belief,
                "confidence": confidence,
                "decision": decision,
                "argument_validity": argument_validity,
                "termination_reason": termination_reason,
                "cognitive_burden": tlx["overall_workload"],
                "cognitive_burden_cumulative": self._burden_cumulative,
                "cognitive_burden_dims": {d: tlx[d] for d in _TLX_DIMS},
                "cognitive_burden_short_rationale": tlx["short_rationale"],
                "cognitive_burden_n_ok": burden_n_ok,
                "cognitive_burden_n_attempted": burden_n_attempted,
                **self._episode_cfg.model_dump(),
            },
        )

    # ── Episode-end wrapper (mirrors v1.score_burden_for_last_turn) ────────────────────

    def score_burden_for_last_turn(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[float, int, int]:
        """When the episode ends by hitting max_turns right after an AI turn, _followup_turn
        (which normally scores burden as a side effect of generating the NEXT doctor turn)
        never gets called again -- the caller must invoke this directly afterward, exactly
        as scripts/run_scaling_poc.py does today for v1. Returns the raw 1-5 overall_workload
        (not normalized -- see _burden_cumulative's docstring), consistent with _followup_turn's
        "cognitive_burden" field."""
        tlx, n_ok, n_attempted = self._score_burden_tlx(case_info, history)
        return tlx["overall_workload"], n_ok, n_attempted

    # ── Stage 1: NASA-TLX burden judge, sampled self._burden_samples times, MEAN per dim ──

    def _score_burden_tlx(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[dict, int, int]:
        """Returns (tlx_dict, n_ok, n_attempted). tlx_dict always has all keys populated
        (either genuinely judged or, if every sample failed to parse, the neutral fallback --
        see _burden_judge_fallback) -- n_ok==0 is how callers detect "fallback was used",
        mirroring v1's n_ok==0 == "unscored" convention.

        Each of self._burden_samples independent judge calls is parsed with up to
        _MAX_FORMAT_RETRIES retries; a sample that exhausts retries contributes nothing
        (dropped, like v1's unparseable burden samples), not a fallback per-sample -- the
        neutral fallback is reserved for "every sample failed", not blended per-sample, to
        avoid quietly diluting genuinely-scored samples with neutral filler.
        """
        ai_utterance = history.turns[-1].text
        messages = self._build_burden_judge_prompt(case_info, history, ai_utterance)

        samples: list[dict] = []
        for _ in range(self._burden_samples):
            parsed = None
            for _retry in range(_MAX_FORMAT_RETRIES + 1):
                raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
                parsed = _parse_burden_judge(raw)
                if parsed is not None:
                    break
            if parsed is not None:
                samples.append(parsed)

        n_ok, n_attempted = len(samples), self._burden_samples
        if not samples:
            return _burden_judge_fallback(), 0, n_attempted

        mean_tlx = {dim: sum(s[dim] for s in samples) / n_ok for dim in _TLX_DIMS}
        # overall_workload recomputed from the MEANED dims (not "mean of each sample's
        # overall_workload") -- avoids double-rounding drift and matches the prompt's own
        # "mean of the 4 dims" definition applied once at the aggregate level.
        overall = round(sum(mean_tlx.values()) / len(mean_tlx), 1)
        rationale = samples[-1]["short_rationale"]  # most recent sample's rationale (illustrative only)
        return {**mean_tlx, "overall_workload": overall, "short_rationale": rationale}, n_ok, n_attempted

    def _build_burden_judge_prompt(self, case_info: CaseInfo, history: DialogueHistory, ai_utterance: str) -> list[dict[str, str]]:
        return build_burden_judge_prompt_v3(
            tmpl=_load_prompts(),
            scenario=case_info.scenario,
            persona_instruction=_load_personas()[self._persona],
            dialogue_text=_fmt_history(history),
            ai_utterance=ai_utterance,
            cumulative_burden=self._burden_cumulative,
        )
