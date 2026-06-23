from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from core.json_utils import safe_json_load
from core.prompt_builder import build_load_judge_prompt
from core.schemas import CaseInfo, DialogueHistory, EpisodeConfig
from plugins.user_llm.base import UserLLMPlugin
from plugins.vllm_base import VLLMBasePlugin

_ROOT = Path(__file__).parent.parent.parent.parent
_PROMPTS_DIR = _ROOT / "prompts"
_PERSONA_DIR = _ROOT / "persona"

_MAX_FORMAT_RETRIES = 2  # MedCOBE_EMNLP uses 3; trimmed since this is a much shorter response
_VALID_LETTERS = ("A", "B", "C", "D")
_VALID_DECISIONS = ("CONTINUE", "END")


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    with open(_PROMPTS_DIR / "user_simulator_v1.yaml") as f:
        return yaml.safe_load(f)


_VALID_PERSONAS = ("veteran_attending", "exhausted_attending", "eager_resident", "burned_out_resident")
# r in the cognitive_burden model (/home/kjy/Jaehoon/cognitive_burden.txt) -- a FIXED,
# authority-driven confidence baseline, separate from q_t (the dynamic, self-reported,
# per-turn confidence). Derived from which persona role is active rather than a hand-picked
# number: attending personas (veteran/exhausted) are the senior/high-authority pair, resident
# personas (eager/burned_out) are the junior/low-authority pair -- already encoded in
# persona/personas.yaml's [Role] flavor text, not a new constant.
_AUTHORITY_PERSONAS = ("veteran_attending", "exhausted_attending")


def _effective_burden(judge_mean_0_3: float, prior_confidence: float | None, r: int) -> float:
    """c_t^eff = judge_score_normalized * (1 + r * q_t_prior).

    judge_mean_0_3 is _score_burden's raw mean (the burden judge already rates AI replies on
    a continuous 0-aligned/high-opposing scale, so no separate conflict term d_t is needed --
    see cognitive_burden.txt section 11). Normalized to [0,1] by /3. prior_confidence is q_t
    from the doctor's OWN immediately preceding turn (confidence going INTO this AI reply,
    not after) -- None means unscored that turn, in which case no amplification is applied
    (multiplier collapses to 1, not fabricated)."""
    normalized = judge_mean_0_3 / 3.0
    if prior_confidence is None:
        return normalized
    return normalized * (1 + r * prior_confidence)


@lru_cache(maxsize=1)
def _load_personas() -> dict:
    """Persona definitions are managed separately under persona/ (not prompts/) -- see
    persona/personas.yaml for the 4 named roles' behavioral text (each fixes a
    confidence x burden_sensitivity combo)."""
    with open(_PERSONA_DIR / "personas.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def _load_sparsity_persona() -> dict:
    """See persona/information_sparsity.yaml for the dense/sparse behavioral text -- mirrors
    v2's EpisodeConfig.information_sparcity concept, reimplemented as a plain instruction
    since v1 has no shard mechanism."""
    with open(_PERSONA_DIR / "information_sparsity.yaml") as f:
        return yaml.safe_load(f)


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
    """q_t for the cognitive_burden model -- clamp to [0,1] rather than reject, since LLMs
    occasionally drift slightly outside the requested range instead of refusing outright."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, v))


def _parse_opening(raw: str) -> tuple[str, str | None, str | None, float | None]:
    data = safe_json_load(raw)
    response = _clean_str(data.get("response")) or raw.strip()
    return (
        response, _letter(data.get("belief")), _clean_str(data.get("reasoning")),
        _confidence(data.get("confidence")),
    )


def _parse_followup(raw: str) -> tuple[str, str | None, str | None, str | None, float | None]:
    data = safe_json_load(raw)
    response = _clean_str(data.get("response")) or raw.strip()
    decision_raw = str(data.get("decision") or "").strip().upper()
    decision = decision_raw if decision_raw in _VALID_DECISIONS else None
    return (
        response, _letter(data.get("belief")), decision, _clean_str(data.get("reasoning")),
        _confidence(data.get("confidence")),
    )


def _parse_followup_no_options(raw: str) -> tuple[str, str | None, float | None]:
    """No options shown -> no "belief" field requested. decision + confidence only."""
    data = safe_json_load(raw)
    response = _clean_str(data.get("response")) or raw.strip()
    decision_raw = str(data.get("decision") or "").strip().upper()
    decision = decision_raw if decision_raw in _VALID_DECISIONS else None
    return response, decision, _confidence(data.get("confidence"))


class UserSimulatorV1(VLLMBasePlugin, UserLLMPlugin):
    """Primitive POC doctor simulator, ported from MedCOBE_EMNLP
    (src/simulation/retry.py + simulation_utils.py + configs/prompts/simulator_prompts.yaml).

    Unlike v2 (plugins/user_llm/user_simulator/v2.py), there is no JSON state machine, no
    shards, no EpisodeConfig-driven persona -- EpisodeConfig is accepted for interface
    compatibility but every field (including initial_fact) is ignored. Persona here is two
    independent plugin-config dials instead: `persona` (one of "veteran_attending" |
    "exhausted_attending" | "eager_resident" | "burned_out_resident", default
    "burned_out_resident") fixes both how readily the doctor revises "belief" in response to
    the AI (confidence) and how the doctor reacts to the conversation's own accumulating
    cognitive_burden (burden_sensitivity) -- the latter is what gives that measured number
    actual behavioral teeth instead of being purely descriptive (see persona/personas.yaml);
    and `information_sparsity` ("dense"|"sparse", default "dense") governs how forthcoming
    the doctor is with case details when responding (persona/information_sparsity.yaml) --
    mirrors v2's EpisodeConfig.information_sparcity concept, reimplemented as a plain
    instruction since v1 has no shard mechanism to gate disclosure on. Two more things are
    kept from the v2 design:
    1. The dialogue's own CONTINUE/END termination decision (MedCOBE's design, criteria
       text kept verbatim).
    2. cognitive_burden scoring (build_load_judge_prompt -- MedCOBE has no equivalent; added
       here, scored using this plugin's own configured model/client).

    Departure from the literal MedCOBE port: turn 0's initial belief is NOT injected from
    outside (MedCOBE's `target_belief`). Instead the doctor independently judges the case
    (correct_option/answer/explanation withheld from the prompt) and states its own belief,
    reported in a "belief" JSON field. This gives a natural "doctor-alone" baseline -- parallel to the
    ai_alone_accuracy already computed in scripts/run_poc_multiturn.py -- instead of an
    experimenter-forced correct/incorrect split.
    """

    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        self._episode_cfg: EpisodeConfig = EpisodeConfig()
        self._force_close: bool = False
        self._last_belief: str | None = None
        self._burden_samples: int = int(config.get("burden_samples_per_item", 3))
        # The doctor's own CONTINUE/END judgment ("[Termination Decision]" in
        # prompts/user_simulator_v1.yaml) only models converged-vs-repetitive reasoning -- no
        # burden-driven dropout, satisficing, or deference. When force_full_turns is set, the
        # "decision" field is still parsed and recorded (informative -- "the doctor wanted to
        # stop here") but is never used to end the episode; only max_turns does. Use this for
        # scaling-curve measurements where checkpoints need real dialogue at every turn count,
        # not a collapse to whatever turn the doctor happened to self-report END at.
        self._force_full_turns: bool = bool(config.get("force_full_turns", False))
        # When false, the doctor never sees the multiple-choice options/question either --
        # only the free-text scenario. No "belief" field is requested in that case (nothing
        # lettered to point it at); doctor-alone correctness then has to be inferred from the
        # free-text opening statement via final_judge instead of a self-reported letter (see
        # scripts/run_scaling_poc.py). A/B test for how much seeing the options itself (not
        # turn count) shapes the doctor's behavior. NOT to be confused with medical_llm, which
        # never sees options regardless -- this toggle is the doctor's, not the AI's.
        self._show_options: bool = bool(config.get("show_options", True))
        # Persona, axis 1/2: a single named role fixing both how readily the doctor revises
        # "belief" in response to the AI's pushback (confidence) and how it reacts to the
        # conversation's accumulating cognitive_burden (burden_sensitivity) -- see
        # persona/personas.yaml for the 2x2 grid and behavioral text. Falls back to
        # "burned_out_resident" if misconfigured.
        persona = str(config.get("persona", "burned_out_resident")).strip().lower()
        self._persona: str = persona if persona in _VALID_PERSONAS else "burned_out_resident"
        # Persona, axis 2/2 (independent of the persona role above): governs how forthcoming
        # the doctor is with case details/findings/reasoning when responding. "dense" = shares
        # freely and proactively; "sparse" = holds back, answers narrowly. Mirrors v2's
        # EpisodeConfig.information_sparcity concept (see persona/information_sparsity.yaml).
        # Falls back to "dense" if misconfigured (matches v2's EpisodeConfig default).
        information_sparsity = str(config.get("information_sparsity", "dense")).strip().lower()
        self._information_sparsity: str = (
            information_sparsity if information_sparsity in ("dense", "sparse") else "dense"
        )
        # Running total shown to the model each followup turn so "sensitive" personas have an
        # actual number to react to, not just a vague notion of "this has been a long talk".
        self._cumulative_burden: float = 0.0

    def reset_episode(self, episode_config: EpisodeConfig) -> None:
        self._episode_cfg = episode_config
        self._force_close = False
        self._last_belief = None
        self._cumulative_burden = 0.0

    def force_close(self) -> None:
        self._force_close = True

    def name(self) -> str:
        return f"user-simulator-v1-{self._model}"

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

    # ── Turn 0: independent opening judgment (no AI input yet, MedCOBE's "turn 1") ──────

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
            sys_key, user_key = "turn0_system", "turn0_user"
        else:
            sys_key, user_key = "turn0_system_no_options", "turn0_user_no_options"
        messages = [
            {"role": "system", "content": tmpl[sys_key].format(**ctx)},
            {"role": "user", "content": tmpl[user_key].format(**ctx)},
        ]
        raw = self._chat(messages, response_format={"type": "json_object"})
        # Both variants return JSON; the no_options prompt simply never asks for "belief", so
        # it naturally comes back None via .get() -- no separate parse path needed.
        utterance, belief, reasoning, confidence = _parse_opening(raw)
        self._last_belief = belief or self._last_belief
        # confidence (q_t) does NOT fall back to a stale value when missing/unparseable --
        # unlike belief, a silently-reused old confidence would corrupt the cognitive_burden
        # formula's q_t term with a number the doctor never actually reported this turn.
        # None here means "unscored", not "unchanged".
        return utterance, False, {
            "belief": self._last_belief, "confidence": confidence,
            "decision": None, "reasoning": reasoning,
            **self._episode_cfg.model_dump(),
        }

    # ── Turn 1+: reactive followup, decides CONTINUE/END (MedCOBE's "turn 2+") ──────────

    def _followup_turn(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[str, bool, dict[str, Any]]:
        burden_raw, burden_n_ok, burden_n_attempted = self._score_burden(case_info, history)
        # q_t_prior: the doctor's OWN confidence going INTO this AI reply -- read off its own
        # immediately preceding turn (turns[-2]; turns[-1] is the AI reply _score_burden just
        # scored), not a separately-cached value. None if that turn's confidence was unscored.
        prior_confidence = getattr(history.turns[-2].user_state, "confidence", None) if history.turns[-2].user_state else None
        r = 1 if self._persona in _AUTHORITY_PERSONAS else 0
        burden = _effective_burden(burden_raw, prior_confidence, r)
        # Accumulate BEFORE building ctx, so the doctor reacts (per its burden_sensitivity
        # persona) to the running total including the AI turn it's responding to right now.
        self._cumulative_burden += burden

        tmpl = _load_prompts()
        ctx = dict(
            scenario=case_info.scenario,
            image_caption=case_info.caption or "N/A",
            question=case_info.question,
            dialogue_text=_fmt_history(history),
            persona_instruction=_load_personas()[self._persona],
            sparsity_instruction=_load_sparsity_persona()[self._information_sparsity],
            cumulative_burden=round(self._cumulative_burden, 2),
        )
        if self._show_options:
            ctx["options_text"] = _fmt_options(case_info.options)
            sys_key, user_key = "followup_system", "followup_user"
        else:
            sys_key, user_key = "followup_system_no_options", "followup_user_no_options"
        messages = [
            {"role": "system", "content": tmpl[sys_key].format(**ctx)},
            {"role": "user", "content": tmpl[user_key].format(**ctx)},
        ]

        utterance, belief, decision, reasoning, confidence = "", None, None, None, None
        for _ in range(_MAX_FORMAT_RETRIES + 1):
            raw = self._chat(messages, response_format={"type": "json_object"})
            if self._show_options:
                utterance, belief, decision, reasoning, confidence = _parse_followup(raw)
                if belief and decision and utterance and confidence is not None:
                    break
            else:
                utterance, decision, confidence = _parse_followup_no_options(raw)
                if decision and utterance and confidence is not None:
                    break

        if not decision:
            decision = "CONTINUE"  # never force a premature close on a parse failure
        if belief:
            self._last_belief = belief
        # confidence (q_t) does NOT fall back to a stale value when missing/unparseable even
        # after exhausting retries -- None means "unscored this turn", not "unchanged from
        # last turn" (a silently-reused old value would corrupt the cognitive_burden
        # formula's q_t term with a number the doctor never actually reported this turn).

        done = (decision == "END") and not self._force_full_turns
        return (
            utterance,
            done,
            {
                "belief": self._last_belief,
                "confidence": confidence,
                "decision": decision,
                "reasoning": reasoning,
                "cognitive_burden": burden,
                "cognitive_burden_raw": burden_raw,
                "cognitive_burden_n_ok": burden_n_ok,
                "cognitive_burden_n_attempted": burden_n_attempted,
                **self._episode_cfg.model_dump(),
            },
        )

    # ── Cognitive burden (kept from the v2 design; MedCOBE has no equivalent) ──────────

    def score_burden_for_last_turn(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[float, int, int]:
        """Public wrapper for episode-end use: when the episode ends by hitting max_turns
        right after an AI turn, _followup_turn (which normally scores burden as a side
        effect of generating the NEXT doctor turn) never gets called again for that last AI
        turn -- the caller must invoke this directly afterward. See
        scripts/run_scaling_poc.py's use of this when closed_by == "max_turns".

        Applies the same c_t^eff transform _followup_turn does (see _effective_burden) --
        otherwise only the episode's LAST turn would be left as a raw, un-amplified burden
        value, breaking comparability with every other turn in the same episode."""
        burden_raw, n_ok, n_attempted = self._score_burden(case_info, history)
        prior_confidence = getattr(history.turns[-2].user_state, "confidence", None) if history.turns[-2].user_state else None
        r = 1 if self._persona in _AUTHORITY_PERSONAS else 0
        return _effective_burden(burden_raw, prior_confidence, r), n_ok, n_attempted

    def _score_burden(self, case_info: CaseInfo, history: DialogueHistory) -> tuple[float, int, int]:
        """Score the AI's latest turn's burden on the doctor.

        turns[-1] is always the AI's latest reply and turns[-2] the doctor's own previous
        utterance -- this method is only called from _followup_turn, i.e. once at least one
        (doctor, AI) pair already exists in history.

        Returns (mean_burden, n_ok, n_attempted). n_ok < n_attempted means some judge calls
        failed to parse -- distinguishing that from a genuine score of 0 matters: a silent
        all-failed case previously collapsed indistinguishably to 0.0 (see the earlier
        max_tokens truncation bug in v2/_build_state, same failure shape). Callers should
        treat n_ok == 0 as "unscored", not "burden is zero".
        """
        ai_utterance = history.turns[-1].text
        doctor_utterance = history.turns[-2].text if len(history.turns) >= 2 else ""
        messages = build_load_judge_prompt(ai_utterance, doctor_utterance, case_info.scenario)
        scores: list[int] = []
        for _ in range(self._burden_samples):
            raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
            data = safe_json_load(raw)
            try:
                scores.append(int(round(float(data.get("cognitive_burden")))))
            except (TypeError, ValueError):
                continue
        mean_burden = sum(scores) / len(scores) if scores else 0.0
        return mean_burden, len(scores), self._burden_samples
