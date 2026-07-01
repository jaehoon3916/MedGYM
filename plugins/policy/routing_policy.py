from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, PolicyOutput

# The two specialists, in the RRS sense: an alignment-optimized behavior (defer/agree, builds
# trust) and a complementarity-optimized behavior (correct/push, boosts performance). Stage/
# locution mirror rule_policy's supported->INFORM.assert and contradicted->CONSIDER.assert so
# r_align scores them consistently with the existing context-blind baseline.
_ALIGNED_ACTION = ("INFORM", "assert", "fact")
_COMPLEMENTARY_ACTION = ("CONSIDER", "assert", "evaluation")

_ALIGNED_PROMPT = (
    "The clinician's reasoning appears sound and they hold it with confidence. Affirm and "
    "build on their assessment, and defer to their clinical judgment. Do not introduce "
    "friction or push an alternative unless they ask for one."
)
_COMPLEMENTARY_PROMPT = (
    "Critically evaluate the clinician's latest claim against established medical evidence. "
    "If it conflicts with the evidence, assert the correct alternative directly with specific "
    "clinical reasoning, and do not defer."
)


class RoutingPolicy(PolicyPlugin):
    """Adaptive routing baseline, adapted from the Rational Routing Shortcut (RRS) of
    Amin, Yin & Khanna (AAAI 2026, arXiv:2602.20104). RRS routes between an alignment-
    optimized specialist and a complementarity-optimized specialist by comparing the two
    specialists' OWN confidences -- m(x) = aligned if C_a(x) >= C_c(x) else complementary --
    with no tuned threshold (the whole point of the "shortcut" is that no human state and no
    calibrated tau are needed).

    Our instantiation maps:
      C_a (aligned pole)        := the doctor simulator's self-reported confidence q_t in
                                   [0,1], read off the most recent user turn's user_state.
      C_c (complementary pole)  := fixed at 0.5 (neutral). No fact-validator is used.

    Route ALIGNED (defer/affirm) when doctor confidence >= 0.5; COMPLEMENTARY otherwise.
    NOTE: doctor confidence is NOT correctness -- a confidently-wrong doctor routes ALIGNED.
    Zero extra LLM calls.
    """

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)
        # Used when the most recent user turn carries no confidence (e.g. a parse miss):
        # default 0.5 ties with an ambiguous C_c (-> ALIGNED via the >= rule).
        self._doctor_confidence_fallback = float(config.get("doctor_confidence_fallback", 0.5))

    def name(self) -> str:
        return "routing-policy"

    def load(self) -> None:
        pass

    def _doctor_confidence(self, history: DialogueHistory) -> float | None:
        """q_t off the most recent user turn (UserState has extra='allow', so the v3
        simulator's `confidence` field is an attribute when present)."""
        for turn in reversed(history.turns):
            if turn.speaker == "user" and turn.user_state is not None:
                c = getattr(turn.user_state, "confidence", None)
                return float(c) if isinstance(c, (int, float)) else None
        return None

    needs_verification = False

    def select_action(  # type: ignore[override]
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
    ) -> PolicyOutput:
        doctor_conf_raw = self._doctor_confidence(dialogue_history)
        c_a = doctor_conf_raw if doctor_conf_raw is not None else self._doctor_confidence_fallback
        c_c = 0.5  # neutral without fact-validator

        if c_a >= c_c:
            route = "aligned"
            stage, locution, locution_type = _ALIGNED_ACTION
            action_prompt = _ALIGNED_PROMPT
        else:
            route = "complementary"
            stage, locution, locution_type = _COMPLEMENTARY_ACTION
            action_prompt = _COMPLEMENTARY_PROMPT

        return PolicyOutput(
            stage=stage,
            locution=locution,
            locution_type=locution_type,
            action_id=f"{stage}.{locution}",
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={
                "policy": "routing",
                "route": route,
                "c_aligned": round(c_a, 4),
                "c_complementary": c_c,
                "doctor_confidence_raw": doctor_conf_raw,
            },
        )
