"""API-backed v2 teacher policy — SFT distillation source.

Emits the SAME v2 control action as PolicyOursV2 ({stage ∈ EXPAND/CHALLENGE/CONVERGE, action_guidance})
but generates it with a REMOTE model (e.g. deepseek-v4) via the OpenRouter chat API instead of a
local Qwen. Used only to produce golden SFT demonstrations; the student (PolicyOursV2/Qwen) is then
trained to imitate the resulting {messages, JSON action} pairs.

Design (see project_medgym_reward_v2_plan):
- Reuses the ours_v2 prompt template (_load("policy_ours_v2")) + control parser so teacher and
  student share EXACTLY the same action schema — the student's SFT prompt == this teacher's prompt.
- Reasons WITHOUT gold (no reveal_correct_option): demonstrations must be attainable by the
  deployable student. The oracle (argmax r_align) teacher is NOT used — it is broken on v2 (BASE is
  keyed on McBurney stages, so every v2 control scores 0 → degenerate always-EXPAND).
"""
from __future__ import annotations

from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from plugins.policy.qwen_policy import _build_policy_output
from core.prompt_builder import _load
from core.reward_align import ctx_from_history
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from plugins.policy.policy_ours_v2 import (
    _reachable_controls,
    _parse_control_guidance,
    _CONTROL_TO_MCB,
    _FALLBACK_MCB,
)


class OursV2APITeacher(VLLMBasePlugin, PolicyPlugin):
    """Prompted deepseek-v4 (or any API model) policy over the 3-way EXPAND/CHALLENGE/CONVERGE control."""

    needs_verification = True

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)          # sets _client / _model / _temperature / _max_tokens
        PolicyPlugin.__init__(self, config, action_space)
        self._mode = config.get("mode", "teacher")
        # v_t (fact-validator relation) is part of state s_t; toggle off for a minimal smoke.
        self._fact_validator_on: bool = bool(config.get("use_fact_validator", True))

    def name(self) -> str:
        return f"ours-v2-api-teacher-{self._model}"

    def load(self) -> None:
        pass  # remote model; nothing to load

    def build_messages(
        self,
        vt: VerificationTemplate | None,
        current_user_utterance: str,
        dialogue_history: DialogueHistory,
    ) -> list[dict[str, str]]:
        """The ours_v2 state-s_t prompt. Identical to PolicyOursV2._messages_ours so the student's
        SFT prompt matches this teacher's prompt verbatim."""
        tmpl = _load("policy_ours_v2")
        ctx = ctx_from_history(dialogue_history)
        reachable = ", ".join(_reachable_controls(ctx))
        turns_so_far = sum(1 for t in dialogue_history.turns if t.speaker == "medical")
        fv_block = ""
        if self._fact_validator_on and vt is not None:
            fv_block = tmpl["fact_validation_block"].format(
                overall_relation=vt.overall_relation,
                confidence=vt.confidence,
                reasoning=vt.reasoning,
            )
        user = tmpl["user"].format(
            reachable_controls=reachable,
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
            turns_so_far=turns_so_far,
        )
        return [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": user},
        ]

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        messages = self.build_messages(verification_template, current_user_utterance, dialogue_history)
        raw = self._chat(messages, response_format={"type": "json_object"})
        control, guidance = _parse_control_guidance(raw)
        ctx = ctx_from_history(dialogue_history)
        if control not in _reachable_controls(ctx):
            control = "EXPAND"
        mcb_stage, mcb_loc, mcb_type = _CONTROL_TO_MCB.get(control, _FALLBACK_MCB)
        po = _build_policy_output(mcb_stage, mcb_loc, mcb_type, self._mode, raw, self.action_space)
        if guidance:
            po.action_prompt = guidance
        po.metadata["control"] = control
        po.metadata["policy"] = self.name()
        return po
