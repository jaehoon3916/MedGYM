from __future__ import annotations

from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from plugins.policy.qwen_policy import _build_policy_output
from plugins.policy.policy_ours_v2 import (
    _CONTROL_TO_MCB,
    _FALLBACK_MCB,
    _parse_control_guidance,
    _reachable_controls,
)
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load
from core.reward_align import ctx_from_history


class ReactControlPolicy(VLLMBasePlugin, PolicyPlugin):
    """ReAct baseline over the SAME 3 control stages (EXPAND/CHALLENGE/CONVERGE) as ours_v2.

    Prompted planner: one Thought->Action call per turn, emitting {stage, action_guidance} parsed
    and mapped to the McBurney realization by the exact same helpers ours_v2 uses -- so the ONLY
    difference vs the learned policy is prompt-reasoning vs GRPO weights, not the action space or
    the output contract. use_fact_validator (default true) mirrors ours_v2 so both planners see the
    same state signal v_t; set false for the no-validator ablation.
    """

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._fact_validator_on: bool = bool(config.get("use_fact_validator", True))
        self.needs_verification = self._fact_validator_on

    def name(self) -> str:
        suffix = "" if self._fact_validator_on else "-no-validator"
        return f"react-control-policy-{self._model}{suffix}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        tmpl = _load("react_control_policy")
        ctx = ctx_from_history(dialogue_history)
        reachable = ", ".join(_reachable_controls(ctx))
        turns_so_far = sum(1 for t in dialogue_history.turns if t.speaker == "medical")

        fv_block = ""
        if self._fact_validator_on and verification_template is not None:
            fv_block = tmpl["fact_validation_block"].format(
                overall_relation=verification_template.overall_relation,
                confidence=verification_template.confidence,
                reasoning=verification_template.reasoning,
            )
        user = tmpl["user"].format(
            reachable_controls=reachable,
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
            turns_so_far=turns_so_far,
        )
        raw = self._chat(
            [{"role": "system", "content": tmpl["system"]}, {"role": "user", "content": user}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        return self._to_output(raw, dialogue_history)

    def _to_output(self, raw: str, dialogue_history: DialogueHistory) -> PolicyOutput:
        control, guidance = _parse_control_guidance(raw)
        ctx = ctx_from_history(dialogue_history)
        if control not in _reachable_controls(ctx):
            control = "EXPAND"
        mcb_stage, mcb_loc, mcb_type = _CONTROL_TO_MCB.get(control, _FALLBACK_MCB)
        po = _build_policy_output(mcb_stage, mcb_loc, mcb_type, "react_control", raw, self.action_space)
        if guidance:
            po.action_prompt = guidance   # the medical AI executes the generated guidance
        po.metadata["control"] = control
        po.metadata["policy"] = self.name()
        return po
