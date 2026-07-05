from __future__ import annotations

from typing import Any

from core.prompt_builder import _load
from core.schemas import CaseInfo, DialogueHistory, PolicyOutput, VerificationTemplate
from plugins.policy.base import PolicyPlugin
from plugins.policy.policy_ours_v3 import _default_guidance, _parse_control_guidance, _reachable_controls
from plugins.vllm_base import VLLMBasePlugin


class ReactControlV3Policy(VLLMBasePlugin, PolicyPlugin):
    """ReAct baseline over action_space_v3's EXTEND/RECOMMEND control (vs ours_v3).

    Prompted planner: one Thought->Action call per turn, emitting {stage, action_guidance} parsed
    by the exact same helpers ours_v3 uses -- so the ONLY difference vs the learned policy is
    prompt-reasoning vs GRPO weights, not the action space or the output contract. Mirrors
    react_control_policy.py (the v2/ours_v2 analog) one-for-one.
    """

    needs_verification = True

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._fact_validator_on: bool = bool(config.get("use_fact_validator", True))
        self.needs_verification = self._fact_validator_on
        self._transitions: dict[str, list[str]] = action_space.get("transitions", {})

    def name(self) -> str:
        suffix = "" if self._fact_validator_on else "-no-validator"
        return f"react-control-v3-policy-{self._model}{suffix}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        tmpl = _load("react_control_v3_policy")
        reachable = _reachable_controls(dialogue_history, self._transitions)

        fv_block = ""
        if self._fact_validator_on and verification_template is not None:
            fv_block = tmpl["fact_validation_block"].format(
                overall_relation=verification_template.overall_relation,
                confidence=verification_template.confidence,
                reasoning=verification_template.reasoning,
            )
        user = tmpl["user"].format(
            reachable_stages=", ".join(reachable),
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
        )
        raw = self._chat(
            [{"role": "system", "content": tmpl["system"]}, {"role": "user", "content": user}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        return self._to_output(raw, reachable)

    def _to_output(self, raw: str, reachable: list[str]) -> PolicyOutput:
        control, guidance = _parse_control_guidance(raw, reachable)
        guidance = guidance or _default_guidance(control)
        return PolicyOutput(
            stage=control,
            locution="",
            locution_type="",
            action_id=control,
            action_prompt=guidance,
            confidence=1.0,
            metadata={"policy": self.name(), "control": control, "raw": raw},
        )
