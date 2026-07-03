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
from core.json_utils import safe_json_load
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load
from core.reward_align import ctx_from_history


class ReflexionPolicy(VLLMBasePlugin, PolicyPlugin):
    """Reflexion baseline over the SAME 3 control stages (EXPAND/CHALLENGE/CONVERGE) as ours_v2.

    Two calls per turn: (1) reflect on the trajectory so far, (2) act -- pick {stage,
    action_guidance} conditioned on that reflection, parsed/mapped by the exact same helpers
    ours_v2 uses. The reflection is recomputed from the dialogue each turn (the dialogue is the full
    record of past stages + their outcomes), so the policy is stateless and thread-safe: no
    cross-episode reflection memory to leak under concurrent eval. Extra reflect call is inherent to
    the method (Reflexion spends reasoning to self-critique before acting).
    """

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._fact_validator_on: bool = bool(config.get("use_fact_validator", True))
        self.needs_verification = self._fact_validator_on

    def name(self) -> str:
        suffix = "" if self._fact_validator_on else "-no-validator"
        return f"reflexion-policy-{self._model}{suffix}"

    def _fv_block(self, tmpl: dict, vt: VerificationTemplate | None) -> str:
        if not (self._fact_validator_on and vt is not None):
            return ""
        return tmpl["fact_validation_block"].format(
            overall_relation=vt.overall_relation, confidence=vt.confidence, reasoning=vt.reasoning,
        )

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        tmpl = _load("reflexion_policy")
        fv_block = self._fv_block(tmpl, verification_template)
        dialogue = dialogue_history.to_prompt_with_actions()

        # Call 1 — reflect on the trajectory so far.
        reflect_user = tmpl["reflect_user"].format(
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block, dialogue=dialogue,
        )
        reflect_raw = self._chat(
            [{"role": "system", "content": tmpl["reflect_system"]},
             {"role": "user", "content": reflect_user}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        reflection = str((safe_json_load(reflect_raw) or {}).get("reflection", "")).strip() or "(no reflection)"

        # Call 2 — act, conditioned on the reflection.
        ctx = ctx_from_history(dialogue_history)
        reachable = ", ".join(_reachable_controls(ctx))
        turns_so_far = sum(1 for t in dialogue_history.turns if t.speaker == "medical")
        act_user = tmpl["act_user"].format(
            reachable_controls=reachable,
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block, dialogue=dialogue,
            reflection=reflection, turns_so_far=turns_so_far,
        )
        act_raw = self._chat(
            [{"role": "system", "content": tmpl["act_system"]}, {"role": "user", "content": act_user}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        return self._to_output(act_raw, reflection, dialogue_history)

    def _to_output(self, raw: str, reflection: str, dialogue_history: DialogueHistory) -> PolicyOutput:
        control, guidance = _parse_control_guidance(raw)
        ctx = ctx_from_history(dialogue_history)
        if control not in _reachable_controls(ctx):
            control = "EXPAND"
        mcb_stage, mcb_loc, mcb_type = _CONTROL_TO_MCB.get(control, _FALLBACK_MCB)
        po = _build_policy_output(mcb_stage, mcb_loc, mcb_type, "reflexion", raw, self.action_space)
        if guidance:
            po.action_prompt = guidance
        po.metadata["control"] = control
        po.metadata["reflection"] = reflection
        po.metadata["policy"] = self.name()
        return po
