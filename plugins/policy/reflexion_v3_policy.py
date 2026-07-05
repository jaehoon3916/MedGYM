from __future__ import annotations

from typing import Any

from core.json_utils import safe_json_load
from core.prompt_builder import _load
from core.schemas import CaseInfo, DialogueHistory, PolicyOutput, VerificationTemplate
from plugins.policy.base import PolicyPlugin
from plugins.policy.policy_ours_v3 import _default_guidance, _parse_control_guidance, _reachable_controls
from plugins.vllm_base import VLLMBasePlugin


class ReflexionV3Policy(VLLMBasePlugin, PolicyPlugin):
    """Reflexion baseline over action_space_v3's EXTEND/RECOMMEND control (vs ours_v3).

    Two calls per turn: (1) reflect on the trajectory so far, (2) act -- pick {stage,
    action_guidance} conditioned on that reflection, parsed by the exact same helpers ours_v3
    uses. Stateless (reflection is recomputed from the dialogue each turn, never stored), thread
    -safe under concurrent eval. Mirrors reflexion_policy.py (the v2/ours_v2 analog) one-for-one.
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
        return f"reflexion-v3-policy-{self._model}{suffix}"

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
        tmpl = _load("reflexion_v3_policy")
        fv_block = self._fv_block(tmpl, verification_template)
        dialogue = dialogue_history.to_prompt_with_actions()

        # Call 1 -- reflect on the trajectory so far.
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

        # Call 2 -- act, conditioned on the reflection.
        reachable = _reachable_controls(dialogue_history, self._transitions)
        act_user = tmpl["act_user"].format(
            reachable_stages=", ".join(reachable),
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block, dialogue=dialogue,
            reflection=reflection,
        )
        act_raw = self._chat(
            [{"role": "system", "content": tmpl["act_system"]}, {"role": "user", "content": act_user}],
            temperature=0.0, response_format={"type": "json_object"},
        )
        return self._to_output(act_raw, reflection, reachable)

    def _to_output(self, raw: str, reflection: str, reachable: list[str]) -> PolicyOutput:
        control, guidance = _parse_control_guidance(raw, reachable)
        guidance = guidance or _default_guidance(control)
        return PolicyOutput(
            stage=control,
            locution="",
            locution_type="",
            action_id=control,
            action_prompt=guidance,
            confidence=1.0,
            metadata={"policy": self.name(), "control": control, "reflection": reflection, "raw": raw},
        )
