from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from plugins.policy.prompt_policy import _parse_action, _DEFAULT_ACTION
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load


class DeliberationLLMPolicy(VLLMBasePlugin, PolicyPlugin):
    """LLM-based deliberation policy: selects a stage/locution AND generates specific
    behavioral guidance (action_guidance) for the medical AI in a single call.

    Improves on PromptPolicy by having the LLM write a concrete 2-4 sentence instruction
    instead of the thin "Stage: X | Locution: Y\\nApply this." fallback text.
    """

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)

    def name(self) -> str:
        return f"deliberation-llm-policy-{self._model}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        tmpl = _load("deliberation_llm_policy")
        vt = verification_template
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(
                current_user_utterance=current_user_utterance,
                overall_relation=vt.overall_relation,
                confidence=vt.confidence,
                reasoning=vt.reasoning,
                evidence_gaps=vt.evidence_gaps,
                dialogue=dialogue_history.to_prompt(),
            )},
        ]
        raw = self._chat(messages, temperature=0.0,
                         response_format={"type": "json_object"})
        try:
            data = json.loads(raw)
            stage, locution, locution_type = _parse_action(data)
            action_guidance = str(data.get("action_guidance", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            stage, locution, locution_type = _DEFAULT_ACTION
            action_guidance = ""

        # fallback: mirror prompt_policy's thin text so medical LLM is never guidanceless
        if not action_guidance:
            action_guidance = (
                f"Stage: {stage} | Locution: {locution}({locution_type})\n"
                f"Apply this deliberation move in your response."
            )

        return PolicyOutput(
            stage=stage,
            locution=locution,
            locution_type=locution_type,
            action_id=f"{stage}.{locution}",
            action_prompt=action_guidance,
            confidence=1.0,
            metadata={
                "policy": "deliberation_llm",
                "model": self._model,
                "raw": raw,
            },
        )
