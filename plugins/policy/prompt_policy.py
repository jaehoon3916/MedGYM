from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load

_VALID_STAGES = {"INFORM", "PROPOSE", "CONSIDER", "REVISE", "RECOMMEND", "CONFIRM", "CLOSE"}
_DEFAULT_ACTION = ("INFORM", "ask_justify", "fact")


def _parse_action(data: dict) -> tuple[str, str, str]:
    stage = str(data.get("stage", "INFORM")).upper()
    locution = str(data.get("locution", "ask_justify")).lower()
    locution_type = str(data.get("locution_type", "fact")).lower()
    if stage not in _VALID_STAGES:
        stage = "INFORM"
    return stage, locution, locution_type


class PromptPolicy(VLLMBasePlugin, PolicyPlugin):
    """LLM-based policy: generates hierarchical stage+locution from verification_template via prompt."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)

    def name(self) -> str:
        return f"prompt-policy-{self._model}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        tmpl = _load("prompt_policy")
        vt = verification_template
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(
                current_user_utterance=current_user_utterance,
                overall_relation=vt.overall_relation,
                confidence=vt.confidence,
                evidence_gaps=vt.evidence_gaps,
                reasoning=vt.reasoning,
                dialogue=dialogue_history.to_prompt(),
            )},
        ]
        raw = self._chat(messages, temperature=0.0)
        try:
            data = json.loads(raw)
            stage, locution, locution_type = _parse_action(data)
        except (json.JSONDecodeError, AttributeError):
            stage, locution, locution_type = _DEFAULT_ACTION

        action_id = f"{stage}.{locution}"
        action_prompt = (
            f"Stage: {stage} | Locution: {locution}({locution_type})\n"
            f"Apply this deliberation move in your response."
        )
        return PolicyOutput(
            stage=stage,
            locution=locution,
            locution_type=locution_type,
            action_id=action_id,
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={"policy": "prompt", "raw": raw},
        )
