from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from plugins.policy.prompt_policy import _parse_action, _DEFAULT_ACTION
from core.schemas import CaseInfo, DialogueHistory, AgendaItem, PolicyOutput
from core.prompt_builder import _load


class AgendaActionPolicy(VLLMBasePlugin, PolicyPlugin):
    """LLM deliberation policy focused on the current agenda item.

    Replaces fact-validator input with the current AgendaItem so the policy plans
    moves that address one specific point of disagreement at a time. Structurally
    mirrors DeliberationLLMPolicy but takes AgendaItem instead of VerificationTemplate.
    """

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)

    def name(self) -> str:
        return f"agenda-action-policy-{self._model}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template=None,
        current_item: AgendaItem | None = None,
    ) -> PolicyOutput:
        tmpl = _load("agenda_action_policy")

        if current_item is not None:
            item_id = current_item.id
            item_issue = current_item.issue
            item_human_position = current_item.human_position or "(not specified)"
            item_ai_position = current_item.ai_position or "(not specified)"
            item_status = current_item.status
        else:
            item_id = 1
            item_issue = "Overall clinical approach"
            item_human_position = "(not specified)"
            item_ai_position = "(not specified)"
            item_status = "unresolved"

        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(
                item_id=item_id,
                item_issue=item_issue,
                item_human_position=item_human_position,
                item_ai_position=item_ai_position,
                item_status=item_status,
                current_user_utterance=current_user_utterance,
                dialogue=dialogue_history.to_prompt_with_actions(),
            )},
        ]
        raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
        try:
            data = json.loads(raw)
            stage, locution, locution_type = _parse_action(data)
            action_guidance = str(data.get("action_guidance", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            stage, locution, locution_type = _DEFAULT_ACTION
            action_guidance = ""

        if not action_guidance:
            action_guidance = (
                f"Stage: {stage} | Locution: {locution}({locution_type})\n"
                f"Focus on agenda item #{item_id}: {item_issue}"
            )

        return PolicyOutput(
            stage=stage,
            locution=locution,
            locution_type=locution_type,
            action_id=f"{stage}.{locution}",
            action_prompt=action_guidance,
            confidence=1.0,
            metadata={
                "policy": "agenda_action",
                "model": self._model,
                "agenda_item_id": item_id,
                "agenda_item_status": item_status,
                "raw": raw,
            },
        )
