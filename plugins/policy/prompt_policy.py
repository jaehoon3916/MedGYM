from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState, PolicyOutput
from core.prompt_builder import _load

_VALID_ACTIONS = {"ACCEPT", "CHALLENGE", "ASK_EVIDENCE", "DEFER", "SUMMARIZE"}


class PromptPolicy(VLLMBasePlugin, PolicyPlugin):
    """LLM-based policy: generates action from user_state via prompt."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, dict[str, str]]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)

    def name(self) -> str:
        return f"prompt-policy-{self._model}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        user_state: UserState,
    ) -> PolicyOutput:
        tmpl = _load("prompt_policy")
        state_lines = "\n".join(
            f"  {k}: {v}" for k, v in user_state.model_dump().items() if v and v != []
        )
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(
                user_state=state_lines,
                dialogue=dialogue_history.to_prompt(),
            )},
        ]
        raw = self._chat(messages, temperature=0.0)
        try:
            data = json.loads(raw)
            action_id = data.get("action_id", "DEFER").upper()
            if action_id not in _VALID_ACTIONS:
                action_id = "DEFER"
        except (json.JSONDecodeError, AttributeError):
            action_id = "DEFER"

        return PolicyOutput(
            action_id=action_id,
            action_prompt=self.action_space[action_id]["prompt"],
            confidence=1.0,
            metadata={"policy": "prompt", "raw": raw},
        )
