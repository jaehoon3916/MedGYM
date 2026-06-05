from __future__ import annotations

from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState, PolicyOutput
from core.prompt_builder import _load

_VALID_ACTIONS = {"ACCEPT", "CHALLENGE", "ASK_EVIDENCE", "DEFER", "SUMMARIZE"}


class QwenPolicy(VLLMBasePlugin, PolicyPlugin):
    """Qwen3-8B policy: baseline / sft / full depending on checkpoint loaded."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, dict[str, str]]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._mode: str = config.get("mode", "baseline")

    def name(self) -> str:
        return f"qwen-policy-{self._mode}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        user_state: UserState,
    ) -> PolicyOutput:
        tmpl = _load("baseline_policy")
        state_lines = "\n".join(
            f"  {k}: {v}" for k, v in user_state.model_dump().items() if v and v != []
        )
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(user_state=state_lines)},
        ]
        raw = self._chat(messages, temperature=0.0, max_tokens=16)
        action_id = raw.strip().upper().split()[0] if raw.strip() else "DEFER"
        if action_id not in _VALID_ACTIONS:
            action_id = "DEFER"

        return PolicyOutput(
            action_id=action_id,
            action_prompt=self.action_space[action_id]["prompt"],
            confidence=1.0,
            metadata={"policy": self._mode},
        )
