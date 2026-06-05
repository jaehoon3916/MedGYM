from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState, PolicyOutput


class NaivePolicy(PolicyPlugin):
    """No policy: formats user_state directly into action_prompt for Medical LLM."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, dict[str, str]]):
        super().__init__(config, action_space)

    def name(self) -> str:
        return "naive-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        user_state: UserState,
    ) -> PolicyOutput:
        state_dump = user_state.model_dump()
        state_lines = "\n".join(
            f"- {k}: {v}" for k, v in state_dump.items() if v and v != [] and k != "summary"
        )
        action_prompt = (
            f"The clinician's current state is as follows:\n{state_lines}\n\n"
            f"Respond appropriately based on the clinician's state."
        )
        return PolicyOutput(
            action_id="DEFER",
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={"policy": "naive"},
        )
