from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState, PolicyOutput


class RulePolicy(PolicyPlugin):
    def __init__(self, config: dict[str, Any], action_space: dict[str, dict[str, str]]):
        super().__init__(config, action_space)

    def name(self) -> str:
        return "rule-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        user_state: UserState,
    ) -> PolicyOutput:
        stance = getattr(user_state, "stance_toward_medical_llm", "unknown")
        certainty = getattr(user_state, "certainty", "neutral")
        facts_to_check = getattr(user_state, "facts_to_check", "unknown")

        if certainty == "uncertain":
            action_id = "ASK_EVIDENCE"
        elif facts_to_check == "false" or stance in ("skeptical", "disagree"):
            action_id = "CHALLENGE"
        elif facts_to_check == "true" and stance == "agree":
            action_id = "ACCEPT"
        else:
            action_id = "DEFER"

        return PolicyOutput(
            action_id=action_id,
            action_prompt=self.action_space[action_id]["prompt"],
            confidence=1.0,
            metadata={"policy": "rule"},
        )
