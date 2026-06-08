from typing import Any

from plugins.user_llm.base import UserLLMPlugin
from core.schemas import CaseInfo, DialogueHistory

_RESPONSES = [
    "I think the patient might have condition A based on the symptoms.",
    "I'm not sure about that. Can you explain the reasoning further?",
    "I disagree — the evidence points toward condition B.",
    "That makes sense. I agree with your assessment.",
    "I'm uncertain. What additional tests would help clarify?",
]


class MockUserLLM(UserLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._call_count = 0

    def name(self) -> str:
        return "mock-user-llm"

    def load(self) -> None:
        pass

    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        response = _RESPONSES[self._call_count % len(_RESPONSES)]
        self._call_count += 1
        return response, False
