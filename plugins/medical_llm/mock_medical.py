from typing import Any

from plugins.medical_llm.base import MedicalLLMPlugin
from core.schemas import CaseInfo, DialogueHistory

_ACTION_RESPONSES: dict[str, str] = {
    "ACCEPT": (
        "I agree with your clinical reasoning. The presentation is consistent with "
        "your hypothesis, and the evidence supports this direction."
    ),
    "CHALLENGE": (
        "I need to challenge that hypothesis. The clinical evidence actually points "
        "in a different direction, and here is my reasoning."
    ),
    "ASK_EVIDENCE": (
        "Before I can make a judgment, I need more information. "
        "Could you share additional clinical findings or test results?"
    ),
    "DEFER": (
        "Given the current uncertainty, I will defer final judgment. "
        "Let me summarize what we know so far."
    ),
    "SUMMARIZE": (
        "Let me summarize the current clinical reasoning: we have some evidence "
        "supporting the hypothesis, but key information is still missing."
    ),
}


class MockMedicalLLM(MedicalLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def name(self) -> str:
        return "mock-medical-llm"

    def load(self) -> None:
        pass

    def generate_medical_response(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        action_prompt: str,
    ) -> str:
        # Determine which action this prompt corresponds to
        for action_id, response in _ACTION_RESPONSES.items():
            if action_id.lower() in action_prompt.lower() or action_id in action_prompt:
                return response
        return _ACTION_RESPONSES["DEFER"]
