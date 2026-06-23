from abc import abstractmethod

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory


class MedicalLLMPlugin(BasePlugin):
    @abstractmethod
    def generate_medical_response(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        action_prompt: str,
        current_user_utterance: str,
    ) -> tuple[str, str | None, str | None]:
        """Returns (response_text, belief_option_letter_or_None, reasoning_or_None)."""
        pass
