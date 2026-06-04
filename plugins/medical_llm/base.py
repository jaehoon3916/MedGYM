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
    ) -> str:
        pass
