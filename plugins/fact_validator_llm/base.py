from abc import abstractmethod

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate


class FactValidatorLLMPlugin(BasePlugin):
    @abstractmethod
    def validate_facts(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
    ) -> VerificationTemplate:
        pass
