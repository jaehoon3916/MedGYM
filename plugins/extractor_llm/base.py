from abc import abstractmethod

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory, UserState


class ExtractorLLMPlugin(BasePlugin):
    @abstractmethod
    def extract_user_state(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
    ) -> UserState:
        pass
