from abc import abstractmethod
from typing import Any

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory


class UserLLMPlugin(BasePlugin):
    @abstractmethod
    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        system_utterance: str,
        user_profile: dict[str, Any] | None = None,
    ) -> str:
        pass
