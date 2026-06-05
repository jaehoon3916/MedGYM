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
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> str:
        pass
