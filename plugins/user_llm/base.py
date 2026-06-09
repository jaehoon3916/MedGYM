from abc import abstractmethod
from typing import Any

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory


class UserLLMPlugin(BasePlugin):
    def reset_episode(self, episode_config: Any) -> None:
        pass

    def force_close(self) -> None:
        pass

    @abstractmethod
    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> tuple[str, bool, dict[str, Any] | None]:
        """Returns (utterance, done, user_state_dict)."""
        pass
