from abc import abstractmethod
from typing import Any

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory, UserState, PolicyOutput


class PolicyPlugin(BasePlugin):
    def __init__(self, config: dict[str, Any], action_space: dict[str, dict[str, str]]):
        super().__init__(config)
        self.action_space = action_space

    @abstractmethod
    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        user_state: UserState,
    ) -> PolicyOutput:
        pass
