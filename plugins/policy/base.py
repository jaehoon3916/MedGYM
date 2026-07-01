from abc import abstractmethod
from typing import Any

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput


class PolicyPlugin(BasePlugin):
    needs_verification: bool = True  # False → select_action skips verification_template param

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config)
        self.action_space = action_space

    @abstractmethod
    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        pass
