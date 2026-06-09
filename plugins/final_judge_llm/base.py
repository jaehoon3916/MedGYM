from abc import abstractmethod

from plugins.base import BasePlugin
from core.schemas import CaseInfo, DialogueHistory, FinalJudgement


class FinalJudgeLLMPlugin(BasePlugin):
    @abstractmethod
    def judge(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
    ) -> FinalJudgement:
        """Read the completed dialogue and decide which option it converged on."""
        ...
