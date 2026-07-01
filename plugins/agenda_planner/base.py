from __future__ import annotations

from abc import abstractmethod

from plugins.base import BasePlugin
from core.schemas import CaseInfo, Agenda


class DisagreementAnalyzerPlugin(BasePlugin):
    @abstractmethod
    def analyze(
        self,
        case_info: CaseInfo,
        human_opinion: str,
        ai_opinion: str,
    ) -> Agenda:
        pass
