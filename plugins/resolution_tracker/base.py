from __future__ import annotations

from abc import abstractmethod

from plugins.base import BasePlugin
from core.schemas import AgendaItem


class ResolutionTrackerPlugin(BasePlugin):
    @abstractmethod
    def check_resolved(
        self,
        agenda_item: AgendaItem,
        recent_medical: str,
        recent_user: str,
    ) -> tuple[bool, str]:
        pass
