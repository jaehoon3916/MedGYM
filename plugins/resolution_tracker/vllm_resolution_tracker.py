from __future__ import annotations

from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.resolution_tracker.base import ResolutionTrackerPlugin
from core.schemas import AgendaItem
from core.prompt_builder import _load
from core.json_utils import safe_json_load


class VLLMResolutionTracker(VLLMBasePlugin, ResolutionTrackerPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)

    def name(self) -> str:
        return f"vllm-resolution-tracker-{self._model}"

    def check_resolved(
        self,
        agenda_item: AgendaItem,
        recent_medical: str,
        recent_user: str,
    ) -> tuple[bool, str]:
        tmpl = _load("resolution_tracker")
        user_content = tmpl["user"].format(
            issue=agenda_item.issue,
            human_position=agenda_item.human_position or "(not specified)",
            ai_position=agenda_item.ai_position or "(not specified)",
            recent_medical=recent_medical or "(none)",
            recent_user=recent_user,
        )
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": user_content},
        ]
        raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
        data = safe_json_load(raw)
        resolved = bool(data.get("resolved", False))
        reason = str(data.get("reason", "")).strip()
        return resolved, reason
