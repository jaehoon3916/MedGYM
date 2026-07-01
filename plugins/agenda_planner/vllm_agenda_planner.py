from __future__ import annotations

from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.agenda_planner.base import DisagreementAnalyzerPlugin
from core.schemas import CaseInfo, Agenda, AgendaItem
from core.prompt_builder import _load
from core.json_utils import safe_json_load


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


class VLLMDisagreementAnalyzer(VLLMBasePlugin, DisagreementAnalyzerPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)

    def name(self) -> str:
        return f"vllm-disagreement-analyzer-{self._model}"

    def analyze(
        self,
        case_info: CaseInfo,
        human_opinion: str,
        ai_opinion: str,
    ) -> Agenda:
        tmpl = _load("disagreement_analyzer")
        user_content = tmpl["user"].format(
            scenario=case_info.scenario,
            options_text=_format_options(case_info.options),
            human_opinion=human_opinion,
            ai_opinion=ai_opinion,
        )
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": user_content},
        ]
        raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
        data = safe_json_load(raw)
        items: list[AgendaItem] = []
        for entry in data.get("agenda", []):
            try:
                items.append(AgendaItem(
                    id=int(entry.get("id", len(items) + 1)),
                    issue=str(entry.get("issue", "")).strip(),
                    human_position=str(entry.get("human_position", "")).strip(),
                    ai_position=str(entry.get("ai_position", "")).strip(),
                    status="unresolved",
                ))
            except (ValueError, TypeError):
                continue
        if not items:
            items = [AgendaItem(id=1, issue="Overall diagnostic or management approach", status="unresolved")]
        return Agenda(items=items)
