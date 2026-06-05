from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.extractor_llm.base import ExtractorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState
from core.prompt_builder import build_extractor_prompt


def _build_json_schema(schema_fields: list[dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in schema_fields:
        name = field["name"]
        ftype = field["type"]
        if ftype == "categorical":
            properties[name] = {"type": "string", "enum": field["values"]}
        elif ftype == "list":
            properties[name] = {"type": "array", "items": {"type": "string"}}
        elif ftype == "text":
            properties[name] = {"type": "string"}
        required.append(name)
    return {"type": "object", "properties": properties, "required": required}


class VLLMExtractorLLM(VLLMBasePlugin, ExtractorLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        schema_fields: list[dict[str, Any]] = config.get("_user_state_schema", [])
        self._json_schema = _build_json_schema(schema_fields)

    def name(self) -> str:
        return f"vllm-extractor-{self._model}"

    def extract_user_state(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
    ) -> UserState:
        messages = build_extractor_prompt(case_info, dialogue_history, self._json_schema)
        raw = self._chat(
            messages,
            temperature=0.0,
            extra_body={"guided_json": self._json_schema},
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return UserState(**data)
