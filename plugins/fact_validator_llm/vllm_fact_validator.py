from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate
from core.prompt_builder import build_fact_validator_prompt


class VLLMFactValidatorLLM(VLLMBasePlugin, FactValidatorLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)

    def name(self) -> str:
        return f"vllm-fact-validator-{self._model}"

    def validate_facts(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
    ) -> VerificationTemplate:
        messages = build_fact_validator_prompt(case_info, dialogue_history, current_user_utterance)
        raw = self._chat(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return VerificationTemplate(**data)
