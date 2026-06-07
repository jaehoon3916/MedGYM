from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate
from core.prompt_builder import build_fact_validator_prompt

_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_relation": {
            "type": "string",
            "enum": ["supported", "contradicted", "insufficient", "mixed", "unknown"],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence_gaps": {"type": "array", "items": {"type": "string"}},
        "short_rationale": {"type": "string"},
        "optional_claim_checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["overall_relation", "confidence", "evidence_gaps", "short_rationale", "optional_claim_checks"],
}


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
            extra_body={"guided_json": _VERIFICATION_SCHEMA},
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return VerificationTemplate(**data)
