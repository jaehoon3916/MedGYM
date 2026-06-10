from __future__ import annotations

from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate
from core.prompt_builder import build_fact_validator_prompt
from core.json_utils import safe_json_load

_RELATIONS = {"supported", "contradicted", "insufficient", "mixed", "unknown"}
_CONFIDENCES = {"high", "medium", "low"}


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else []


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
        # safe_json_load tolerates ```json fences / surrounding prose (the cause of spurious "unknown").
        data = safe_json_load(raw)
        # Coerce defensively so an out-of-vocab literal never raises or silently falls back to unknown.
        rel = str(data.get("overall_relation", "unknown")).strip().lower()
        conf = str(data.get("confidence", "low")).strip().lower()
        return VerificationTemplate(
            overall_relation=rel if rel in _RELATIONS else "unknown",  # type: ignore[arg-type]
            confidence=conf if conf in _CONFIDENCES else "low",        # type: ignore[arg-type]
            evidence_gaps=_as_list(data.get("evidence_gaps")),
            reasoning=str(data.get("reasoning", "")),
            optional_claim_checks=_as_list(data.get("optional_claim_checks")),
        )
