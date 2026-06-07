from typing import Any

from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate


class MockFactValidatorLLM(FactValidatorLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def name(self) -> str:
        return "mock-fact-validator"

    def load(self) -> None:
        pass

    def validate_facts(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
    ) -> VerificationTemplate:
        return VerificationTemplate(
            overall_relation="insufficient",
            confidence="low",
            evidence_gaps=["clinical evidence not yet provided"],
            short_rationale="[mock] Insufficient information to validate the clinician's claim.",
            optional_claim_checks=[],
        )
