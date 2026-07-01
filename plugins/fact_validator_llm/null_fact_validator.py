from __future__ import annotations

from typing import Any

from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate


class NullFactValidatorLLM(FactValidatorLLMPlugin):
    """No-op fact validator for policies that don't use verification output."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        pass

    def name(self) -> str:
        return "null-fact-validator"

    def load(self) -> None:
        pass

    def validate_facts(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
    ) -> VerificationTemplate:
        return VerificationTemplate()
