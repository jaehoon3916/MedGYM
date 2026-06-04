from typing import Any

from plugins.extractor_llm.base import ExtractorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState


class MockExtractorLLM(ExtractorLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def name(self) -> str:
        return "mock-extractor-llm"

    def load(self) -> None:
        pass

    def extract_user_state(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
    ) -> UserState:
        return UserState(
            intent="express_uncertainty",
            certainty="uncertain",
            stance_toward_medical_llm="unknown",
            summary="[mock] User expressed uncertainty about the diagnosis.",
        )
