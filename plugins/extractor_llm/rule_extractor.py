from typing import Any

from plugins.extractor_llm.base import ExtractorLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState

_UNCERTAIN_KEYWORDS = ["not sure", "uncertain", "don't know", "unclear", "unsure", "confused"]
_AGREE_KEYWORDS = ["agree", "that makes sense", "correct", "right", "yes"]
_DISAGREE_KEYWORDS = ["disagree", "wrong", "incorrect", "no,", "that's not"]
_SKEPTICAL_KEYWORDS = ["challenge", "doubt", "question", "not convinced", "evidence"]
_CHALLENGE_KEYWORDS = ["but", "however", "actually", "instead", "rather"]
_EVIDENCE_KEYWORDS = ["test", "results", "findings", "data", "evidence", "symptoms"]


class RuleExtractorLLM(ExtractorLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

    def name(self) -> str:
        return "rule-extractor"

    def load(self) -> None:
        pass

    def extract_user_state(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
    ) -> UserState:
        if not dialogue_history.turns:
            return UserState(summary="No dialogue yet.")

        last_user_turns = [t for t in dialogue_history.turns if t.speaker == "user"]
        if not last_user_turns:
            return UserState(summary="No user utterance yet.")

        text = last_user_turns[-1].text.lower()

        certainty = "uncertain" if any(kw in text for kw in _UNCERTAIN_KEYWORDS) else "neutral"

        if any(kw in text for kw in _AGREE_KEYWORDS):
            stance = "agree"
            intent = "accept"
        elif any(kw in text for kw in _SKEPTICAL_KEYWORDS):
            stance = "skeptical"
            intent = "challenge"
        elif any(kw in text for kw in _DISAGREE_KEYWORDS):
            stance = "disagree"
            intent = "challenge"
        else:
            stance = "unknown"
            intent = "express_uncertainty" if certainty == "uncertain" else "other"

        missing = []
        if any(kw in text for kw in _EVIDENCE_KEYWORDS):
            missing.append("additional clinical evidence")
        if certainty == "uncertain":
            missing.append("clarification on the diagnosis")

        key_facts = []
        if any(kw in text for kw in _CHALLENGE_KEYWORDS):
            key_facts.append("user raised a counterpoint")

        return UserState(
            intent=intent,
            certainty=certainty,
            stance_toward_medical_llm=stance,
            key_facts=key_facts,
            missing_information=missing,
            summary=f"User stance: {stance}. Certainty: {certainty}.",
        )
