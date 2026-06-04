from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, UserState, PolicyOutput


def _has_wrong_claim(user_state: UserState, case_info: CaseInfo) -> bool:
    """Check if user's medical claims contradict the correct answer."""
    if case_info.correct_answer is None:
        return False
    for claim in user_state.medical_claims:
        if (claim.confidence in ("high", "medium")
                and case_info.correct_answer.lower() not in claim.claim.lower()):
            return True
    return False


def _has_correct_claim(user_state: UserState, case_info: CaseInfo) -> bool:
    """Check if user's medical claims align with the correct answer."""
    if case_info.correct_answer is None:
        return False
    for claim in user_state.medical_claims:
        if case_info.correct_answer.lower() in claim.claim.lower():
            return True
    if user_state.stance_toward_medical_llm == "agree" and user_state.certainty == "certain":
        return True
    return False


class RulePolicy(PolicyPlugin):
    def __init__(self, config: dict[str, Any], action_space: dict[str, dict[str, str]]):
        super().__init__(config, action_space)

    def name(self) -> str:
        return "rule-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        user_state: UserState,
    ) -> PolicyOutput:
        # Guideline 8.3 heuristic
        if user_state.certainty == "uncertain":
            action_id = "ASK_EVIDENCE"
        elif user_state.stance_toward_medical_llm == "skeptical":
            action_id = "SUMMARIZE"
        elif _has_wrong_claim(user_state, case_info):
            action_id = "CHALLENGE"
        elif _has_correct_claim(user_state, case_info):
            action_id = "ACCEPT"
        else:
            action_id = "DEFER"

        return PolicyOutput(
            action_id=action_id,
            action_prompt=self.action_space[action_id]["prompt"],
            confidence=1.0,
            metadata={"policy": "rule"},
        )
