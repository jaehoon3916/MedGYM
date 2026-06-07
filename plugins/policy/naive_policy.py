from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput


class NaivePolicy(PolicyPlugin):
    """No policy: formats verification_template directly into action_prompt for Medical LLM."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)

    def name(self) -> str:
        return "naive-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        vt = verification_template
        action_prompt = (
            f"The clinician's latest claim has been validated as follows:\n"
            f"  Relation: {vt.overall_relation}\n"
            f"  Confidence: {vt.confidence}\n"
            f"  Rationale: {vt.short_rationale}\n\n"
            f"Respond appropriately based on this assessment."
        )
        return PolicyOutput(
            stage="INFORM",
            locution="assert",
            locution_type="fact",
            action_id="INFORM.assert",
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={"policy": "naive"},
        )
