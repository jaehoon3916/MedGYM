from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, PolicyOutput


class NaivePolicy(PolicyPlugin):
    """No-policy baseline: generic action_prompt with no fact-validator input."""

    needs_verification = False

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)

    def name(self) -> str:
        return "naive-policy"

    def load(self) -> None:
        pass

    def select_action(  # type: ignore[override]
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
    ) -> PolicyOutput:
        return PolicyOutput(
            stage="INFORM",
            locution="assert",
            locution_type="fact",
            action_id="INFORM.assert",
            action_prompt="Respond to the clinician's latest statement based on your medical expertise and the case context.",
            confidence=1.0,
            metadata={"policy": "naive"},
        )
