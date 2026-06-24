from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load


class MedCobeNaivePolicy(PolicyPlugin):
    """MedCOBE_EMNLP's plain AI-doctor baseline: ignores verification_template entirely so the
    medical LLM decides to argue/accept/defer from its own clinical judgment alone, with no
    fact-validator hint -- a separate policy type from "naive" (NaivePolicy) to keep existing
    naive-policy rollouts comparable."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)
        self._action_prompt: str = _load("medcobe_naive_policy")["action_prompt"]

    def name(self) -> str:
        return "medcobe-naive-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        return PolicyOutput(
            stage="INFORM",
            locution="assert",
            locution_type="fact",
            action_id="INFORM.assert",
            action_prompt=self._action_prompt,
            confidence=1.0,
            metadata={"policy": "medcobe_naive"},
        )
