from __future__ import annotations

from pathlib import Path
from typing import Any

from core.schemas import CaseInfo, DialogueHistory, StepResult
from core.logger import RolloutLogger
from plugins.user_llm.base import UserLLMPlugin
from plugins.medical_llm.base import MedicalLLMPlugin
from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from plugins.policy.base import PolicyPlugin


class MedicalHACEnvironment:
    def __init__(
        self,
        user_llm: UserLLMPlugin,
        medical_llm: MedicalLLMPlugin,
        fact_validator_llm: FactValidatorLLMPlugin,
        policy: PolicyPlugin,
        config: dict[str, Any],
    ):
        self.user_llm = user_llm
        self.medical_llm = medical_llm
        self.fact_validator_llm = fact_validator_llm
        self.policy = policy
        self.config = config

        self._case_info: CaseInfo | None = None
        self._history: DialogueHistory | None = None
        self._turn_id: int = 0

    def reset(self, case_info: CaseInfo) -> DialogueHistory:
        self._case_info = case_info
        self._history = DialogueHistory(case_id=case_info.case_id)
        self._turn_id = 0
        return self._history

    def step(self) -> StepResult:
        assert self._case_info is not None and self._history is not None, "Call reset() before step()"

        # 1. User LLM generates user utterance (history empty on turn 0)
        user_utterance = self.user_llm.generate_user_utterance(
            case_info=self._case_info,
            dialogue_history=self._history,
            turn_id=self._turn_id,
        )

        # 2. Update dialogue history with user utterance
        self._history.add_turn("user", user_utterance)

        # 3. Fact Validator LLM validates facts from current user utterance
        verification_template = self.fact_validator_llm.validate_facts(
            case_info=self._case_info,
            dialogue_history=self._history,
            current_user_utterance=user_utterance,
        )

        # 4. Policy selects next action based on user utterance + verification template
        policy_output = self.policy.select_action(
            case_info=self._case_info,
            dialogue_history=self._history,
            current_user_utterance=user_utterance,
            verification_template=verification_template,
        )

        # 5. Medical LLM generates system utterance using selected action prompt
        medical_response = self.medical_llm.generate_medical_response(
            case_info=self._case_info,
            dialogue_history=self._history,
            action_prompt=policy_output.action_prompt,
        )
        self._history.add_turn("medical", medical_response, action=policy_output.action_id)

        result = StepResult(
            case_id=self._case_info.case_id,
            turn_id=self._turn_id,
            user_utterance=user_utterance,
            verification_template=verification_template,
            selected_action=policy_output.action_id,
            action_prompt=policy_output.action_prompt,
            medical_response=medical_response,
            reward=None,
        )
        self._turn_id += 1
        return result

    def run_episode(
        self,
        case_info: CaseInfo,
        max_turns: int | None = None,
        output_path: str | Path | None = None,
    ) -> list[StepResult]:
        if max_turns is None:
            max_turns = int(self.config.get("experiment", {}).get("max_turns", 2))

        model_names = {
            "user_llm": self.user_llm.name(),
            "medical_llm": self.medical_llm.name(),
            "fact_validator_llm": self.fact_validator_llm.name(),
            "policy": self.policy.name(),
        }
        logger = RolloutLogger(case_info=case_info, model_names=model_names)

        self.reset(case_info)
        results: list[StepResult] = []

        for _ in range(max_turns):
            result = self.step()
            results.append(result)
            assert self._history is not None
            logger.log_step(
                result,
                dialogue_snapshot=[t.model_dump() for t in self._history.turns],
            )

        if output_path is not None:
            logger.save(output_path)

        return results
