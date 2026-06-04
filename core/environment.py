from __future__ import annotations

from pathlib import Path
from typing import Any

from core.schemas import CaseInfo, DialogueHistory, PolicyOutput, StepResult
from core.logger import RolloutLogger
from plugins.user_llm.base import UserLLMPlugin
from plugins.medical_llm.base import MedicalLLMPlugin
from plugins.extractor_llm.base import ExtractorLLMPlugin
from plugins.policy.base import PolicyPlugin


class MedicalHACEnvironment:
    def __init__(
        self,
        user_llm: UserLLMPlugin,
        medical_llm: MedicalLLMPlugin,
        extractor_llm: ExtractorLLMPlugin,
        policy: PolicyPlugin,
        config: dict[str, Any],
    ):
        self.user_llm = user_llm
        self.medical_llm = medical_llm
        self.extractor_llm = extractor_llm
        self.policy = policy
        self.config = config

        self._case_info: CaseInfo | None = None
        self._history: DialogueHistory | None = None
        self._current_action: PolicyOutput | None = None
        self._turn_id: int = 0

    def reset(self, case_info: CaseInfo) -> DialogueHistory:
        self._case_info = case_info
        self._history = DialogueHistory(case_id=case_info.case_id)
        self._turn_id = 0

        # Initial action: DEFER until first user state is available
        from core.schemas import PolicyOutput
        action_space = self.policy.action_space
        self._current_action = PolicyOutput(
            action_id="DEFER",
            action_prompt=action_space["DEFER"]["prompt"],
            confidence=1.0,
        )
        return self._history

    def step(self) -> StepResult:
        assert self._case_info is not None, "Call reset() before step()"

        action_prompt = self._current_action.action_prompt

        # 1. Medical LLM generates system utterance
        medical_response = self.medical_llm.generate_medical_response(
            case_info=self._case_info,
            dialogue_history=self._history,
            action_prompt=action_prompt,
        )
        self._history.add_turn("medical", medical_response, action=self._current_action.action_id)

        # 2. User LLM generates user utterance
        user_utterance = self.user_llm.generate_user_utterance(
            case_info=self._case_info,
            dialogue_history=self._history,
            system_utterance=medical_response,
        )
        self._history.add_turn("user", user_utterance)

        # 3. Extractor LLM extracts user state
        user_state = self.extractor_llm.extract_user_state(
            case_info=self._case_info,
            dialogue_history=self._history,
        )

        # 4. Policy selects next action
        policy_output = self.policy.select_action(
            case_info=self._case_info,
            dialogue_history=self._history,
            user_state=user_state,
        )
        self._current_action = policy_output

        result = StepResult(
            case_id=self._case_info.case_id,
            turn_id=self._turn_id,
            medical_response=medical_response,
            user_utterance=user_utterance,
            user_state=user_state,
            selected_action=policy_output.action_id,
            action_prompt=policy_output.action_prompt,
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
            max_turns = self.config.get("experiment", {}).get("max_turns", 2)

        model_names = {
            "user_llm": self.user_llm.name(),
            "medical_llm": self.medical_llm.name(),
            "extractor_llm": self.extractor_llm.name(),
            "policy": self.policy.name(),
        }
        logger = RolloutLogger(case_info=case_info, model_names=model_names)

        self.reset(case_info)
        results: list[StepResult] = []

        for _ in range(max_turns):
            result = self.step()
            results.append(result)
            logger.log_step(
                result,
                dialogue_snapshot=[t.model_dump() for t in self._history.turns],
            )

        if output_path is not None:
            logger.save(output_path)

        return results
