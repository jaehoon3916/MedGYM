from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.schemas import CaseInfo, StepResult


class RolloutLogger:
    def __init__(self, case_info: CaseInfo, model_names: dict[str, str]):
        self.case_info = case_info
        self.model_names = model_names
        self._steps: list[dict[str, Any]] = []

    def log_step(self, step_result: StepResult, dialogue_snapshot: list[dict]) -> None:
        record = {
            "case_id": step_result.case_id,
            "turn_id": step_result.turn_id,
            "case_info": self.case_info.model_dump(),
            "dialogue_history": dialogue_snapshot,
            "user_state": step_result.user_state.model_dump(),
            "selected_action": step_result.selected_action,
            "action_prompt": step_result.action_prompt,
            "medical_response": step_result.medical_response,
            "user_utterance": step_result.user_utterance,
            "reward": step_result.reward,
            "judge_output": step_result.metadata.get("judge_output"),
            "model_name": self.model_names,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._steps.append(record)

    def save(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in self._steps:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._steps)
