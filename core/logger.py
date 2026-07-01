from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.schemas import CaseInfo, StepResult


class RolloutLogger:
    def __init__(self, case_info: CaseInfo, model_names: dict[str, str],
                 episode_config: dict[str, Any] | None = None):
        self.case_info = case_info
        self.model_names = model_names
        self.episode_config = episode_config or {}
        self._steps: list[dict[str, Any]] = []

    def log_step(self, step_result: StepResult, dialogue_snapshot: list[dict]) -> None:
        record = {
            "case_id": step_result.case_id,
            "turn_id": step_result.turn_id,
            "case_info": self.case_info.model_dump(),
            "episode_config": self.episode_config,
            "dialogue_history": dialogue_snapshot,
            "verification_template": step_result.verification_template.model_dump(),
            "selected_action": step_result.selected_action,
            "action_prompt": step_result.action_prompt,
            "policy_raw_output": step_result.metadata.get("policy", {}).get("raw", "") or step_result.metadata.get("policy", {}).get("raw_output", ""),
            "turn_prompts": step_result.metadata.get("turn_prompts", []),
            "medical_response": step_result.medical_response,
            "user_utterance": step_result.user_utterance,
            "reward": step_result.reward,
            "judge_output": step_result.metadata.get("judge_output"),
            "final_judgement": None,   # episode-level; set on the last record by finalize()
            "closed_by": None,
            "model_name": self.model_names,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._steps.append(record)

    def finalize(self, final_judgement: dict | None, closed_by: str) -> None:
        """Attach the episode-level final judgement and close reason to the last record."""
        if self._steps:
            self._steps[-1]["final_judgement"] = final_judgement
            self._steps[-1]["closed_by"] = closed_by

    def save(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for record in self._steps:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._steps)
