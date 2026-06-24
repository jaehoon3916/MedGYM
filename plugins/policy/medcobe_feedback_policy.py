import json
from pathlib import Path
from typing import Any

from plugins.policy.base import PolicyPlugin
from plugins.policy.medcobe_feedback_text import generate_feedback
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput

DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parents[2] / "outputs" / "medcobe_feedback_calibration"


def _safe_model_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


class MedCobeFeedbackPolicy(PolicyPlugin):
    """MedCOBE-as-feedback baseline: injects a static, per-(medical) model behavior guideline
    (passive/aggressive/balanced, picked from a pre-computed sycophancy/obstruction rate) before
    the medical LLM's response -- a prompt-only behavior guide, not a learned policy. The
    calibration file is produced offline by scripts/calibrate_medcobe_feedback.py; if it is
    missing for this target_model, falls back to the balanced (untuned) guideline so the policy
    never hard-fails before calibration has been run."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)
        self._target_model: str | None = config.get("target_model")
        calibration_dir = Path(config.get("calibration_dir", DEFAULT_CALIBRATION_DIR))
        self._sycophancy = 0.0
        self._obstruction = 0.0
        if self._target_model:
            path = calibration_dir / f"{_safe_model_dir_name(self._target_model)}.json"
            if path.is_file():
                payload = json.loads(path.read_text())
                self._sycophancy = float(payload.get("sycophancy", 0.0))
                self._obstruction = float(payload.get("obstruction", 0.0))
            else:
                print(
                    f"[medcobe_feedback] no calibration file for target_model="
                    f"{self._target_model!r} at {path} -- falling back to balanced guideline. "
                    f"Run scripts/calibrate_medcobe_feedback.py to calibrate."
                )
        self._guideline = generate_feedback(self._sycophancy, self._obstruction)

    def name(self) -> str:
        return "medcobe-feedback-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        action_prompt = (
            f"[Behavior Guideline]\n{self._guideline}\n\n"
            f"Follow the Behavior Guideline above when deciding how to phrase your response, "
            f"when to argue vs. accept, and how to ground your reasoning."
        )
        return PolicyOutput(
            stage="INFORM",
            locution="assert",
            locution_type="fact",
            action_id="INFORM.assert",
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={
                "policy": "medcobe_feedback",
                "sycophancy": self._sycophancy,
                "obstruction": self._obstruction,
            },
        )
