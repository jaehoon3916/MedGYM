from __future__ import annotations

from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load

_VALID_STAGES = {"INFORM", "PROPOSE", "CONSIDER", "REVISE", "RECOMMEND", "CONFIRM", "CLOSE"}


def _parse_dot_action(raw: str) -> tuple[str, str, str]:
    """Parse 'STAGE.locution' format from model output."""
    parts = raw.strip().upper().split(".")
    if len(parts) >= 2:
        stage = parts[0]
        locution = parts[1].lower()
    else:
        stage, locution = "INFORM", "ask_justify"
    if stage not in _VALID_STAGES:
        stage = "INFORM"
    return stage, locution, "fact"


class QwenPolicy(VLLMBasePlugin, PolicyPlugin):
    """Qwen3-8B policy: baseline / sft / full depending on checkpoint loaded."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._mode: str = config.get("mode", "baseline")

    def name(self) -> str:
        return f"qwen-policy-{self._mode}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        tmpl = _load("baseline_policy")
        vt = verification_template
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(
                overall_relation=vt.overall_relation,
                confidence=vt.confidence,
                short_rationale=vt.short_rationale,
                current_user_utterance=current_user_utterance,
            )},
        ]
        raw = self._chat(messages, temperature=0.0, max_tokens=16)
        stage, locution, locution_type = _parse_dot_action(raw)
        action_id = f"{stage}.{locution}"
        action_prompt = (
            f"Stage: {stage} | Locution: {locution}({locution_type})\n"
            f"Apply this deliberation move in your response."
        )
        return PolicyOutput(
            stage=stage,
            locution=locution,
            locution_type=locution_type,
            action_id=action_id,
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={"policy": self._mode},
        )
