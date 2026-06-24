from __future__ import annotations

import json
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load

_VALID_ACTIONS = {"ARGUE", "ACCEPT", "DEFER"}
_DEFAULT_ACTION = "DEFER"

_ACTION_INSTRUCTIONS = {
    "ARGUE": "Challenge or correct the doctor's claim with specific clinical reasoning, or propose an alternative.",
    "ACCEPT": "Affirm the doctor's claim with supporting clinical reasoning.",
    "DEFER": "Acknowledge neutrally and, if genuinely unclear, ask for clarification rather than committing to a position.",
}


def _normalize_action(x: str) -> str:
    x = (x or "").strip().upper()
    return x if x in _VALID_ACTIONS else _DEFAULT_ACTION


class ReactPolicy(VLLMBasePlugin, PolicyPlugin):
    """ReAct-style baseline: one reasoning call produces a Thought + an ARGUE/ACCEPT/DEFER
    Action (blind to verification_template, like medcobe_naive/medcobe_feedback), which is
    then handed to the medical LLM as a normal action_prompt via the existing pipeline."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)

    def name(self) -> str:
        return f"react-policy-{self._model}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        tmpl = _load("react_policy")
        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": tmpl["user"].format(
                scenario=case_info.scenario,
                dialogue=dialogue_history.to_prompt(),
                current_user_utterance=current_user_utterance,
            )},
        ]
        raw = self._chat(messages, temperature=0.0)
        try:
            data = json.loads(raw)
            action = _normalize_action(data.get("action", ""))
            reason = (data.get("reason") or "").strip()
            thought = (data.get("thought") or "").strip()
        except (json.JSONDecodeError, AttributeError):
            action, reason, thought = _DEFAULT_ACTION, "", ""

        action_prompt = (
            f"Chosen strategy: {action} — {reason}\n"
            f"{_ACTION_INSTRUCTIONS[action]}"
        )
        return PolicyOutput(
            stage="INFORM",
            locution="assert",
            locution_type="fact",
            action_id="INFORM.assert",
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={"policy": "react", "react_action": action, "thought": thought, "raw": raw},
        )
