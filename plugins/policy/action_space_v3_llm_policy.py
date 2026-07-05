"""LLM policy specialized for action_space_v3.yaml.

Unlike action_space_llm_policy.py, this is intentionally not generic and does not mention the
old deliberation stage/locution layer. It chooses exactly one v3 disclosure mode:

  EXTEND    -- help the clinician reason without stating the AI's conclusion.
  RECOMMEND -- disclose the medical AI's own conclusion directly.
"""
from __future__ import annotations

import json
import re
from typing import Any

from core.prompt_builder import _load
from core.schemas import CaseInfo, DialogueHistory, PolicyOutput, VerificationTemplate
from plugins.policy.base import PolicyPlugin
from plugins.vllm_base import VLLMBasePlugin

_V3_CONTROLS = ("EXTEND", "RECOMMEND")


def _controls_so_far(dialogue_history: DialogueHistory) -> set[str]:
    return {
        str(t.action).split(".")[0].upper()
        for t in dialogue_history.turns
        if t.speaker == "medical" and t.action
    }


class ActionSpaceV3LLMPolicy(VLLMBasePlugin, PolicyPlugin):
    """Prompted LLM policy over the v3 EXTEND/RECOMMEND disclosure modes."""

    needs_verification = True

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._use_fact_validator = bool(config.get("use_fact_validator", True))
        self.needs_verification = self._use_fact_validator

        self._stages: dict[str, dict[str, Any]] = {
            cid: action_space["stages"][cid]
            for cid in _V3_CONTROLS
            if cid in action_space.get("stages", {})
        }
        missing = [cid for cid in _V3_CONTROLS if cid not in self._stages]
        if missing:
            raise ValueError(
                "ActionSpaceV3LLMPolicy requires action_space_v3-style controls; "
                f"missing: {', '.join(missing)}"
            )
        self._transitions: dict[str, list[str]] = action_space.get("transitions", {})

    def name(self) -> str:
        suffix = "" if self._use_fact_validator else "-no-validator"
        return f"action-space-v3-llm-policy-{self._model}{suffix}"

    def load(self) -> None:
        pass

    def _stage_menu_text(self) -> str:
        lines = []
        for cid in _V3_CONTROLS:
            desc = " ".join(str(self._stages[cid].get("description", "")).split())
            lines.append(f"- {cid}: {desc}")
        return "\n".join(lines)

    def _reachable(self, dialogue_history: DialogueHistory) -> list[str]:
        occurred = _controls_so_far(dialogue_history)
        reachable = [
            cid for cid in _V3_CONTROLS
            if all(req in occurred for req in self._transitions.get(cid, []))
        ]
        return reachable or ["EXTEND"]

    def _build_messages(
        self,
        vt: VerificationTemplate | None,
        current_user_utterance: str,
        dialogue_history: DialogueHistory,
        reachable: list[str],
    ) -> list[dict[str, str]]:
        tmpl = _load("action_space_v3_llm_policy")
        fv_block = ""
        if self._use_fact_validator and vt is not None:
            fv_block = tmpl["fact_validation_block"].format(
                overall_relation=vt.overall_relation,
                confidence=vt.confidence,
                reasoning=vt.reasoning,
            )
        user = tmpl["user"].format(
            reachable_stages=", ".join(reachable),
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
        )
        return [
            {"role": "system", "content": tmpl["system"].format(stage_menu=self._stage_menu_text())},
            {"role": "user", "content": user},
        ]

    def _parse(self, raw: str, reachable: list[str]) -> tuple[str, str]:
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        control, guidance = reachable[0], ""
        if "{" in text and "}" in text:
            blob = text[text.find("{"): text.rfind("}") + 1]
            try:
                data = json.loads(blob)
                maybe_control = str(data.get("stage", "")).strip().upper()
                if maybe_control in _V3_CONTROLS:
                    control = maybe_control
                guidance = str(data.get("action_guidance", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                pass
        if control not in reachable:
            control = reachable[0]
        return control, guidance

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        reachable = self._reachable(dialogue_history)
        messages = self._build_messages(
            verification_template,
            current_user_utterance,
            dialogue_history,
            reachable,
        )
        raw = self._chat(messages, response_format={"type": "json_object"})
        control, guidance = self._parse(raw, reachable)

        if not guidance:
            guidance = (
                "Do not state your own conclusion. Help the clinician reason by surfacing one "
                "missing consideration or asking one targeted justification question."
                if control == "EXTEND"
                else
                "State your own conclusion directly and explicitly, then give concise reasoning."
            )

        return PolicyOutput(
            stage=control,
            locution="",
            locution_type="",
            action_id=control,
            action_prompt=guidance,
            confidence=1.0,
            metadata={"policy": self.name(), "control": control, "raw": raw},
        )
