from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from core.schemas import CaseInfo, DialogueHistory
from plugins.user_llm.base import UserLLMPlugin
from plugins.vllm_base import VLLMBasePlugin

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


@lru_cache(maxsize=1)
def _load_prompts() -> dict:
    with open(_PROMPTS_DIR / "user_simulator.yaml") as f:
        return yaml.safe_load(f)


def _safe_json_load(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def _fmt_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


def _fmt_history(history: DialogueHistory) -> str:
    if not history.turns:
        return "(No conversation yet)"
    lines = []
    for turn in history.turns:
        role = "Doctor" if turn.speaker == "medical" else "Patient"
        lines.append(f"[{role}]: {turn.text}")
    return "\n".join(lines)


def _fmt_forbidden(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) if items else "(none)"


class SimulatedUserState(BaseModel):
    clinical_claim: str = ""
    confidence: Literal["certain", "uncertain", "neutral"] = "uncertain"
    intent: str = ""
    evidence_level: Literal["strong", "moderate", "weak", "none"] = "weak"
    emotional_tone: str = "neutral"
    forbidden_information: list[str] = []


class VLLMUserLLM(VLLMBasePlugin, UserLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        self._prev_state: SimulatedUserState | None = None

    def name(self) -> str:
        return f"vllm-user-{self._model}"

    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> str:
        state = self._build_state(case_info, dialogue_history)
        utterance = self._generate_utterance(case_info, dialogue_history, state)
        utterance = self._mask_utterance(utterance, state, case_info, dialogue_history)
        self._prev_state = state
        return utterance

    # ── Stage 1: State Builder ───────────────────────────────────────────────

    def _build_state(self, case_info: CaseInfo, history: DialogueHistory) -> SimulatedUserState:
        tmpl = _load_prompts()
        prev_str = self._prev_state.model_dump_json(indent=2) if self._prev_state else "None"
        ctx = dict(
            scenario=case_info.scenario,
            options=_fmt_options(case_info.options),
            dialogue_history=_fmt_history(history),
            prev_state=prev_str,
        )
        messages = [
            {"role": "system", "content": tmpl["state_builder_system"].format(**ctx)},
            {"role": "user",   "content": tmpl["state_builder_user"]},
        ]
        raw = self._chat(messages, temperature=0.0)
        parsed = _safe_json_load(raw)

        # Guarantee gold answer and rationale are always forbidden
        forbidden: list[str] = parsed.get("forbidden_information", [])
        for item in filter(None, [case_info.answer, case_info.explanation]):
            if item not in forbidden:
                forbidden.append(item)

        return SimulatedUserState(
            clinical_claim=parsed.get("clinical_claim", ""),
            confidence=parsed.get("confidence", "uncertain"),
            intent=parsed.get("intent", ""),
            evidence_level=parsed.get("evidence_level", "weak"),
            emotional_tone=parsed.get("emotional_tone", "neutral"),
            forbidden_information=forbidden,
        )

    # ── Stage 2: Utterance Generator ────────────────────────────────────────

    def _generate_utterance(
        self,
        case_info: CaseInfo,
        history: DialogueHistory,
        state: SimulatedUserState,
    ) -> str:
        tmpl = _load_prompts()
        system_utterance = next(
            (t.text for t in reversed(history.turns) if t.speaker == "medical"),
            "(This is the start of the conversation.)",
        )
        ctx = dict(
            scenario=case_info.scenario,
            clinical_claim=state.clinical_claim,
            confidence=state.confidence,
            intent=state.intent,
            evidence_level=state.evidence_level,
            emotional_tone=state.emotional_tone,
            dialogue_history=_fmt_history(history),
            forbidden_information=_fmt_forbidden(state.forbidden_information),
        )
        messages = [
            {"role": "system", "content": tmpl["utterance_generator_system"].format(**ctx)},
            {"role": "user",   "content": tmpl["utterance_generator_user"].format(
                system_utterance=system_utterance,
            )},
        ]
        return self._chat(messages).strip()

    # ── Stage 3: Utterance Mask ──────────────────────────────────────────────

    def _mask_utterance(
        self,
        draft: str,
        state: SimulatedUserState,
        case_info: CaseInfo,
        history: DialogueHistory,
    ) -> str:
        tmpl = _load_prompts()
        ctx = dict(
            scenario=case_info.scenario,
            clinical_claim=state.clinical_claim,
            confidence=state.confidence,
            intent=state.intent,
            evidence_level=state.evidence_level,
            emotional_tone=state.emotional_tone,
            dialogue_history=_fmt_history(history),
            forbidden_information=_fmt_forbidden(state.forbidden_information),
        )
        messages = [
            {"role": "system", "content": tmpl["utterance_mask_system"].format(**ctx)},
            {"role": "user",   "content": tmpl["utterance_mask_user"].format(
                draft_utterance=draft,
            )},
        ]
        raw = self._chat(messages, temperature=0.0)
        parsed = _safe_json_load(raw)
        return (parsed.get("refined_utterance") or draft).strip() or draft
