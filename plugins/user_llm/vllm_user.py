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


class EpisodeConfig(BaseModel):
    initial_fact: Literal["correct", "incorrect"] = "incorrect"
    confidence: Literal["certain", "uncertain", "neutral"] = "uncertain"
    authority_push: Literal["high", "low"] = "low"
    information_sparcity: Literal["dense", "sparse"] = "dense"
    safety_push: Literal["true", "false"] = "false"


class SimulatedUserState(BaseModel):
    clinical_claim: str = ""
    forbidden_information: list[str] = []
    # Conversation Locutions (dynamic — updated each turn)
    dialogue_stage: Literal["Inform", "Propose", "Consider", "Revise", "Recommend", "Confirm", "Close"] = "Inform"
    locution: Literal["propose", "assert", "prefer", "ask_justify", "move", "reject", "retract", "withdraw_dialogue"] = "assert"
    locution_type: Literal["goal", "constraint", "perspective", "fact", "action", "evaluation"] = "fact"


class VLLMUserLLM(VLLMBasePlugin, UserLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        self._prev_state: SimulatedUserState | None = None
        self._episode_cfg: EpisodeConfig = EpisodeConfig()
        self._force_close: bool = False

    def reset_episode(self, episode_config: EpisodeConfig) -> None:
        self._episode_cfg = episode_config
        self._prev_state = None
        self._force_close = False

    def force_close(self) -> None:
        self._force_close = True

    def name(self) -> str:
        return f"vllm-user-{self._model}"

    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        if self._force_close:
            self._force_close = False
            return "I understand. Let's close the discussion here.", True
        state = self._build_state(case_info, dialogue_history)
        utterance = self._generate_utterance(case_info, dialogue_history, state)
        utterance = self._mask_utterance(utterance, state, case_info, dialogue_history)
        self._prev_state = state
        done = state.dialogue_stage == "Close"
        return utterance, done

    # ── Stage 1: State Builder ───────────────────────────────────────────────

    def _build_state(self, case_info: CaseInfo, history: DialogueHistory) -> SimulatedUserState:
        tmpl = _load_prompts()
        prev_str = self._prev_state.model_dump_json(indent=2) if self._prev_state else "None"
        ctx = dict(
            scenario=case_info.scenario,
            options=_fmt_options(case_info.options),
            dialogue_history=_fmt_history(history),
            prev_state=prev_str,
            initial_fact=self._episode_cfg.initial_fact,
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
            forbidden_information=forbidden,
            dialogue_stage=parsed.get("dialogue_stage", "Inform"),
            locution=parsed.get("locution", "assert"),
            locution_type=parsed.get("locution_type", "fact"),
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
            confidence=self._episode_cfg.confidence,
            dialogue_history=_fmt_history(history),
            forbidden_information=_fmt_forbidden(state.forbidden_information),
            dialogue_stage=state.dialogue_stage,
            locution=state.locution,
            locution_type=state.locution_type,
            authority_push=self._episode_cfg.authority_push,
            information_sparcity=self._episode_cfg.information_sparcity,
            safety_push=self._episode_cfg.safety_push,
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
            confidence=self._episode_cfg.confidence,
            dialogue_history=_fmt_history(history),
            forbidden_information=_fmt_forbidden(state.forbidden_information),
            dialogue_stage=state.dialogue_stage,
            locution=state.locution,
            locution_type=state.locution_type,
            authority_push=self._episode_cfg.authority_push,
            information_sparcity=self._episode_cfg.information_sparcity,
            safety_push=self._episode_cfg.safety_push,
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
