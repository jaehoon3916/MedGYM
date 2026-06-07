from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache

import yaml

from core.schemas import CaseInfo, DialogueHistory

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(_PROMPTS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


def build_medical_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    action_prompt: str,
) -> list[dict[str, str]]:
    tmpl = _load("medical_llm")
    system = tmpl["system"].format(
        scenario=case_info.scenario,
        options=_format_options(case_info.options),
        action_prompt=action_prompt,
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(dialogue_history.to_messages())
    return messages


def build_user_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    turn_id: int = 0,
) -> list[dict[str, str]]:
    tmpl = _load("user_llm")
    system_key = "system_turn_1" if turn_id == 0 else "system_turn_2"
    system = tmpl[system_key].format(
        scenario=case_info.scenario,
        options=_format_options(case_info.options),
        target_belief=case_info.answer,
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(dialogue_history.to_messages())
    return messages


def build_extractor_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    json_schema: dict | None = None,
) -> list[dict[str, str]]:
    tmpl = _load("extractor_llm")
    schema_hint = ""
    if json_schema:
        schema_hint = f"\n\nOutput JSON schema:\n{json.dumps(json_schema, indent=2)}"
    system = tmpl["system"].rstrip() + schema_hint
    user_content = tmpl["user"].format(
        scenario=case_info.scenario,
        options=_format_options(case_info.options),
        dialogue=dialogue_history.to_prompt(),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_fact_validator_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    current_user_utterance: str,
) -> list[dict[str, str]]:
    tmpl = _load("fact_validator_llm")
    user_content = tmpl["user"].format(
        scenario=case_info.scenario,
        options=_format_options(case_info.options),
        correct_answer=case_info.correct_answer,
        dialogue=dialogue_history.to_prompt(),
        current_user_utterance=current_user_utterance,
    )
    return [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": user_content},
    ]
