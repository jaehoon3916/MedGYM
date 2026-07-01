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


_COMMAND_FRAME = """\
You MUST respond to the clinician based on the following behavioral instruction. \
This is a strict constraint — your response MUST conform to it, not merely consider it.

{action_prompt}\
"""

_REFERENCE_FRAME = """\
The following is deliberation guidance for this turn. Use it as a reference when \
deciding how to structure and phrase your response, but you are not strictly bound to it.

{action_prompt}\
"""


def frame_directive(action_prompt: str, style: str) -> str:
    """Wrap a raw policy action_prompt in an imperative ("command") or referential
    ("reference") framing before injection into the medical LLM's [Instruction] section."""
    if style not in ("command", "reference"):
        raise ValueError(f"style must be 'command' or 'reference', got '{style}'")
    template = _COMMAND_FRAME if style == "command" else _REFERENCE_FRAME
    return template.format(action_prompt=action_prompt)


_NO_CASE_INFO_NOTICE = (
    "(You have NOT been given the case file directly. You only know what the clinician has "
    "told you so far in this conversation -- do not assume or invent findings beyond what "
    "they've actually stated.)"
)


def build_medical_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    action_prompt: str,
    current_user_utterance: str,
    show_case_info: bool = True,
) -> list[dict[str, str]]:
    """show_case_info=False creates genuine information asymmetry: the AI does not see
    case_info.scenario at all and must rely entirely on what the clinician relays in
    dialogue -- this is what gives plugins.user_llm's information_sparsity persona (see
    persona/information_sparsity.yaml) actual teeth instead of being purely cosmetic, since
    with show_case_info=True (the historical default) the AI already has full case access
    regardless of what the doctor chooses to share."""
    tmpl = _load("medical_llm")
    system = tmpl["system"].format(
        scenario=case_info.scenario if show_case_info else _NO_CASE_INFO_NOTICE,
        action_prompt=action_prompt,
    )
    messages = [{"role": "system", "content": system}]
    # Prior conversation excludes the current user turn (which env appends to history
    # before this call); it is re-emitted below as an explicitly labeled final message.
    role_map = {"medical": "assistant", "user": "user"}
    for t in dialogue_history.turns[:-1]:
        messages.append({"role": role_map[t.speaker], "content": t.text})
    messages.append({
        "role": "user",
        "content": tmpl["current_turn"].format(current_user_utterance=current_user_utterance),
    })
    return messages


def build_final_judge_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
) -> list[dict[str, str]]:
    tmpl = _load("final_judge")
    user = tmpl["user"].format(
        scenario=case_info.scenario,
        options=_format_options(case_info.options),
        dialogue=dialogue_history.to_prompt(),
    )
    return [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": user},
    ]


def build_load_judge_prompt(
    ai_utterance: str,
    doctor_utterance: str,
    scenario: str = "",
) -> list[dict[str, str]]:
    """Prompt for the cognitive-load judge: rate the mental effort the AI's turn (a reply to
    the clinician's statement) demands of the clinician, independent of medical correctness."""
    tmpl = _load("load_judge")
    user = tmpl["user"].format(
        scenario=scenario or "(not provided)",
        doctor_utterance=doctor_utterance,
        ai_utterance=ai_utterance,
    )
    return [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": user},
    ]


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
    see_case_info: bool = True,
) -> list[dict[str, str]]:
    """see_case_info=False denies the validator the case file (system_blind/user_blind
    templates), mirroring medical_llm.show_case_info=False so the validator is blind to exactly
    what the AI is blind to. It then judges the claim against general medicine + only what's been
    disclosed in dialogue -- a blind second opinion, not a ground-truth check. Default True keeps
    the original oracle-style behavior for every existing caller/config."""
    tmpl = _load("fact_validator_llm")
    if see_case_info:
        user_content = tmpl["user"].format(
            scenario=case_info.scenario,
            options=_format_options(case_info.options),
            dialogue=dialogue_history.to_prompt(),
            current_user_utterance=current_user_utterance,
        )
        system_content = tmpl["system"]
    else:
        user_content = tmpl["user_blind"].format(
            options=_format_options(case_info.options),
            dialogue=dialogue_history.to_prompt(),
            current_user_utterance=current_user_utterance,
        )
        system_content = tmpl["system_blind"]
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
