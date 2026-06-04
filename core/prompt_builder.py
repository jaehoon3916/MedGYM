from __future__ import annotations

from core.schemas import CaseInfo, DialogueHistory


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


def build_medical_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    action_prompt: str,
) -> list[dict[str, str]]:
    system = (
        "You are a medical AI assistant collaborating with a clinician on a JAMA Clinical Challenge case.\n\n"
        f"Case:\n{case_info.scenario}\n\n"
        f"Options:\n{_format_options(case_info.options)}\n\n"
        f"Instruction: {action_prompt}"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(dialogue_history.to_messages())
    return messages


def build_user_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    system_utterance: str,
) -> list[dict[str, str]]:
    system = (
        "You are a clinician working through a JAMA Clinical Challenge case with a medical AI assistant.\n\n"
        f"Case:\n{case_info.scenario}\n\n"
        f"Options:\n{_format_options(case_info.options)}"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(dialogue_history.to_messages())
    messages.append({"role": "assistant", "content": system_utterance})
    return messages


def build_extractor_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
) -> list[dict[str, str]]:
    system = (
        "Extract a structured JSON summary of the clinician's current reasoning state "
        "from the dialogue. Output valid JSON only, no explanation."
    )
    content = (
        f"Case:\n{case_info.scenario}\n\n"
        f"Options:\n{_format_options(case_info.options)}\n\n"
        f"Dialogue:\n{dialogue_history.to_prompt()}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
