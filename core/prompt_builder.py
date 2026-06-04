from __future__ import annotations

from core.schemas import CaseInfo, DialogueHistory


def build_medical_prompt(
    case_info: CaseInfo,
    dialogue_history: DialogueHistory,
    action_prompt: str,
) -> list[dict[str, str]]:
    system = (
        "You are a medical AI assistant collaborating with a clinician. "
        f"Case: {case_info.scenario}\n"
        f"Question: {case_info.question}\n"
        f"Options: {case_info.options}\n\n"
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
        "You are a clinician consulting a medical AI assistant. "
        f"Case: {case_info.scenario}\n"
        f"Question: {case_info.question}\n"
        f"Options: {case_info.options}"
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
        "Extract a structured JSON summary of the user's (clinician's) current state "
        "from the dialogue. Output valid JSON only, no explanation."
    )
    content = (
        f"Case: {case_info.scenario}\n"
        f"Dialogue:\n{dialogue_history.to_prompt()}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
