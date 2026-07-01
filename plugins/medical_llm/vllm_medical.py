from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.medical_llm.base import MedicalLLMPlugin
from core.schemas import CaseInfo, DialogueHistory
from core.prompt_builder import build_medical_prompt, frame_directive
from core.json_utils import safe_json_load


def _clean_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _confidence(value: Any) -> float | None:
    """Parse the AI's self-reported confidence, clamped to [0,1] (LLMs occasionally drift just
    outside the range rather than refusing). None if unparseable/missing -- the estimate-mode
    policy treats a missing AI confidence as UNKNOWN (e.g. on turn 0, no AI turn exists yet)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, v))


class VLLMMedicalLLM(VLLMBasePlugin, MedicalLLMPlugin):
    """The AI assistant never sees the multiple-choice options/question -- only the free-text
    scenario. This was always the case and is not configurable here; the option-visibility
    A/B test lives on the DOCTOR side instead (plugins/user_llm/user_simulator/v1.py's
    show_options), since the doctor is the one whose belief/accuracy we're tracking against
    the options. The AI still reports a belief each turn -- just free text (its own best
    answer in plain language), not a lettered option, since it was never given a lettered
    list. Response is a single JSON object (response_format=json_object), not inline tags --
    final_judge separately reads its free-text replies/beliefs and maps them onto the real
    options independently (see scripts/run_scaling_poc.py /
    core/prompt_builder.py:build_final_judge_prompt)."""

    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        self._frame_style = config.get("frame_style", "command")
        # When False, the AI does NOT receive case_info.scenario directly (core/
        # prompt_builder.py:build_medical_prompt substitutes a notice instead) -- it must
        # rely entirely on what the clinician relays in dialogue. This is what gives
        # plugins.user_llm's information_sparsity persona genuine teeth (real information
        # asymmetry) instead of being purely cosmetic. Default True preserves the historical
        # behavior (AI always has full case access).
        self._show_case_info: bool = bool(config.get("show_case_info", True))

    def name(self) -> str:
        return f"vllm-medical-{self._model}"

    def generate_medical_response(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        action_prompt: str,
        current_user_utterance: str,
    ) -> tuple[str, str | None, str | None, float | None]:
        messages = build_medical_prompt(
            case_info, dialogue_history, frame_directive(action_prompt, self._frame_style),
            current_user_utterance, show_case_info=self._show_case_info,
        )
        raw = self._chat(messages, response_format={"type": "json_object"})
        data = safe_json_load(raw)
        text = _clean_str(data.get("response")) or raw.strip()
        return (text, _clean_str(data.get("belief")), _clean_str(data.get("reasoning")),
                _confidence(data.get("confidence")))
