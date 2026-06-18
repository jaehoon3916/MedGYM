from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.medical_llm.base import MedicalLLMPlugin
from core.schemas import CaseInfo, DialogueHistory
from core.prompt_builder import build_medical_prompt, frame_directive


class VLLMMedicalLLM(VLLMBasePlugin, MedicalLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)

    def name(self) -> str:
        return f"vllm-medical-{self._model}"

    def generate_medical_response(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        action_prompt: str,
        current_user_utterance: str,
    ) -> str:
        messages = build_medical_prompt(
            case_info, dialogue_history, frame_directive(action_prompt, "command"), current_user_utterance
        )
        return self._chat(messages)
