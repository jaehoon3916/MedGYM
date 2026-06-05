from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.user_llm.base import UserLLMPlugin
from core.schemas import CaseInfo, DialogueHistory
from core.prompt_builder import build_user_prompt


class VLLMUserLLM(VLLMBasePlugin, UserLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)

    def name(self) -> str:
        return f"vllm-user-{self._model}"

    def generate_user_utterance(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        turn_id: int = 0,
        user_profile: dict[str, Any] | None = None,
    ) -> str:
        messages = build_user_prompt(case_info, dialogue_history, turn_id=turn_id)
        return self._chat(messages)
