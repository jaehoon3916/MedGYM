from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.final_judge_llm.base import FinalJudgeLLMPlugin
from core.schemas import CaseInfo, DialogueHistory, FinalJudgement
from core.prompt_builder import build_final_judge_prompt
from core.json_utils import safe_json_load


class VLLMFinalJudgeLLM(VLLMBasePlugin, FinalJudgeLLMPlugin):
    def __init__(self, config: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)

    def name(self) -> str:
        return f"vllm-final-judge-{self._model}"

    def judge(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
    ) -> FinalJudgement:
        messages = build_final_judge_prompt(case_info, dialogue_history)
        raw = self._chat(messages, temperature=0.0, response_format={"type": "json_object"})
        data = safe_json_load(raw)
        option = str(data.get("concluded_option", "none")).strip() or "none"
        # is_correct is computed deterministically by the environment against the gold option.
        return FinalJudgement(concluded_option=option, reason=str(data.get("reason", "")).strip())
