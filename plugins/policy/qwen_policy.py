from __future__ import annotations

import re
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load

_VALID_STAGES = {"INFORM", "PROPOSE", "CONSIDER", "REVISE", "RECOMMEND", "CONFIRM", "CLOSE"}


def _parse_dot_action(raw: str) -> tuple[str, str, str]:
    """Parse 'STAGE.locution' format from model output."""
    # Strip thinking tokens if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    parts = raw.strip().upper().split(".")
    if len(parts) >= 2:
        stage = parts[0]
        locution = parts[1].lower()
    else:
        stage, locution = "INFORM", "ask_justify"
    if stage not in _VALID_STAGES:
        stage = "INFORM"
    return stage, locution, "fact"


def _build_policy_output(
    stage: str,
    locution: str,
    locution_type: str,
    mode: str,
    raw: str = "",
    action_space: dict | None = None,
) -> PolicyOutput:
    action_id = f"{stage}.{locution}"

    stage_desc = ""
    locution_desc = ""
    if action_space:
        stage_info = action_space.get("stages", {}).get(stage, {})
        locution_info = action_space.get("locutions", {}).get(locution, {})
        stage_desc = stage_info.get("description", "")
        locution_desc = locution_info.get("description", "")

    action_prompt = (
        f"[Deliberation Stage: {stage}]\n"
        f"{stage_desc}\n\n"
        f"[Locution: {locution} — type: {locution_type}]\n"
        f"{locution_desc}\n\n"
        f"Apply this deliberation move in your response."
    ).strip()

    return PolicyOutput(
        stage=stage,
        locution=locution,
        locution_type=locution_type,
        action_id=action_id,
        action_prompt=action_prompt,
        confidence=1.0,
        metadata={"policy": mode, "raw_output": raw},
    )


class QwenPolicy(VLLMBasePlugin, PolicyPlugin):
    """Qwen3 policy via vLLM server."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._mode: str = config.get("mode", "baseline")
        self._enable_thinking: bool = config.get("enable_thinking", False)

    def name(self) -> str:
        return f"qwen-policy-{self._mode}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        messages = _build_messages(verification_template, current_user_utterance)
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": self._enable_thinking},
        }
        raw = self._chat(messages, temperature=0.0, max_tokens=16, extra_body=extra_body)
        stage, locution, locution_type = _parse_dot_action(raw)
        return _build_policy_output(stage, locution, locution_type, self._mode, raw, self.action_space)


class LocalQwenPolicy(PolicyPlugin):
    """Qwen3 policy loaded directly via transformers — no vLLM server needed."""

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        PolicyPlugin.__init__(self, config, action_space)
        self._mode: str = config.get("mode", "baseline")
        self._model_path: str = config.get("model", "Qwen/Qwen3-8B")
        self._enable_thinking: bool = config.get("enable_thinking", False)
        self._max_new_tokens: int = config.get("max_tokens", 512)
        self._device: str = config.get("device", "auto")
        self._model = None
        self._tokenizer = None

    def name(self) -> str:
        return f"local-qwen-policy-{self._mode}"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        self._model.eval()

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        import torch
        messages = _build_messages(verification_template, current_user_utterance)
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=None,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        stage, locution, locution_type = _parse_dot_action(raw)
        return _build_policy_output(stage, locution, locution_type, self._mode, raw, self.action_space)


def _build_messages(vt: VerificationTemplate, current_user_utterance: str) -> list[dict[str, str]]:
    tmpl = _load("baseline_policy")
    return [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": tmpl["user"].format(
            overall_relation=vt.overall_relation,
            confidence=vt.confidence,
            short_rationale=vt.short_rationale,
            current_user_utterance=current_user_utterance,
        )},
    ]
