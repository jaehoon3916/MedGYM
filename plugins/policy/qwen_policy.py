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
        messages = _build_messages(
            verification_template, current_user_utterance, dialogue_history,
            self.action_space, _last_medical_action(dialogue_history),
        )
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
        # training (GRPO) knobs — off by default so the inference path is unchanged
        self._trainable: bool = config.get("trainable", False)
        self._temperature: float = config.get("temperature", 0.8)
        self._lora_cfg: dict = config.get("lora", {})
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
        if self._trainable:
            from peft import LoraConfig, get_peft_model
            lc = self._lora_cfg
            peft_cfg = LoraConfig(
                r=lc.get("r", 16),
                lora_alpha=lc.get("alpha", 32),
                lora_dropout=lc.get("dropout", 0.05),
                target_modules=lc.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                task_type="CAUSAL_LM",
            )
            self._model = get_peft_model(self._model, peft_cfg)
            self._model.train()
        else:
            self._model.eval()

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        import torch
        messages = _build_messages(
            verification_template, current_user_utterance, dialogue_history,
            self.action_space, _last_medical_action(dialogue_history),
        )
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

    # ── GRPO training hooks ──────────────────────────────────────────────────

    def _encode(self, dialogue_history, current_user_utterance, verification_template):
        messages = _build_messages(
            verification_template, current_user_utterance, dialogue_history,
            self.action_space, _last_medical_action(dialogue_history),
        )
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self._enable_thinking,
        )
        return self._tokenizer(text, return_tensors="pt").to(self._model.device)

    def sample_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        """Rollout-time sampling (do_sample). Records prompt_ids + action_ids for the GRPO update."""
        import torch
        inputs = self._encode(dialogue_history, current_user_utterance, verification_template)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=True,
                temperature=self._temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        action_ids = out[0][prompt_len:]
        raw = self._tokenizer.decode(action_ids, skip_special_tokens=True)
        stage, locution, locution_type = _parse_dot_action(raw)
        po = _build_policy_output(stage, locution, locution_type, self._mode, raw, self.action_space)
        po.metadata["prompt_ids"] = inputs["input_ids"][0].tolist()
        po.metadata["action_ids"] = action_ids.tolist()
        return po

    def action_logprob(self, prompt_ids: list[int], action_ids: list[int], use_ref: bool = False):
        """Summed per-token logπ(action | prompt). use_ref → adapter-disabled reference policy (no grad)."""
        import torch
        device = self._model.device
        pid = torch.tensor(prompt_ids, device=device).unsqueeze(0)
        aid = torch.tensor(action_ids, device=device).unsqueeze(0)
        full = torch.cat([pid, aid], dim=1)

        def _compute():
            logits = self._model(full).logits.float()        # [1, L, V]
            plen = pid.shape[1]
            logp = torch.log_softmax(logits[0, plen - 1:-1, :], dim=-1)  # predicts the action tokens
            tok = aid[0]
            return logp[torch.arange(tok.shape[0], device=device), tok].sum()

        if use_ref:
            with torch.no_grad(), self._model.disable_adapter():
                return _compute()
        return _compute()

    def save(self, path: str) -> None:
        self._tokenizer.save_pretrained(path)
        self._model.save_pretrained(path)   # PEFT saves the LoRA adapter


def _format_action_space(action_space: dict | None) -> str:
    """Render the stages (with descriptions + allowed locutions) and the locution glossary."""
    if not action_space:
        return ""
    lines = ["Stages:"]
    for sid, info in action_space.get("stages", {}).items():
        locs = ", ".join(info.get("allowed_locutions", []))
        lines.append(f"  {sid} — {info.get('description', '')} [locutions: {locs}]")
    lines.append("Locutions:")
    for lid, info in action_space.get("locutions", {}).items():
        lines.append(f"  {lid} — {info.get('description', '')}")
    return "\n".join(lines)


def _last_medical_action(dialogue_history: DialogueHistory) -> str | None:
    """The action_id of the most recent medical (AI) turn, if any."""
    return next(
        (t.action for t in reversed(dialogue_history.turns) if t.speaker == "medical" and t.action),
        None,
    )


def _build_messages(
    vt: VerificationTemplate,
    current_user_utterance: str,
    dialogue_history: DialogueHistory,
    action_space: dict | None,
    last_action: str | None,
) -> list[dict[str, str]]:
    tmpl = _load("baseline_policy")
    return [
        {"role": "system", "content": tmpl["system"].format(
            action_guide=_format_action_space(action_space),
        )},
        {"role": "user", "content": tmpl["user"].format(
            dialogue=dialogue_history.to_prompt(),
            last_action=last_action or "(none yet)",
            overall_relation=vt.overall_relation,
            confidence=vt.confidence,
            reasoning=vt.reasoning,
            current_user_utterance=current_user_utterance,
        )},
    ]
