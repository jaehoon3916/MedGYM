from __future__ import annotations

import re
from typing import Any

from types import SimpleNamespace

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load
from core.token_tracker import tracker as _tracker

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
        self._adapter_path: str | None = config.get("adapter_path") or config.get("lora_adapter_path")
        self._load_in_4bit: bool = config.get("load_in_4bit", False)   # QLoRA
        self._model = None
        self._tokenizer = None
        self._train_adapter_name = "default"
        self._ref_adapter_name = "reference"
        self._has_ref_adapter = False

    def name(self) -> str:
        return f"local-qwen-policy-{self._mode}"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        model_kwargs: dict[str, Any] = {"device_map": self._device}
        if self._trainable and self._load_in_4bit:
            # QLoRA: frozen base in 4-bit (~5GB vs ~16GB) so 8B fits a 24GB GPU.
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        self._model = AutoModelForCausalLM.from_pretrained(self._model_path, **model_kwargs)

        if self._trainable:
            from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
            if self._load_in_4bit:
                # IMPORTANT: use_gradient_checkpointing=False — checkpointing disables the KV cache
                # in generate(), making rollout sampling crawl (the "Caching is incompatible…" hang).
                # 4-bit already gives plenty of headroom, so we don't need checkpointing.
                self._model = prepare_model_for_kbit_training(
                    self._model, use_gradient_checkpointing=False
                )
            if self._adapter_path:
                self._model = PeftModel.from_pretrained(
                    self._model, self._adapter_path, is_trainable=True
                )
                try:
                    # GRPO reference should be the frozen SFT initialization, not the bare base
                    # model. Load the same adapter a second time as a frozen ref; the trainable
                    # "default" adapter is updated by the optimizer, while "reference" stays fixed.
                    self._model.load_adapter(
                        self._adapter_path,
                        adapter_name=self._ref_adapter_name,
                        is_trainable=False,
                    )
                    self._model.set_adapter(self._train_adapter_name)
                    self._has_ref_adapter = True
                except Exception as e:  # noqa: BLE001 -- older PEFT versions may lack load_adapter
                    print(
                        f"[policy] warning: could not load frozen reference adapter from "
                        f"{self._adapter_path}: {e}; KL reference will fall back to base model"
                    )
            else:
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
            if self._adapter_path:
                from peft import PeftModel
                self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
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

        # Record this local generate() in the same unified format as every API call.
        # No usage object from a local HF generate, so we synthesize one from token lengths.
        n_action = int(action_ids.shape[0])
        reasoning_tokens = 0
        if self._enable_thinking and "</think>" in raw:
            think_text = raw.split("</think>", 1)[0]
            reasoning_tokens = len(self._tokenizer(think_text, add_special_tokens=False).input_ids)
        usage = SimpleNamespace(
            prompt_tokens=prompt_len,
            completion_tokens=n_action,
            total_tokens=prompt_len + n_action,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        )
        prompt_text = self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        _tracker.record(self.name(), [{"role": "policy_prompt", "content": prompt_text}], raw, usage)
        return po

    def action_logprob(self, prompt_ids: list[int], action_ids: list[int], use_ref: bool = False):
        """Summed per-token logπ(action | prompt).

        For SFT-initialized GRPO, use_ref uses a frozen copy of the initial SFT adapter. For
        scratch LoRA runs, it falls back to adapter-disabled base-model reference.
        """
        import torch
        device = self._model.device
        pid = torch.tensor(prompt_ids, device=device).unsqueeze(0)
        aid = torch.tensor(action_ids, device=device).unsqueeze(0)
        full = torch.cat([pid, aid], dim=1)

        n_keep = aid.shape[1] + 1   # only compute lm_head for the action-predicting positions

        def _compute():
            # logits_to_keep → logits only for the last n_keep positions ([1, A+1, V]), avoiding a
            # full [1, L, vocab] float tensor (the OOM cause). use_cache=False for the training forward.
            logits = self._model(full, use_cache=False, logits_to_keep=n_keep).logits
            logp = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)  # [A, V] → predicts action tokens
            tok = aid[0]
            return logp[torch.arange(tok.shape[0], device=device), tok].sum()

        if use_ref:
            if self._has_ref_adapter:
                active = getattr(self._model, "active_adapter", self._train_adapter_name)
                try:
                    self._model.set_adapter(self._ref_adapter_name)
                    with torch.no_grad():
                        return _compute()
                finally:
                    self._model.set_adapter(active or self._train_adapter_name)
            with torch.no_grad(), self._model.disable_adapter():
                return _compute()
        return _compute()

    def save(self, path: str) -> None:
        self._tokenizer.save_pretrained(path)
        if hasattr(self._model, "set_adapter"):
            self._model.set_adapter(self._train_adapter_name)
        try:
            self._model.save_pretrained(path, selected_adapters=[self._train_adapter_name])
        except TypeError:
            self._model.save_pretrained(path)   # older PEFT: saves the active LoRA adapter


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
