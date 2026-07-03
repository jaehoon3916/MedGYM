"""policy_ours_v2 — hybrid factored-generative deliberation policy (docs/problem_formulation.md).

Subclasses LocalQwenPolicy so it inherits the GRPO training hooks unchanged (load / QLoRA /
action_logprob / save). It overrides only two seams:
  1. prompt construction  -> the 'ours' state s_t = (D_t, v_t, c_t, b_t) over the 3-way control space
  2. output parsing        -> {stage in EXPAND/CHALLENGE/CONVERGE, action_guidance}

Two-layer action (action_space_v2.yaml): the policy reasons at the CONTROL layer (3 stages, the
credit-assignment anchor that appears verbatim in the generated tokens), and each control stage is
mapped to a representative McBurney (stage, locution) REALIZATION so the existing environment stage
gate + r_align table keep working. The actual behavior is carried by the free-text action_guidance,
which becomes the medical AI's action_prompt. The control label is recorded in metadata['control']
for the frontier analysis.
"""
from __future__ import annotations

import json
import re
from typing import Any

from plugins.policy.qwen_policy import LocalQwenPolicy, _build_policy_output
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load
from core.reward_align import ctx_from_history

_CONTROLS = ("EXPAND", "CHALLENGE", "CONVERGE")

# control -> representative McBurney (stage, locution, locution_type) for env/reward compatibility.
# These are realization DEFAULTS; the real behavior lives in the generated action_guidance.
_CONTROL_TO_MCB: dict[str, tuple[str, str, str]] = {
    "EXPAND":    ("INFORM", "assert", "perspective"),
    "CHALLENGE": ("CONSIDER", "assert", "evaluation"),
    "CONVERGE":  ("RECOMMEND", "move", "action"),
}
_FALLBACK_MCB = ("INFORM", "ask_justify", "fact")


def _reachable_controls(ctx: dict[str, Any]) -> list[str]:
    """v2 transition rule (action_space_v2.yaml): EXPAND always; CHALLENGE/CONVERGE need a prior
    EXPAND (== has_inform in the McBurney ctx, since EXPAND realizes INFORM)."""
    if ctx.get("has_inform", False):
        return ["EXPAND", "CHALLENGE", "CONVERGE"]
    return ["EXPAND"]


def _parse_control_guidance(raw: str) -> tuple[str, str]:
    """Parse {stage, action_guidance} from the model's JSON generation, with robust fallbacks.
    Returns (control_stage, action_guidance). control defaults to EXPAND; guidance to ''."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    control, guidance = "EXPAND", ""
    # try strict-ish JSON on the outermost braces
    if "{" in text and "}" in text:
        blob = text[text.find("{"): text.rfind("}") + 1]
        try:
            data = json.loads(blob)
            control = str(data.get("stage", "")).strip().upper()
            guidance = str(data.get("action_guidance", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    # keyword fallback for the control if JSON failed or stage was junk
    if control not in _CONTROLS:
        m = re.search(r"\b(EXPAND|CHALLENGE|CONVERGE)\b", text.upper())
        control = m.group(1) if m else "EXPAND"
    return control, guidance


class PolicyOursV2(LocalQwenPolicy):
    needs_verification = True

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)
        # v_t (fact-validator relation) is part of state s_t; toggle to drop it for a minimal smoke.
        self._fact_validator_on: bool = bool(config.get("use_fact_validator", True))

    def name(self) -> str:
        return f"policy-ours-v2-{self._mode}"

    # ── prompt (state s_t over the control space) ────────────────────────────
    def _messages_ours(
        self,
        vt: VerificationTemplate | None,
        current_user_utterance: str,
        dialogue_history: DialogueHistory,
    ) -> list[dict[str, str]]:
        tmpl = _load("policy_ours_v2")
        ctx = ctx_from_history(dialogue_history)
        reachable = ", ".join(_reachable_controls(ctx))
        turns_so_far = sum(1 for t in dialogue_history.turns if t.speaker == "medical")

        fv_block = ""
        if self._fact_validator_on and vt is not None:
            fv_block = tmpl["fact_validation_block"].format(
                overall_relation=vt.overall_relation,
                confidence=vt.confidence,
                reasoning=vt.reasoning,
            )
        user = tmpl["user"].format(
            reachable_controls=reachable,
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
            turns_so_far=turns_so_far,
        )
        return [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": user},
        ]

    # ── output parsing (control -> McBurney realization + guidance) ──────────
    def _to_output(self, raw: str, dialogue_history: DialogueHistory) -> PolicyOutput:
        control, guidance = _parse_control_guidance(raw)
        ctx = ctx_from_history(dialogue_history)
        if control not in _reachable_controls(ctx):
            control = "EXPAND"
        mcb_stage, mcb_loc, mcb_type = _CONTROL_TO_MCB.get(control, _FALLBACK_MCB)
        po = _build_policy_output(mcb_stage, mcb_loc, mcb_type, self._mode, raw, self.action_space)
        # The medical AI executes the generated guidance, not the generic stage description.
        if guidance:
            po.action_prompt = guidance
        po.metadata["control"] = control
        po.metadata["policy"] = self.name()
        return po

    # ── generation (mirror parent, but with the ours prompt + parser) ────────
    def _encode(self, dialogue_history, current_user_utterance, verification_template):
        messages = self._messages_ours(verification_template, current_user_utterance, dialogue_history)
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self._enable_thinking,
        )
        return self._tokenizer(text, return_tensors="pt").to(self._model.device)

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        import torch
        inputs = self._encode(dialogue_history, current_user_utterance, verification_template)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens,
                do_sample=False, temperature=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        raw = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._to_output(raw, dialogue_history)

    def sample_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        """Rollout-time sampling; records prompt_ids + action_ids for the GRPO update."""
        import torch
        from types import SimpleNamespace
        from core.token_tracker import tracker as _tracker
        inputs = self._encode(dialogue_history, current_user_utterance, verification_template)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens,
                do_sample=True, temperature=self._temperature,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        action_ids = out[0][prompt_len:]
        raw = self._tokenizer.decode(action_ids, skip_special_tokens=True)
        po = self._to_output(raw, dialogue_history)
        po.metadata["prompt_ids"] = inputs["input_ids"][0].tolist()
        po.metadata["action_ids"] = action_ids.tolist()
        usage = SimpleNamespace(
            prompt_tokens=prompt_len, completion_tokens=int(action_ids.shape[0]),
            total_tokens=prompt_len + int(action_ids.shape[0]),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        prompt_text = self._tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        _tracker.record(self.name(), [{"role": "policy_prompt", "content": prompt_text}], raw, usage)
        return po
