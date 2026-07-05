"""policy_ours_v3 -- local trainable GRPO policy over action_space_v3.yaml's EXTEND/RECOMMEND.

Subclasses LocalQwenPolicy so it inherits the GRPO training hooks unchanged (load / QLoRA /
action_logprob / save). It overrides only:
  1. prompt construction  -> the v3 EXTEND/RECOMMEND disclosure-mode prompt. Reuses
     prompts/action_space_v3_llm_policy.yaml verbatim (same schema as the API-backed
     action_space_v3_llm_policy.py) so student/teacher prompts stay identical if a same-prompt
     teacher is used for distillation.
  2. output parsing        -> {stage in EXTEND/RECOMMEND, action_guidance}

Unlike policy_ours_v2.py, no McBurney (stage, locution, locution_type) realization mapping is
needed here: action_space_v3 dropped that layer entirely. locution/locution_type are emitted
empty, exactly like action_space_v3_llm_policy.py and action_space_v3_oracle_policy.py already
do, and core/reward.py's r_align term was retired specifically because v3 controls have no
McBurney BASE table to score against.

_reachable_controls/_parse_control_guidance are exposed at module level (mirroring
policy_ours_v2.py's _reachable_controls/_parse_control_guidance) so the v3 prompt-baseline
policies (react_control_v3_policy.py, reflexion_v3_policy.py) can import the exact same
reachability/parsing logic instead of re-deriving it.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

from core.prompt_builder import _load
from core.schemas import CaseInfo, DialogueHistory, PolicyOutput, VerificationTemplate
from core.token_tracker import tracker as _tracker
from plugins.policy.qwen_policy import LocalQwenPolicy

_V3_CONTROLS = ("EXTEND", "RECOMMEND")


def _controls_so_far(dialogue_history: DialogueHistory) -> set[str]:
    return {
        str(t.action).split(".")[0].upper()
        for t in dialogue_history.turns
        if t.speaker == "medical" and t.action
    }


def _reachable_controls(dialogue_history: DialogueHistory, transitions: dict[str, list[str]]) -> list[str]:
    """action_space_v3.yaml's transition gate: RECOMMEND requires a prior EXTEND."""
    occurred = _controls_so_far(dialogue_history)
    reachable = [
        cid for cid in _V3_CONTROLS
        if all(req in occurred for req in transitions.get(cid, []))
    ]
    return reachable or ["EXTEND"]


def _parse_control_guidance(raw: str, reachable: list[str]) -> tuple[str, str]:
    """Parse {stage, action_guidance} from the model's JSON generation, with robust fallbacks.
    control defaults to reachable[0] (never a control the transition gate forbids this turn)."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    control, guidance = reachable[0], ""
    if "{" in text and "}" in text:
        blob = text[text.find("{"): text.rfind("}") + 1]
        try:
            data = json.loads(blob)
            maybe_control = str(data.get("stage", "")).strip().upper()
            if maybe_control in _V3_CONTROLS:
                control = maybe_control
            guidance = str(data.get("action_guidance", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    if control not in reachable:
        control = reachable[0]
    return control, guidance


def _default_guidance(control: str) -> str:
    return (
        "Do not state your own conclusion. Help the clinician reason by surfacing one "
        "missing consideration or asking one targeted justification question."
        if control == "EXTEND"
        else
        "State your own conclusion directly and explicitly, then give concise reasoning."
    )


class PolicyOursV3(LocalQwenPolicy):
    needs_verification = True

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)
        self._fact_validator_on: bool = bool(config.get("use_fact_validator", True))
        self._stages: dict[str, dict[str, Any]] = {
            cid: action_space["stages"][cid]
            for cid in _V3_CONTROLS
            if cid in action_space.get("stages", {})
        }
        missing = [cid for cid in _V3_CONTROLS if cid not in self._stages]
        if missing:
            raise ValueError(
                "PolicyOursV3 requires action_space_v3-style controls; "
                f"missing: {', '.join(missing)}"
            )
        self._transitions: dict[str, list[str]] = action_space.get("transitions", {})

    def name(self) -> str:
        return f"policy-ours-v3-{self._mode}"

    def _reachable(self, dialogue_history: DialogueHistory) -> list[str]:
        return _reachable_controls(dialogue_history, self._transitions)

    def _stage_menu_text(self) -> str:
        lines = []
        for cid in _V3_CONTROLS:
            desc = " ".join(str(self._stages[cid].get("description", "")).split())
            lines.append(f"- {cid}: {desc}")
        return "\n".join(lines)

    # ── prompt (state s_t over the EXTEND/RECOMMEND control space) ──────────
    def _messages_ours(
        self,
        vt: VerificationTemplate | None,
        current_user_utterance: str,
        dialogue_history: DialogueHistory,
    ) -> list[dict[str, str]]:
        tmpl = _load("action_space_v3_llm_policy")
        reachable = self._reachable(dialogue_history)
        fv_block = ""
        if self._fact_validator_on and vt is not None:
            fv_block = tmpl["fact_validation_block"].format(
                overall_relation=vt.overall_relation,
                confidence=vt.confidence,
                reasoning=vt.reasoning,
            )
        user = tmpl["user"].format(
            reachable_stages=", ".join(reachable),
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
        )
        return [
            {"role": "system", "content": tmpl["system"].format(stage_menu=self._stage_menu_text())},
            {"role": "user", "content": user},
        ]

    # ── output parsing (no McBurney realization -- v3 has none) ─────────────
    def _to_output(self, raw: str, dialogue_history: DialogueHistory) -> PolicyOutput:
        reachable = self._reachable(dialogue_history)
        control, guidance = _parse_control_guidance(raw, reachable)
        guidance = guidance or _default_guidance(control)

        return PolicyOutput(
            stage=control,
            locution="",
            locution_type="",
            action_id=control,
            action_prompt=guidance,
            confidence=1.0,
            metadata={"policy": self.name(), "control": control, "raw": raw},
        )

    # ── generation (mirror LocalQwenPolicy, but with the v3 prompt + parser) ─
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
        """Rollout-time sampling. Records prompt_ids + action_ids for the GRPO update."""
        import torch
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
