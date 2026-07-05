"""Generic API-backed LLM policy driven ENTIRELY by whatever action_space YAML is loaded --
no hardcoded stage vocabulary in this file (contrast plugins/policy/policy_ours_v2.py's
_CONTROLS/_CONTROL_TO_MCB, hardcoded in Python once per action-space version).

Testing a new control layer (action_space_v3.yaml's EXTEND/RECOMMEND, or any older McBurney-
compatible spec like action_space_v2.yaml, or any future action_space_vN.yaml) needs only a new
YAML + a config pointing policy.type at this class -- never a new policy module.

`allowed_locutions`/`allowed_types`/`realizes` per stage are all OPTIONAL (older McBurney-
compatible specs like action_space_v2.yaml have them; action_space_v3.yaml has neither -- the
stage/locution deliberation paradigm was retired, see core/reward.py's docstring). When a stage
has no `realizes`, its McBurney representative defaults to its own id, so action_space_v3.yaml's
controls ("EXTEND"/"RECOMMEND") ARE PolicyOutput.stage directly -- core/environment.py no longer
stage-gates against a fixed McBurney vocabulary (that safety net was removed along with
r_align), so no mapping is required anymore, it's just still supported for older specs. When a
stage has no `allowed_locutions` at all anywhere in the action space, this class stops asking
the LLM for a locution/locution_type and leaves both "" on the resulting PolicyOutput.

The real control label the LLM chose is always preserved in PolicyOutput.metadata["control"]
(what analysis scripts and core/environment.py's r_fmt should read), regardless of style.
"""
from __future__ import annotations

import json
import re
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load

_FALLBACK_MCB = ("INFORM", "ask_justify", "fact")


def _mcb_stages_so_far(dialogue_history: DialogueHistory) -> set[str]:
    """Realized stage labels already produced this episode. dialogue_history.turns[i].action is
    either "STAGE.locution" (older McBurney-compatible specs) or just "STAGE" (action_space_v3.yaml,
    no locution) -- split(".")[0] handles both since a string with no "." is its own single-element
    split result."""
    return {
        t.action.split(".")[0].upper()
        for t in dialogue_history.turns
        if t.speaker == "medical" and t.action
    }


class ActionSpaceLLMPolicy(VLLMBasePlugin, PolicyPlugin):
    """Prompted LLM policy over whatever control stages `action_space` defines."""

    needs_verification = True

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._use_fact_validator = bool(config.get("use_fact_validator", True))
        self.needs_verification = self._use_fact_validator

        self._stages: dict[str, dict] = self.action_space["stages"]
        self._locutions: dict[str, dict] = self.action_space.get("locutions", {})
        self._transitions: dict[str, list[str]] = self.action_space.get("transitions", {})
        # v3-style specs declare allowed_locutions per stage; v4-style specs (no McBurney
        # sublayer) declare none anywhere -- in that case skip the locution question entirely
        # rather than asking the LLM to invent a meaningless field.
        self._has_locutions: bool = any(info.get("allowed_locutions") for info in self._stages.values())
        # control -> (mcb_stage, default_locution, default_type), read straight off each
        # stage's own `realizes`/allowed_locutions/allowed_types. Falling back to the control's
        # own id when `realizes` is absent is what makes this work unmodified on action_space.yaml
        # (v1), whose stage ids already ARE McBurney stages.
        self._mcb_repr: dict[str, tuple[str, str, str]] = {}
        for cid, info in self._stages.items():
            mcb_stage = str((info.get("realizes") or [cid])[0]).upper()
            locs = info.get("allowed_locutions") or ["assert"]
            types = info.get("allowed_types") or ["fact"]
            self._mcb_repr[cid] = (mcb_stage, locs[0], types[0])

    def name(self) -> str:
        return f"action-space-llm-policy-{self._model}"

    def load(self) -> None:
        pass

    # ── prompt construction (entirely from the loaded action_space) ─────────────────────
    def _stage_menu_text(self) -> str:
        lines = []
        for cid, info in self._stages.items():
            desc = " ".join(str(info.get("description", "")).split())
            line = f"- {cid}: {desc}"
            if info.get("allowed_locutions") or info.get("allowed_types"):
                locs = ", ".join(info.get("allowed_locutions", []))
                types = ", ".join(info.get("allowed_types", []))
                line += f"\n  (locutions: {locs} | types: {types})"
            lines.append(line)
        return "\n".join(lines)

    def _locution_glossary_text(self) -> str:
        return "\n".join(
            f"- {lid}: {info.get('description', '')}" for lid, info in self._locutions.items()
        )

    def _reachable(self, dialogue_history: DialogueHistory) -> list[str]:
        """Which control stages satisfy `transitions` right now, derived from the realized
        McBurney history (see _mcb_stages_so_far) -- generic equivalent of
        policy_ours_v2.py's _reachable_controls, but reading transitions from the YAML instead
        of a hardcoded has_inform/has_proposal/... check."""
        occurred_mcb = _mcb_stages_so_far(dialogue_history)
        occurred_controls = {cid for cid, (mcb, _, _) in self._mcb_repr.items() if mcb in occurred_mcb}
        reachable = [
            cid for cid in self._stages
            if all(req in occurred_controls for req in self._transitions.get(cid, []))
        ]
        return reachable or list(self._stages)

    def _build_messages(
        self,
        vt: VerificationTemplate | None,
        current_user_utterance: str,
        dialogue_history: DialogueHistory,
        reachable: list[str],
    ) -> list[dict[str, str]]:
        tmpl = _load("action_space_llm_policy")
        fv_block = ""
        if self._use_fact_validator and vt is not None:
            fv_block = tmpl["fact_validation_block"].format(
                overall_relation=vt.overall_relation, confidence=vt.confidence, reasoning=vt.reasoning,
            )
        locutions_section, locution_json_fields = "", ""
        if self._has_locutions:
            locutions_section = (
                "\n## Locutions (pick one that is valid for your chosen stage; see each stage's "
                f"own list above)\n{self._locution_glossary_text()}\n"
            )
            locution_json_fields = (
                '\n    "locution": "<one locution id valid for that stage>",'
                '\n    "locution_type": "<one type id valid for that stage>",'
            )
        system = tmpl["system"].format(
            stage_menu=self._stage_menu_text(),
            locutions_section=locutions_section,
            locution_json_fields=locution_json_fields,
        )
        user = tmpl["user"].format(
            reachable_stages=", ".join(reachable),
            current_user_utterance=current_user_utterance,
            fact_validation_block=fv_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ── output parsing ───────────────────────────────────────────────────────────────────
    def _parse(self, raw: str, reachable: list[str]) -> tuple[str, str, str, str]:
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        control, locution, locution_type, guidance = reachable[0], "", "", ""
        if "{" in text and "}" in text:
            blob = text[text.find("{"): text.rfind("}") + 1]
            try:
                data = json.loads(blob)
                c = str(data.get("stage", "")).strip().upper()
                if c in self._stages:
                    control = c
                locution = str(data.get("locution", "")).strip().lower()
                locution_type = str(data.get("locution_type", "")).strip().lower()
                guidance = str(data.get("action_guidance", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                pass
        if control not in reachable:
            control = reachable[0]
        return control, locution, locution_type, guidance

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        reachable = self._reachable(dialogue_history)
        messages = self._build_messages(verification_template, current_user_utterance, dialogue_history, reachable)
        raw = self._chat(messages, response_format={"type": "json_object"})
        control, locution, locution_type, guidance = self._parse(raw, reachable)

        mcb_stage, default_loc, default_type = self._mcb_repr.get(control, _FALLBACK_MCB)
        info = self._stages.get(control, {})
        if self._has_locutions:
            if locution not in (info.get("allowed_locutions") or []):
                locution = default_loc
            if locution_type not in (info.get("allowed_types") or []):
                locution_type = default_type
            action_id = f"{mcb_stage}.{locution}"
        else:
            locution, locution_type = "", ""
            action_id = mcb_stage

        return PolicyOutput(
            stage=mcb_stage,
            locution=locution,
            locution_type=locution_type,
            action_id=action_id,
            action_prompt=guidance or f"Apply the {control} move.",
            confidence=1.0,
            metadata={"policy": self.name(), "control": control, "raw": raw},
        )
