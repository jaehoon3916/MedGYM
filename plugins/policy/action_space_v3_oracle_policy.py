"""Oracle policy specialized for action_space_v3.yaml.

This is intentionally separate from action_space_v3_llm_policy.py. It uses the same A3 controls
(EXTEND / RECOMMEND) but can receive ground-truth oracle signals for controlled experiments:

  - reveal_correct_option: true      -> literal correct option letter/text (capability ceiling)
  - reveal_quadrant_meta: true       -> AI-alone correctness + doctor-current correctness only
  - reveal_quadrant_meta_raw: true   -> same facts, no playbook text
  - reveal_user_state: true          -> perfect knowledge of the DOCTOR's state only (belief_dist,
                                        confidence, burden-driven w_eff/lam_eff, cumulative burden)
                                        plus the persona's static C/BS Bayes params (c0, lambda, w,
                                        rho, kappa, burden_dropout_threshold) -- correctness/y* is
                                        NOT revealed in this mode, only the human side of s_t.
                                        Requires plugins.policy.persona to be set (scripts/run_poc.
                                        py's run_sweep() passes this through automatically from
                                        plugins.user_llm.persona; a single-condition run without
                                        run_sweep must set plugins.policy.persona explicitly).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from core.prompt_builder import _load
from core.schemas import CaseInfo, DialogueHistory, PolicyOutput, VerificationTemplate
from plugins.policy.base import PolicyPlugin
from plugins.vllm_base import VLLMBasePlugin

_V3_CONTROLS = ("EXTEND", "RECOMMEND")
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AI_ALONE_CACHE = _ROOT / "outputs" / "_cache" / "ai_alone_deepseek_v3_2.json"
_LEGACY_AI_BLIND_CACHE = _ROOT / "outputs" / "ai_blind_trials_cache.json"
_BAYES_PARAMS_PATH = _ROOT / "source" / "persona" / "bayes_params.yaml"
_BURDEN_DROPOUT_THRESHOLDS_PATH = _ROOT / "source" / "persona" / "burden_dropout_thresholds.yaml"


def _controls_so_far(dialogue_history: DialogueHistory) -> set[str]:
    return {
        str(t.action).split(".")[0].upper()
        for t in dialogue_history.turns
        if t.speaker == "medical" and t.action
    }


class ActionSpaceV3OraclePolicy(VLLMBasePlugin, PolicyPlugin):
    """A3 oracle policy over EXTEND / RECOMMEND."""

    needs_verification = True

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        self._use_fact_validator = bool(config.get("use_fact_validator", True))
        self.needs_verification = self._use_fact_validator
        self._reveal_correct_option = bool(config.get("reveal_correct_option", False))
        self._reveal_quadrant_meta = bool(config.get("reveal_quadrant_meta", False))
        self._reveal_quadrant_meta_raw = bool(config.get("reveal_quadrant_meta_raw", False))
        self._reveal_user_state = bool(config.get("reveal_user_state", False))
        self._persona: str | None = config.get("persona")

        _reveal_flags = (
            self._reveal_correct_option, self._reveal_quadrant_meta,
            self._reveal_quadrant_meta_raw, self._reveal_user_state,
        )
        if sum(_reveal_flags) > 1:
            raise ValueError(
                "Use at most one of reveal_correct_option / reveal_quadrant_meta / "
                "reveal_quadrant_meta_raw / reveal_user_state at a time."
            )
        if self._reveal_user_state and not self._persona:
            raise ValueError(
                "reveal_user_state requires plugins.policy.persona to be set (run_sweep() sets "
                "this automatically per condition; a single-condition run must set it explicitly)."
            )

        self._stages: dict[str, dict[str, Any]] = {
            cid: action_space["stages"][cid]
            for cid in _V3_CONTROLS
            if cid in action_space.get("stages", {})
        }
        missing = [cid for cid in _V3_CONTROLS if cid not in self._stages]
        if missing:
            raise ValueError(
                "ActionSpaceV3OraclePolicy requires action_space_v3-style controls; "
                f"missing: {', '.join(missing)}"
            )
        self._transitions: dict[str, list[str]] = action_space.get("transitions", {})
        self._ai_alone_correct_by_case = self._load_ai_alone_correct(config)

    def name(self) -> str:
        if self._reveal_correct_option:
            suffix = "-true-oracle"
        elif self._reveal_quadrant_meta_raw:
            suffix = "-meta-oracle-raw"
        elif self._reveal_quadrant_meta:
            suffix = "-meta-oracle"
        elif self._reveal_user_state:
            suffix = "-user-state-oracle"
        else:
            suffix = "-no-oracle"
        if not self._use_fact_validator:
            suffix += "-no-validator"
        return f"action-space-v3-oracle-policy-{self._model}{suffix}"

    def load(self) -> None:
        pass

    def _load_ai_alone_correct(self, config: dict[str, Any]) -> dict[str, bool]:
        cache_path = Path(config.get("ai_alone_cache", _DEFAULT_AI_ALONE_CACHE))
        if not cache_path.is_absolute():
            cache_path = _ROOT / cache_path
        if cache_path.is_file():
            data = json.loads(cache_path.read_text())
            entries = data.get("entries", data) if isinstance(data, dict) else {}
            out: dict[str, bool] = {}
            for key, value in entries.items():
                if isinstance(value, dict):
                    case_id = str(value.get("case_id") or "").strip()
                    if case_id and "correct" in value:
                        out[case_id] = bool(value["correct"])
                elif isinstance(value, bool):
                    # Compat for simple {case_id: bool} caches.
                    out[str(key)] = bool(value)
            if out:
                return out

        if _LEGACY_AI_BLIND_CACHE.is_file():
            legacy = json.loads(_LEGACY_AI_BLIND_CACHE.read_text())
            return {
                str(case_id): sum(1 for v in votes if v) >= 2
                for case_id, votes in legacy.items()
                if isinstance(votes, list)
            }
        return {}

    def _stage_menu_text(self) -> str:
        lines = []
        for cid in _V3_CONTROLS:
            desc = " ".join(str(self._stages[cid].get("description", "")).split())
            lines.append(f"- {cid}: {desc}")
        return "\n".join(lines)

    def _reachable(self, dialogue_history: DialogueHistory) -> list[str]:
        occurred = _controls_so_far(dialogue_history)
        reachable = [
            cid for cid in _V3_CONTROLS
            if all(req in occurred for req in self._transitions.get(cid, []))
        ]
        return reachable or ["EXTEND"]

    def _doctor_ok(self, case_info: CaseInfo, dialogue_history: DialogueHistory) -> bool | None:
        belief = next(
            (
                t.user_state.model_dump().get("belief")
                for t in reversed(dialogue_history.turns)
                if t.speaker == "user" and t.user_state is not None
            ),
            None,
        )
        if not belief:
            return None
        return str(belief).strip().upper() == str(case_info.correct_option).strip().upper()

    def _latest_user_state(self, dialogue_history: DialogueHistory) -> dict[str, Any] | None:
        return next(
            (
                t.user_state.model_dump()
                for t in reversed(dialogue_history.turns)
                if t.speaker == "user" and t.user_state is not None
            ),
            None,
        )

    def _persona_bs_c_params(self) -> dict[str, Any]:
        """Static Bayes dials for self._persona (source/persona/bayes_params.yaml +
        burden_dropout_thresholds.yaml) -- the C axis (c0, lambda, w) and BS axis (rho, kappa,
        burden_dropout_threshold). A run config's plugins.user_llm.bayes override is NOT
        reflected here (would require threading that override through to the policy too);
        this reads the named persona's yaml defaults only."""
        bayes = yaml.safe_load(_BAYES_PARAMS_PATH.read_text())
        thresholds = yaml.safe_load(_BURDEN_DROPOUT_THRESHOLDS_PATH.read_text())
        params = dict(bayes["personas"][self._persona])
        params["burden_dropout_threshold"] = thresholds[self._persona]
        return params

    def _oracle_block(self, tmpl: dict[str, str], case_info: CaseInfo, dialogue_history: DialogueHistory) -> str:
        if self._reveal_correct_option:
            options_text = "\n".join(f"  {k}. {v}" for k, v in case_info.options.items())
            return tmpl["true_oracle_block"].format(
                correct_option=case_info.correct_option,
                correct_option_text=case_info.options.get(case_info.correct_option, ""),
                options_text=options_text,
            )

        if self._reveal_quadrant_meta or self._reveal_quadrant_meta_raw:
            ai_ok = self._ai_alone_correct_by_case.get(str(case_info.case_id))
            doctor_ok = self._doctor_ok(case_info, dialogue_history)
            key = "meta_oracle_block_raw" if self._reveal_quadrant_meta_raw else "meta_oracle_block"
            return tmpl[key].format(
                ai_alone_status="CORRECT" if ai_ok else "INCORRECT" if ai_ok is not None else "UNKNOWN",
                doctor_status=(
                    "CORRECT" if doctor_ok else "INCORRECT" if doctor_ok is not None
                    else "UNKNOWN (no stated option)"
                ),
            )

        if self._reveal_user_state:
            state = self._latest_user_state(dialogue_history)
            bs_c = self._persona_bs_c_params()
            if state is None:
                belief_dist_text = "(doctor has not stated a belief yet)"
                confidence_text = w_eff_text = lam_eff_text = burden_text = "UNKNOWN"
            else:
                belief_dist = state.get("belief_dist") or {}
                belief_dist_text = ", ".join(f"{k}={v:.2f}" for k, v in sorted(belief_dist.items()))
                confidence_text = f"{state.get('confidence'):.2f}" if state.get("confidence") is not None else "UNKNOWN"
                w_eff_text = state.get("w_eff", "UNKNOWN (turn 0 -- no burden response yet)")
                lam_eff_text = state.get("lam_eff", "UNKNOWN (turn 0 -- no burden response yet)")
                burden_cum = state.get("cognitive_burden_cumulative")
                threshold = bs_c["burden_dropout_threshold"]
                burden_text = (
                    f"{burden_cum:.1f} / {threshold:.1f} dropout threshold"
                    if burden_cum is not None else "0 (no turns scored yet)"
                )
            return tmpl["user_state_oracle_block"].format(
                belief_dist=belief_dist_text,
                confidence=confidence_text,
                w_eff=w_eff_text,
                lam_eff=lam_eff_text,
                cumulative_burden=burden_text,
                persona=self._persona,
                c0=bs_c["c0"], lam=bs_c["lambda"], w=bs_c["w"],
                rho=bs_c["rho"], kappa=bs_c["kappa"],
                burden_dropout_threshold=bs_c["burden_dropout_threshold"],
            )

        return tmpl["no_oracle_block"]

    def _build_messages(
        self,
        case_info: CaseInfo,
        vt: VerificationTemplate | None,
        current_user_utterance: str,
        dialogue_history: DialogueHistory,
        reachable: list[str],
    ) -> list[dict[str, str]]:
        tmpl = _load("action_space_v3_oracle_policy")
        fv_block = ""
        if self._use_fact_validator and vt is not None:
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
            oracle_block=self._oracle_block(tmpl, case_info, dialogue_history),
        )
        return [
            {"role": "system", "content": tmpl["system"].format(stage_menu=self._stage_menu_text())},
            {"role": "user", "content": user},
        ]

    def _parse(self, raw: str, reachable: list[str]) -> tuple[str, str]:
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

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        reachable = self._reachable(dialogue_history)
        messages = self._build_messages(
            case_info,
            verification_template,
            current_user_utterance,
            dialogue_history,
            reachable,
        )
        raw = self._chat(messages, response_format={"type": "json_object"})
        control, guidance = self._parse(raw, reachable)

        if not guidance:
            if control == "EXTEND":
                guidance = (
                    "Do not state your own conclusion. Surface one high-yield clinical criterion "
                    "or ask one targeted question that helps the clinician reassess their reasoning."
                )
            elif self._reveal_correct_option:
                # Only this mode actually knows the answer -- safe to say so in the fallback.
                guidance = (
                    "State the clinically correct option directly and explicitly, then give concise "
                    "case-based reasoning."
                )
            else:
                # Neutral fallback for meta/user-state/no-oracle modes: none of them are told the
                # answer, so the fallback must not presuppose one is known (see file docstring).
                guidance = "State your own conclusion directly and explicitly, then give concise reasoning."

        return PolicyOutput(
            stage=control,
            locution="",
            locution_type="",
            action_id=control,
            action_prompt=guidance,
            confidence=1.0,
            metadata={
                "policy": self.name(),
                "control": control,
                "reveal_correct_option": self._reveal_correct_option,
                "reveal_quadrant_meta": self._reveal_quadrant_meta,
                "reveal_quadrant_meta_raw": self._reveal_quadrant_meta_raw,
                "reveal_user_state": self._reveal_user_state,
                "raw": raw,
            },
        )
