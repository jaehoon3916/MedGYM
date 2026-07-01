from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plugins.vllm_base import VLLMBasePlugin
from plugins.policy.base import PolicyPlugin
from plugins.policy.prompt_policy import _parse_action, _DEFAULT_ACTION
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.prompt_builder import _load
from core.reward_align import ctx_from_history, allowed_stages

_DEFAULT_AI_BLIND_CACHE = Path(__file__).resolve().parents[2] / "outputs" / "ai_blind_trials_cache.json"


class DeliberationLLMPolicy(VLLMBasePlugin, PolicyPlugin):
    """LLM-based deliberation policy: selects a stage/locution AND generates specific
    behavioral guidance (action_guidance) for the medical AI in a single call.

    Improves on PromptPolicy by having the LLM write a concrete 2-4 sentence instruction
    instead of the thin "Stage: X | Locution: Y\\nApply this." fallback text.
    """

    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        VLLMBasePlugin.__init__(self, config)
        PolicyPlugin.__init__(self, config, action_space)
        # When True, this policy becomes a "true oracle": it is told the ground-truth correct
        # option (the literal letter + option text) and asked to pick the deliberation move most
        # likely to steer the clinician toward it (without ever revealing the letter outright --
        # the doctor must still form the belief themselves, or the trajectory is meaningless).
        # This is a CAPABILITY-CEILING condition, not a realistic policy: it answers "if the
        # policy already had the answer, how well could pure legitimate deliberation still
        # recover an anchored wrong doctor" -- NOT "can a policy that doesn't know the answer
        # behave well." Do not read results from this mode as validating the action-space /
        # commitment-level design; it trivially benefits from knowing the destination.
        self._reveal_correct_option = bool(config.get("reveal_correct_option", False))
        # The realistic counterpart: reveals ONLY the meta fact (am I independently right, is
        # the clinician currently right) -- CORRECT/INCORRECT labels -- and NEVER the option
        # letter or option text anywhere in the prompt. This is the actual quadrant-aware
        # design the framework is meant to validate; unlike reveal_correct_option it doesn't
        # hand the policy the destination, only a trust signal to set its posture (assert
        # confidently vs defer vs probe). Independent of reveal_correct_option -- either or
        # both may be set, though in practice only one is used per run.
        self._reveal_quadrant_meta = bool(config.get("reveal_quadrant_meta", False))
        # Same two facts as reveal_quadrant_meta, but with the hand-written "-> do X" playbook
        # stripped out (see oracle_quadrant_block_raw in the prompt file). reveal_quadrant_meta
        # tests OUR playbook; this tests whether the policy LLM's own judgment, given only the
        # trust signal, produces good behavior -- the actual question the framework needs
        # answered. Takes precedence over reveal_quadrant_meta if both are set (shouldn't be).
        self._reveal_quadrant_meta_raw = bool(config.get("reveal_quadrant_meta_raw", False))
        # Deployable, NO-ground-truth counterpart to the meta modes. Instead of the true
        # correct/incorrect quadrant, it feeds the policy two inference-time ESTIMATES:
        #   - doctor-side: the fact-verifier's overall_relation on the doctor's latest claim
        #     (supported->likely correct, contradicted->likely wrong).
        #   - AI-side: the medical AI's own self-reported confidence from its last turn.
        # Neither uses case_info.correct_option nor the ai_blind cache -- nothing ground-truth
        # reaches the prompt. Raw (facts only, no playbook), since _raw >= the rule version.
        self._estimate_from_signals = bool(config.get("estimate_from_signals", False))
        # Fact validator is a feature OF this policy, toggled here (not a separate policy class).
        # use_fact_validator=false => ablation: NullFactValidator is injected by config.py (0 API
        # calls) and needs_verification=False makes the env skip passing the template, so the
        # {fact_validation_block} placeholder is filled with "" -- same template, section omitted.
        self._use_fact_validator = bool(config.get("use_fact_validator", True))
        self.needs_verification = self._use_fact_validator

        # Explicit quadrant flag (am I independently right, is the clinician currently right) --
        # needed by either oracle mode above, since both use it (reveal_correct_option layers it
        # on top of the literal answer; reveal_quadrant_meta uses it as the ONLY oracle signal).
        # Sourced from the ai_blind majority-vote cache (3 independent no-dialogue trials per
        # case, already computed by scripts/analyze_ai_blind.py) rather than a fresh call.
        self._ai_alone_correct_by_case: dict[str, bool] = {}
        if self._reveal_correct_option or self._reveal_quadrant_meta or self._reveal_quadrant_meta_raw:
            cache_path = Path(config.get("ai_blind_cache", _DEFAULT_AI_BLIND_CACHE))
            if cache_path.is_file():
                trials: dict[str, list[bool]] = json.loads(cache_path.read_text())
                self._ai_alone_correct_by_case = {
                    cid: sum(1 for v in votes if v) >= 2 for cid, votes in trials.items()
                }
            else:
                print(
                    f"[deliberation_llm] an oracle/meta flag is set but no ai_blind cache at "
                    f"{cache_path} -- explicit quadrant block will omit the 'your own independent "
                    f"judgment' fact for every case (doctor-side fact still works)."
                )

    def name(self) -> str:
        if self._reveal_correct_option:
            suffix = "-oracle"
        elif self._reveal_quadrant_meta_raw:
            suffix = "-meta-oracle-raw"
        elif self._reveal_quadrant_meta:
            suffix = "-meta-oracle"
        elif self._estimate_from_signals:
            suffix = "-estimate"
        else:
            suffix = ""
        if not self._use_fact_validator:
            suffix += "-no-validator"
        return f"deliberation-llm-policy-{self._model}{suffix}"

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate | None = None,
    ) -> PolicyOutput:
        tmpl = _load("deliberation_llm_policy")
        oracle_hint = ""
        if self._reveal_correct_option:
            # Literal-answer mode: the actual letter + option text goes in. Capability-ceiling
            # only -- see the __init__ docstring warning.
            options_text = "\n".join(f"  {k}. {v}" for k, v in case_info.options.items())
            oracle_hint = tmpl["oracle_hint"].format(
                correct_option=case_info.correct_option,
                options_text=options_text,
            )

        if self._reveal_correct_option or self._reveal_quadrant_meta or self._reveal_quadrant_meta_raw:
            # Meta-only fact: CORRECT/INCORRECT labels, never the option letter or text. This is
            # the only oracle content reveal_quadrant_meta/reveal_quadrant_meta_raw ever injects.
            ai_ok = self._ai_alone_correct_by_case.get(case_info.case_id)
            doctor_belief = next(
                (
                    t.user_state.model_dump().get("belief")
                    for t in reversed(dialogue_history.turns)
                    if t.speaker == "user" and t.user_state is not None
                ),
                None,
            )
            doctor_ok = (
                str(doctor_belief).strip().upper() == str(case_info.correct_option).strip().upper()
                if doctor_belief else None
            )
            if ai_ok is not None or doctor_ok is not None:
                block_key = "oracle_quadrant_block_raw" if self._reveal_quadrant_meta_raw else "oracle_quadrant_block"
                oracle_hint += tmpl[block_key].format(
                    ai_alone_status="CORRECT" if ai_ok else "INCORRECT" if ai_ok is not None else "UNKNOWN",
                    doctor_status=(
                        "CORRECT" if doctor_ok else "INCORRECT" if doctor_ok is not None
                        else "UNKNOWN (no position stated yet)"
                    ),
                )

        if self._estimate_from_signals:
            # Deployable estimates -- NO ground truth. Doctor-side: the fact-verifier's relation
            # on the doctor's latest claim. AI-side: the medical AI's own confidence from its
            # last turn (metadata["confidence"]; None on turn 0 -> UNKNOWN).
            doctor_relation_status = {
                "supported": "LIKELY CORRECT",
                "contradicted": "LIKELY WRONG",
            }.get(
                verification_template.overall_relation if verification_template else "",
                "UNCLEAR",
            )
            ai_conf = next(
                (
                    t.metadata.get("confidence")
                    for t in reversed(dialogue_history.turns)
                    if t.speaker == "medical" and t.metadata.get("confidence") is not None
                ),
                None,
            )
            if ai_conf is None:
                ai_conf_status = "UNKNOWN (you have not spoken yet)"
            elif ai_conf >= 0.7:
                ai_conf_status = "CONFIDENT"
            elif ai_conf <= 0.4:
                ai_conf_status = "UNCERTAIN"
            else:
                ai_conf_status = "MODERATELY CONFIDENT"
            oracle_hint += tmpl["estimate_quadrant_block_raw"].format(
                ai_conf_status=ai_conf_status,
                doctor_relation_status=doctor_relation_status,
            )
        ctx = ctx_from_history(dialogue_history)
        reachable_text = ", ".join(sorted(allowed_stages(ctx)))

        # Single template; the validator section is the only difference between modes.
        if self._use_fact_validator and verification_template is not None:
            fact_validation_block = tmpl["fact_validation_block"].format(
                overall_relation=verification_template.overall_relation,
                confidence=verification_template.confidence,
                reasoning=verification_template.reasoning,
                evidence_gaps=verification_template.evidence_gaps,
            )
        else:
            fact_validation_block = ""

        user_content = tmpl["user"].format(
            current_user_utterance=current_user_utterance,
            fact_validation_block=fact_validation_block,
            dialogue=dialogue_history.to_prompt_with_actions(),
            oracle_hint=oracle_hint,
            reachable_stages=reachable_text,
        )

        messages = [
            {"role": "system", "content": tmpl["system"]},
            {"role": "user", "content": user_content},
        ]
        raw = self._chat(messages, temperature=0.0,
                         response_format={"type": "json_object"})
        try:
            data = json.loads(raw)
            stage, locution, locution_type = _parse_action(data)
            action_guidance = str(data.get("action_guidance", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            stage, locution, locution_type = _DEFAULT_ACTION
            action_guidance = ""

        # fallback: mirror prompt_policy's thin text so medical LLM is never guidanceless
        if not action_guidance:
            action_guidance = (
                f"Stage: {stage} | Locution: {locution}({locution_type})\n"
                f"Apply this deliberation move in your response."
            )

        return PolicyOutput(
            stage=stage,
            locution=locution,
            locution_type=locution_type,
            action_id=f"{stage}.{locution}",
            action_prompt=action_guidance,
            confidence=1.0,
            metadata={
                "policy": self.name(),
                "model": self._model,
                "reveal_correct_option": self._reveal_correct_option,
                "reveal_quadrant_meta": self._reveal_quadrant_meta,
                "reveal_quadrant_meta_raw": self._reveal_quadrant_meta_raw,
                "estimate_from_signals": self._estimate_from_signals,
                "use_fact_validator": self._use_fact_validator,
                "raw": raw,
            },
        )
