from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.schemas import CaseInfo, DialogueHistory, EpisodeConfig, StepResult, Observation, UserState, FinalJudgement
from core.logger import RolloutLogger
from core.token_tracker import tracker as _tracker
from plugins.user_llm.base import UserLLMPlugin
from plugins.medical_llm.base import MedicalLLMPlugin
from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from plugins.policy.base import PolicyPlugin
from plugins.final_judge_llm.base import FinalJudgeLLMPlugin


def _decision_letter(belief: Any) -> str:
    """Extract an A–D option letter from a persona's self-reported belief; 'none' if absent.

    Lettered beliefs (options shown) are already 'A'..'D'; free-text falls back to the first
    A–D token. This only FORMATS the persona's own stated belief — it does not decide anything.
    """
    s = str(belief or "").strip().upper()
    if s in ("A", "B", "C", "D"):
        return s
    m = re.search(r"(?<![A-Z])([A-D])(?![A-Z])", s)   # standalone A–D token, not a letter inside a word
    return m.group(1) if m else "none"


class MedicalHACEnvironment:
    """Agent-external dialogue environment.

    Gym-style: `reset()` produces the first Observation (clinician utterance + fact-validation), and
    `step(policy_output)` applies the action (medical response), advances to the next clinician turn,
    and returns a StepResult (with the per-step r_fmt reward -- the old McBurney r_align process
    reward was retired with the deliberation/stage-locution paradigm; see core/reward.py). The policy lives OUTSIDE the env; the
    trainer drives reset/step directly. `run_episode()` is a thin wrapper that drives it with `self.policy`
    for eval / data-generation, preserving the old behavior.
    """

    def __init__(
        self,
        user_llm: UserLLMPlugin,
        medical_llm: MedicalLLMPlugin,
        fact_validator_llm: FactValidatorLLMPlugin,
        policy: PolicyPlugin,
        config: dict[str, Any],
        final_judge: "FinalJudgeLLMPlugin | None" = None,
    ):
        self.user_llm = user_llm
        self.medical_llm = medical_llm
        self.fact_validator_llm = fact_validator_llm
        self.policy = policy
        self.final_judge = final_judge
        self.config = config

        self._case_info: CaseInfo | None = None
        self._history: DialogueHistory | None = None
        self._turn_id: int = 0
        self._max_turns: int = 2
        self._obs: Observation | None = None
        # current observation fields the next action responds to
        self._cur_user_utterance: str = ""
        self._cur_verification = None
        self._cur_user_state: UserState | None = None
        self._user_closed: bool = False

    # ── gym-style API ────────────────────────────────────────────────────────

    def reset(
        self,
        case_info: CaseInfo,
        episode_config: EpisodeConfig | None = None,
        max_turns: int | None = None,
    ) -> Observation:
        self._case_info = case_info
        self._history = DialogueHistory(case_id=case_info.case_id)
        self._turn_id = 0
        self._max_turns = (
            max_turns if max_turns is not None
            else int(self.config.get("experiment", {}).get("max_turns", 2))
        )
        self._user_closed = False
        self.user_llm.reset_episode(episode_config or EpisodeConfig())
        _tracker.begin_turn()
        self._obs = self._advance_user_turn()
        return self._obs

    @property
    def observation(self) -> Observation | None:
        return self._obs

    @property
    def history(self) -> DialogueHistory | None:
        return self._history

    def step(self, policy_output) -> StepResult:
        assert self._case_info is not None and self._history is not None, "Call reset() before step()"
        t = self._turn_id
        cur_utt = self._cur_user_utterance
        cur_verif = self._cur_verification

        # Apply the action: medical LLM responds using the selected action prompt.
        medical_response, medical_belief, medical_reasoning, medical_confidence = (
            self.medical_llm.generate_medical_response(
                case_info=self._case_info,
                dialogue_history=self._history,
                action_prompt=policy_output.action_prompt,
                current_user_utterance=cur_utt,
            )
        )
        self._history.add_turn(
            "medical", medical_response, action=policy_output.action_id,
            belief=medical_belief, reasoning=medical_reasoning, confidence=medical_confidence,
        )

        # r_fmt: did the policy emit a legal, non-empty action for this action_space (any
        # McBurney stage-gate / r_align scoring was retired with the deliberation paradigm --
        # see core/reward_align.py, still importable for old policies but no longer wired here).
        # metadata['control'] (set by policy_ours_v2/action_space_llm_policy) is the control
        # label actually chosen from the action_space; policies with no control/McBurney split
        # (rule/naive/qwen/...) emit it straight as `stage`, so this falls back to that.
        control_label = policy_output.metadata.get("control", policy_output.stage)
        r_fmt = 1.0 if (
            control_label in self.policy.action_space.get("stages", {}) and policy_output.action_prompt
        ) else 0.0
        self._turn_id = t + 1

        # Advance to the next clinician turn (unless the turn cap is hit).
        if self._turn_id >= self._max_turns:
            done = True
            closed_by = "max_turns"
        else:
            next_obs = self._advance_user_turn()
            done = next_obs.done
            # If the user_llm plugin reports a specific termination_reason in its user_state
            # (e.g. v3's "agreement" / "burden_dropout"), use that; otherwise fall back to the
            # old default ("agreement") -- v1/v2's user_state never has this key, so existing
            # callers (GRPO, run_scaling_poc.py) are unaffected.
            reported_reason = (
                next_obs.user_state.model_dump().get("termination_reason")
                if next_obs.user_state else None
            )
            closed_by = reported_reason or ("agreement" if done else None)

        final_judgement = self._finalize() if done else None

        result = StepResult(
            case_id=self._case_info.case_id,
            turn_id=t,
            medical_response=medical_response,
            user_utterance=cur_utt,
            verification_template=cur_verif,
            selected_action=policy_output.action_id,
            action_prompt=policy_output.action_prompt,
            done=done,
            reward=r_fmt,
            metadata={
                "r_fmt": r_fmt,
                "medical_belief": medical_belief,
                "policy": policy_output.metadata,
                "turn_prompts": _tracker.flush_turn(),
                "final_judgement": final_judgement,
                "closed_by": closed_by,
            },
        )
        _tracker.begin_turn()

        if done:
            self._obs = self._obs.model_copy(update={"done": True}) if self._obs else None
        return result

    # ── internals ────────────────────────────────────────────────────────────

    def _advance_user_turn(self) -> Observation:
        user_utterance, user_done, user_state_dict = self.user_llm.generate_user_utterance(
            case_info=self._case_info,
            dialogue_history=self._history,
            turn_id=self._turn_id,
        )
        user_state = UserState(**user_state_dict) if user_state_dict else None
        self._history.add_turn("user", user_utterance, user_state=user_state)
        verification = self.fact_validator_llm.validate_facts(
            case_info=self._case_info,
            dialogue_history=self._history,
            current_user_utterance=user_utterance,
        )
        self._cur_user_utterance = user_utterance
        self._cur_verification = verification
        self._cur_user_state = user_state
        self._user_closed = user_done
        last_action = next(
            (t.action for t in reversed(self._history.turns) if t.speaker == "medical" and t.action),
            None,
        )
        obs = Observation(
            case_info=self._case_info,
            dialogue_history=self._history,
            verification=verification,
            current_user_utterance=user_utterance,
            user_state=user_state,
            last_action=last_action,
            turn_id=self._turn_id,
            done=user_done,
        )
        self._obs = obs
        return obs

    def _finalize(self) -> dict | None:
        if self._history is None or self._case_info is None:
            return None
        # The terminal decision is the clinician persona's OWN last self-reported belief — the
        # option it holds after living through the deliberation — NOT a separate judge re-reading
        # the transcript. This makes is_correct structurally a function of the dialogue the persona
        # actually participated in (a judge could re-answer the case and ignore the conversation).
        belief = next(
            (t.user_state.model_dump().get("belief")
             for t in reversed(self._history.turns)
             if t.speaker == "user" and t.user_state is not None),
            None,
        )
        option = _decision_letter(belief)
        gold = str(self._case_info.correct_option).strip().upper()
        return FinalJudgement(
            concluded_option=option,
            is_correct=(option != "none" and option == gold),
            reason="clinician persona's terminal self-reported decision",
        ).model_dump()

    # ── convenience wrapper (eval / data generation) ──────────────────────────

    def run_episode(
        self,
        case_info: CaseInfo,
        max_turns: int | None = None,
        output_path: str | Path | None = None,
        episode_config: EpisodeConfig | None = None,
    ) -> list[StepResult]:
        model_names = {
            "user_llm": self.user_llm.name(),
            "medical_llm": self.medical_llm.name(),
            "fact_validator_llm": self.fact_validator_llm.name(),
            "policy": self.policy.name(),
        }
        eff_cfg = episode_config or EpisodeConfig()
        logger = RolloutLogger(
            case_info=case_info, model_names=model_names, episode_config=eff_cfg.model_dump(),
        )

        obs = self.reset(case_info, eff_cfg, max_turns=max_turns)
        results: list[StepResult] = []
        while not obs.done:
            kw = {"verification_template": obs.verification} if self.policy.needs_verification else {}
            policy_output = self.policy.select_action(
                case_info=obs.case_info,
                dialogue_history=obs.dialogue_history,
                current_user_utterance=obs.current_user_utterance,
                **kw,
            )
            result = self.step(policy_output)
            results.append(result)
            assert self._history is not None
            logger.log_step(
                result,
                dialogue_snapshot=[t.model_dump() for t in self._history.turns],
            )
            obs = self._obs

        # Attach the episode-level final judgement (computed in step at terminal) to the last record.
        if results:
            fj = results[-1].metadata.get("final_judgement")
            closed_by = results[-1].metadata.get("closed_by")
        else:  # user closed on the very first turn → no actions taken
            fj = self._finalize()
            closed_by = "agreement"
        logger.finalize(fj, closed_by)

        if output_path is not None:
            logger.save(output_path)

        return results
