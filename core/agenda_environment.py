from __future__ import annotations

from typing import Any

from core.environment import MedicalHACEnvironment
from core.schemas import (
    CaseInfo, EpisodeConfig, Observation, PolicyOutput, StepResult, UserState,
    VerificationTemplate, Agenda, AgendaItem,
)
from plugins.agenda_planner.base import DisagreementAnalyzerPlugin
from plugins.resolution_tracker.base import ResolutionTrackerPlugin
from plugins.user_llm.base import UserLLMPlugin
from plugins.medical_llm.base import MedicalLLMPlugin
from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from plugins.policy.base import PolicyPlugin
from plugins.final_judge_llm.base import FinalJudgeLLMPlugin

# Template for the turn-0 SHARE_AGENDA action.
_SHARE_AGENDA_ACTION_ID = "INFORM.assert"
_SHARE_AGENDA_TEMPLATE = (
    "Share the following agenda with the clinician at the START of your response, "
    "before anything else. Use a warm, collaborative tone — frame it as a structured "
    "plan you and the clinician will work through together, not as a list of their errors.\n\n"
    "Agenda items to share:\n{agenda_text}\n\n"
    "After sharing the agenda, briefly invite the clinician to react to item 1 specifically."
)


def _format_agenda_for_share(agenda: Agenda) -> str:
    lines = []
    for item in agenda.items:
        lines.append(f"  {item.id}. {item.issue}")
        lines.append(f"     Your view: {item.human_position}")
        lines.append(f"     My view:   {item.ai_position}")
    return "\n".join(lines)


class AgendaEnvironment(MedicalHACEnvironment):
    """Subclass of MedicalHACEnvironment that replaces fact-validation with agenda-based
    deliberation. On reset(), an AI-alone solve + Disagreement Analyzer build a 2-3 item
    agenda from the human–AI opinion gap. Each turn the Resolution Tracker checks whether
    the current item is resolved; when all items are resolved (or burden/max_turns hit),
    the episode ends.

    Base env is NOT modified; GRPO / existing configs are unaffected.
    """

    def __init__(
        self,
        user_llm: UserLLMPlugin,
        medical_llm: MedicalLLMPlugin,
        fact_validator_llm: FactValidatorLLMPlugin,  # kept for base constructor; unused internally
        policy: PolicyPlugin,
        config: dict[str, Any],
        final_judge: "FinalJudgeLLMPlugin | None" = None,
        agenda_planner: DisagreementAnalyzerPlugin | None = None,
        resolution_tracker: ResolutionTrackerPlugin | None = None,
    ):
        super().__init__(user_llm, medical_llm, fact_validator_llm, policy, config, final_judge)
        self.agenda_planner = agenda_planner
        self.resolution_tracker = resolution_tracker
        self._agenda: Agenda | None = None
        self._agenda_pointer: int = 0
        self._ai_alone_option: str = ""
        self._ai_alone_rationale: str = ""

    # ── public extras ────────────────────────────────────────────────────────

    def set_ai_alone_result(self, selected_option: str, reasoning: str) -> None:
        """Call before reset() to supply the AI-alone solve result."""
        self._ai_alone_option = selected_option
        self._ai_alone_rationale = reasoning

    @property
    def agenda(self) -> Agenda | None:
        return self._agenda

    # ── override: reset ──────────────────────────────────────────────────────

    def reset(
        self,
        case_info: CaseInfo,
        episode_config: EpisodeConfig | None = None,
        max_turns: int | None = None,
    ) -> Observation:
        self._agenda = None
        self._agenda_pointer = 0
        # Run base reset (calls _advance_user_turn() for turn 0; our override skips
        # fact_validator and leaves _agenda=None so no resolution tracking at turn 0).
        obs = super().reset(case_info, episode_config, max_turns)

        # Build agenda from turn-0 human utterance + AI-alone result.
        if self.agenda_planner is not None:
            human_opinion = obs.current_user_utterance
            ai_opinion = (
                f"I would select option {self._ai_alone_option}. "
                f"{self._ai_alone_rationale}"
            ).strip()
            if not ai_opinion.strip("I would select option ."):
                ai_opinion = "I have an independent assessment of the best management approach."
            self._agenda = self.agenda_planner.analyze(
                case_info=case_info,
                human_opinion=human_opinion,
                ai_opinion=ai_opinion,
            )
        else:
            # No planner configured: one-item fallback so the env still runs.
            self._agenda = Agenda(items=[
                AgendaItem(id=1, issue="Overall clinical approach", status="unresolved")
            ])

        # Update obs with the fresh agenda.
        updated_obs = obs.model_copy(update={
            "agenda": self._agenda,
            "current_item": self._current_agenda_item,
        })
        self._obs = updated_obs
        return updated_obs

    # ── override: _advance_user_turn ─────────────────────────────────────────

    def _advance_user_turn(self) -> Observation:
        """Fact-validator removed; agenda pointer is driven by policy CLOSE in step()."""
        assert self._case_info is not None and self._history is not None, "Call reset() first"
        user_utterance, user_done, user_state_dict = self.user_llm.generate_user_utterance(
            case_info=self._case_info,
            dialogue_history=self._history,
            turn_id=self._turn_id,
        )
        user_state = UserState(**user_state_dict) if user_state_dict else None
        self._history.add_turn("user", user_utterance, user_state=user_state)

        # Agenda-complete safety net (pointer advanced in step() via CLOSE intercept).
        agenda_complete = (
            self._agenda is not None
            and len(self._agenda.items) > 0
            and all(it.status == "resolved" for it in self._agenda.items)
        )
        effective_done = user_done or agenda_complete
        if agenda_complete and not user_done and user_state is not None:
            user_state = user_state.model_copy(
                update={"termination_reason": "agenda_complete"}
            )

        self._cur_user_utterance = user_utterance
        self._cur_verification = VerificationTemplate()  # unused; satisfies base type
        self._cur_user_state = user_state
        self._user_closed = effective_done

        last_action = next(
            (t.action for t in reversed(self._history.turns) if t.speaker == "medical" and t.action),
            None,
        )
        obs = Observation(
            case_info=self._case_info,
            dialogue_history=self._history,
            current_user_utterance=user_utterance,
            user_state=user_state,
            last_action=last_action,
            turn_id=self._turn_id,
            done=effective_done,
            agenda=self._agenda,
            current_item=self._current_agenda_item,
        )
        self._obs = obs
        return obs

    # ── override: step ───────────────────────────────────────────────────────

    def step(self, policy_output) -> StepResult:
        # Turn 0: force SHARE_AGENDA.
        if self._turn_id == 0 and self._agenda is not None:
            agenda_text = _format_agenda_for_share(self._agenda)
            share_prompt = _SHARE_AGENDA_TEMPLATE.format(agenda_text=agenda_text)
            policy_output = PolicyOutput(
                stage="INFORM",
                locution="assert",
                locution_type="goal",
                action_id=_SHARE_AGENDA_ACTION_ID,
                action_prompt=share_prompt,
                confidence=1.0,
                metadata={"policy": "share_agenda_forced"},
            )

        # CLOSE intercept: policy CLOSE = "done with current item".
        # If items remain, mark current resolved, advance pointer, rewrite action so
        # the base env does NOT terminate — episode continues with the next item.
        # Only let CLOSE pass through when all items are resolved (or no agenda set).
        elif (
            policy_output.stage.upper() == "CLOSE"
            and self._agenda is not None
            and self._current_agenda_item is not None  # items still remain
        ):
            self._agenda.items[self._agenda_pointer].status = "resolved"
            self._agenda_pointer += 1
            next_item = self._current_agenda_item  # may be None if last item was just resolved
            if next_item is not None:
                # Rewrite to INFORM so base env continues the episode.
                intro = (
                    f"You have addressed the previous issue. Now transition to agenda item "
                    f"#{next_item.id}: '{next_item.issue}'. Briefly acknowledge the prior "
                    f"discussion and introduce this new focus area."
                )
                policy_output = PolicyOutput(
                    stage="INFORM",
                    locution="assert",
                    locution_type="goal",
                    action_id="INFORM.assert",
                    action_prompt=intro,
                    confidence=1.0,
                    metadata={**policy_output.metadata, "agenda_advanced": True,
                               "resolved_item_id": self._agenda_pointer},
                )
            # else: last item resolved → let CLOSE pass through → base env terminates

        return super().step(policy_output)

    # ── run_episode override (passes current_item to select_action) ──────────

    def run_episode(
        self,
        case_info: CaseInfo,
        max_turns: int | None = None,
        output_path=None,
        episode_config=None,
    ) -> list:
        from core.logger import RolloutLogger
        from pathlib import Path

        model_names = {
            "user_llm": self.user_llm.name(),
            "medical_llm": self.medical_llm.name(),
            "fact_validator_llm": "(agenda-arm: disabled)",
            "policy": self.policy.name(),
        }
        eff_cfg = episode_config or EpisodeConfig()
        logger = RolloutLogger(
            case_info=case_info, model_names=model_names,
            episode_config=eff_cfg.model_dump(),
        )

        obs = self.reset(case_info, eff_cfg, max_turns=max_turns)
        results = []
        while not obs.done:
            kw = {"verification_template": obs.verification} if self.policy.needs_verification else {}
            policy_output = self.policy.select_action(
                case_info=obs.case_info,
                dialogue_history=obs.dialogue_history,
                current_user_utterance=obs.current_user_utterance,
                current_item=obs.current_item,
                **kw,
            )
            result = self.step(policy_output)
            results.append(result)
            assert self._history is not None
            logger.log_step(
                result,
                dialogue_snapshot=[t.model_dump() for t in self._history.turns],
            )
            obs = self._obs  # type: ignore[assignment]

        if results:
            fj = results[-1].metadata.get("final_judgement")
            closed_by = results[-1].metadata.get("closed_by")
        else:
            fj = self._finalize()
            closed_by = "agreement"
        logger.finalize(fj, closed_by)

        if output_path is not None:
            logger.save(Path(output_path))

        return results

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def _current_agenda_item(self) -> AgendaItem | None:
        if self._agenda is None:
            return None
        items = self._agenda.items
        if self._agenda_pointer < len(items):
            return items[self._agenda_pointer]
        return None
