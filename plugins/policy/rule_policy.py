from typing import Any

from plugins.policy.base import PolicyPlugin
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput


_RELATION_TO_ACTION: dict[str, tuple[str, str, str]] = {
    # (stage, locution, locution_type)
    # contradicted/supported corrected to match the best action under r_align.txt's BASE
    # table that is SAFE without ctx (i.e. not gated behind a hard precondition this
    # context-blind policy can't check) -- INFORM.assert/RECOMMEND.assert previously
    # scored 0.0 despite a precondition-free 0.7-1.0 action existing for both. Note
    # RECOMMEND.move/CONFIRM.assert score -1.0 without a prior has_proposal, so they are
    # NOT safe fixed choices here even though they top the BASE table when that holds.
    "contradicted":  ("CONSIDER",   "assert",       "evaluation"),
    "insufficient":  ("INFORM",     "ask_justify",  "fact"),
    "supported":     ("INFORM",     "assert",       "fact"),
    "mixed":         ("CONSIDER",   "assert",       "evaluation"),
    "unknown":       ("INFORM",     "ask_justify",  "fact"),
}

_ACTION_PROMPTS: dict[str, str] = {
    "INFORM.assert":            "Assert the correct clinical fact based on established medical evidence.",
    "INFORM.ask_justify":       "Ask the clinician to justify or provide evidence for their claim.",
    "RECOMMEND.assert":         "Assert and recommend the clinically supported action-option.",
    "CONSIDER.assert":          "Evaluate and comment on the proposal from a clinical perspective.",
    "INFORM.retract":           "Retract a previously stated clinical claim in light of new evidence.",
    "PROPOSE.propose":          "Propose a specific management or diagnostic action-option.",
    "REVISE.propose":           "Revise the previously proposed action-option based on the discussion.",
    "REVISE.retract":           "Retract and revise a previous position.",
    "REVISE.ask_justify":       "Ask for justification before revising the clinical plan.",
    "RECOMMEND.move":           "Call for collective agreement on the recommended action.",
    "RECOMMEND.reject":         "Reject the proposed action as clinically inappropriate.",
    "CONFIRM.assert":           "Confirm collective acceptance of the recommended action.",
    "CLOSE.withdraw_dialogue":  "Close the deliberation dialogue.",
}


class RulePolicy(PolicyPlugin):
    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        super().__init__(config, action_space)

    def name(self) -> str:
        return "rule-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        relation = verification_template.overall_relation
        stage, locution, locution_type = _RELATION_TO_ACTION.get(
            relation, ("INFORM", "ask_justify", "fact")
        )
        action_id = f"{stage}.{locution}"
        action_prompt = _ACTION_PROMPTS.get(action_id, f"Apply {locution} in the {stage} stage.")

        return PolicyOutput(
            stage=stage,
            locution=locution,
            locution_type=locution_type,
            action_id=action_id,
            action_prompt=action_prompt,
            confidence=1.0,
            metadata={"policy": "rule", "overall_relation": relation},
        )
