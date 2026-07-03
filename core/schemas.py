from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class CaseInfo(BaseModel):
    """JAMA Clinical Challenge case schema."""
    case_id: str
    scenario: str
    options: dict[str, str]                  # {"A": "...", "B": "...", ...}
    correct_option: str                       # e.g. "D"
    answer: str                              # full text of the correct option
    distractors: list[str] = Field(default_factory=list)  # incorrect option texts
    caption: str | None = None               # image/figure caption
    explanation: str | None = None           # expert explanation
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def question(self) -> str:
        """JAMA cases don't have an explicit question field; derive from context."""
        return "What is the most appropriate next step in management?"

    @property
    def correct_answer(self) -> str:
        return self.answer


class EpisodeConfig(BaseModel):
    """Per-episode persona/condition spec, fed to a user_llm plugin's reset_episode().

    v1 (plugins/user_llm/user_simulator/v1.py) ignores every field, including
    `initial_fact` — its belief is self-determined (judged independently on turn 0)
    rather than injected. v2 (plugins/user_llm/user_simulator/v2.py) reads every field.
    """
    initial_fact: Literal["correct", "incorrect"] = "incorrect"
    certainty: Literal["certain", "uncertain", "neutral"] = "uncertain"
    authority_push: Literal["high", "low"] = "low"
    information_sparcity: Literal["dense", "sparse"] = "dense"
    safety_push: Literal["true", "false"] = "false"
    # v4 (plugins/user_llm/user_simulator/v4.py) only: overrides the plugin's default persona
    # for this episode (re-derives c0/lambda/w/rho/kappa/burden_dropout_threshold). None = keep
    # whatever persona the plugin was constructed with.
    persona: str | None = None


class UserState(BaseModel):
    """Fields and allowed values are defined in configs/user_state.yaml."""
    model_config = ConfigDict(extra="allow")
    summary: str = ""


class FinalJudgement(BaseModel):
    """End-of-episode verdict: which option the consultation converged on, vs the gold option."""
    concluded_option: str = "none"   # "A" | "B" | "C" | "D" | "none"
    is_correct: bool = False         # set by the environment: concluded_option == gold correct_option
    reason: str = ""


class VerificationTemplate(BaseModel):
    """NLI-based fact validation output from the Fact Validator LLM."""
    overall_relation: Literal["supported", "contradicted", "insufficient", "mixed", "unknown"] = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_gaps: list[str] = Field(default_factory=list)
    # Substantive distilled clinical reasoning (the medical knowledge handed to the policy as environment).
    reasoning: str = ""
    optional_claim_checks: list[str] = Field(default_factory=list)


class AgendaItem(BaseModel):
    """One issue where the clinician's opinion and the AI-alone opinion diverge.

    Built by the Disagreement Analyzer (plugins/agenda_planner) at turn 0. NOTE the analyzer
    must NOT label either side correct -- it only names WHERE they differ. `status` is advanced
    to "resolved" by the Resolution Tracker (plugins/resolution_tracker) during the dialogue."""
    id: int
    issue: str                       # the point of divergence, phrased neutrally
    human_position: str = ""         # what the clinician holds on this issue
    ai_position: str = ""            # what the AI-alone opinion holds on this issue
    status: Literal["unresolved", "resolved"] = "unresolved"


class Agenda(BaseModel):
    items: list[AgendaItem] = Field(default_factory=list)

    @property
    def unresolved(self) -> list[AgendaItem]:
        return [it for it in self.items if it.status == "unresolved"]


class DialogueTurn(BaseModel):
    speaker: Literal["medical", "user"]
    text: str
    action: str | None = None
    user_state: UserState | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DialogueHistory(BaseModel):
    case_id: str
    turns: list[DialogueTurn] = Field(default_factory=list)

    def add_turn(
        self,
        speaker: Literal["medical", "user"],
        text: str,
        action: str | None = None,
        user_state: UserState | None = None,
        **metadata: Any,
    ) -> None:
        self.turns.append(
            DialogueTurn(
                speaker=speaker,
                text=text,
                action=action,
                user_state=user_state,
                metadata=metadata,
            )
        )

    def to_prompt(self) -> str:
        lines = []
        for turn in self.turns:
            prefix = "Medical" if turn.speaker == "medical" else "User"
            lines.append(f"{prefix}: {turn.text}")
        return "\n".join(lines)

    def to_prompt_with_actions(self) -> str:
        """Like to_prompt(), but annotates each medical turn with the deliberation action that
        produced it (e.g. 'Medical [INFORM.ask_justify]: ...'). For planning/deliberation
        policies so the policy LLM can see its OWN past tactics and avoid repeating an
        ineffective move -- to_prompt() drops turn.action, hiding the tactic sequence (this is
        why a policy can loop on the same ask_justify turn after turn). Deliberately NOT used in
        medical_llm / fact_validator / final_judge prompts -- those must not see internal
        action labels."""
        lines = []
        for turn in self.turns:
            if turn.speaker == "medical":
                tag = f" [{turn.action}]" if turn.action else ""
                lines.append(f"Medical{tag}: {turn.text}")
            else:
                lines.append(f"User: {turn.text}")
        return "\n".join(lines)

    def to_messages(self) -> list[dict[str, str]]:
        role_map = {"medical": "assistant", "user": "user"}
        return [
            {"role": role_map[t.speaker], "content": t.text}
            for t in self.turns
        ]


class PolicyOutput(BaseModel):
    stage: str       # e.g. "INFORM"
    locution: str    # e.g. "ask_justify"
    locution_type: str  # e.g. "fact"
    action_id: str   # composite: "INFORM.ask_justify"
    action_prompt: str
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    case_id: str
    turn_id: int
    medical_response: str
    user_utterance: str
    verification_template: VerificationTemplate
    selected_action: str
    action_prompt: str
    done: bool = False
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Agent-external env observation: exactly the context the policy consumes each turn."""
    case_info: CaseInfo
    dialogue_history: DialogueHistory
    verification: VerificationTemplate = Field(default_factory=VerificationTemplate)
    current_user_utterance: str
    user_state: UserState | None = None
    last_action: str | None = None
    turn_id: int = 0
    done: bool = False
    # Agenda arm (D) extensions — None in base env; populated by AgendaEnvironment.
    agenda: Agenda | None = None
    current_item: AgendaItem | None = None
