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


class UserState(BaseModel):
    """Fields and allowed values are defined in configs/user_state.yaml."""
    model_config = ConfigDict(extra="allow")
    summary: str = ""


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

    def to_messages(self) -> list[dict[str, str]]:
        role_map = {"medical": "assistant", "user": "user"}
        return [
            {"role": role_map[t.speaker], "content": t.text}
            for t in self.turns
        ]


class PolicyOutput(BaseModel):
    action_id: Literal["ACCEPT", "CHALLENGE", "ASK_EVIDENCE", "DEFER", "SUMMARIZE"]
    action_prompt: str
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    case_id: str
    turn_id: int
    medical_response: str
    user_utterance: str
    user_state: UserState
    selected_action: str
    action_prompt: str
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
