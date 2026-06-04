from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class CaseInfo(BaseModel):
    case_id: str
    scenario: str
    question: str
    options: dict[str, str]
    correct_answer: str | None = None
    expert_explanation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MedicalClaim(BaseModel):
    claim: str
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class UserNoise(BaseModel):
    irrelevant_info: bool = False
    contradiction: bool = False
    emotional_expression: bool = False


class UserState(BaseModel):
    intent: Literal[
        "accept", "challenge", "ask_question",
        "express_uncertainty", "provide_evidence", "other"
    ] = "other"
    certainty: Literal["certain", "uncertain", "neutral"] = "neutral"
    stance_toward_medical_llm: Literal[
        "agree", "disagree", "skeptical", "confused", "unknown"
    ] = "unknown"
    medical_claims: list[MedicalClaim] = Field(default_factory=list)
    user_noise: UserNoise = Field(default_factory=UserNoise)
    key_facts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
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
