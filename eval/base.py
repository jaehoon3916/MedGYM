from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    scores: dict[str, Any]
    per_turn: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalPlugin(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(self, turns: list[dict[str, Any]], case_info: dict[str, Any]) -> EvalResult:
        """Evaluate a single episode.

        Args:
            turns: All JSONL records for one episode (one per step, in order).
            case_info: CaseInfo dict from the first record.

        Returns:
            EvalResult with aggregate scores and per-turn annotations.
        """
