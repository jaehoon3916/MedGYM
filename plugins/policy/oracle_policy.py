"""Reward-oracle policy — picks argmax_a r_align. Used as the SFT teacher (and a rule-grounded baseline).

It does NOT need a model: given the fact-validation + trajectory context, it selects the legal action
that maximizes r_align. The student policy is later trained to imitate it (SFT).
"""
from __future__ import annotations

from typing import Any

from plugins.policy.base import PolicyPlugin
from plugins.policy.qwen_policy import _build_policy_output
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate, PolicyOutput
from core.reward_align import ctx_from_history, best_action


class RewardOraclePolicy(PolicyPlugin):
    def __init__(self, config: dict[str, Any], action_space: dict[str, Any]):
        PolicyPlugin.__init__(self, config, action_space)
        self._mode = config.get("mode", "oracle")

    def name(self) -> str:
        return "reward-oracle-policy"

    def load(self) -> None:
        pass

    def select_action(
        self,
        case_info: CaseInfo,
        dialogue_history: DialogueHistory,
        current_user_utterance: str,
        verification_template: VerificationTemplate,
    ) -> PolicyOutput:
        ctx = ctx_from_history(dialogue_history)
        user_loc = None
        last_user = next(
            (t for t in reversed(dialogue_history.turns) if t.speaker == "user"), None
        )
        if last_user is not None and last_user.user_state is not None:
            user_loc = last_user.user_state.model_dump().get("locution")
        (stage, locution), score = best_action(
            verification_template.overall_relation, user_loc, self.action_space, ctx
        )
        po = _build_policy_output(stage, locution, "fact", self._mode, "", self.action_space)
        po.metadata["r_align"] = score
        return po
