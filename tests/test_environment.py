"""Unit tests for core/environment.py's closed_by termination-reason logic -- zero LLM calls
(stub plugins). See /home/kjy/.claude/plans/baseline-snug-acorn.md: v3's user_state can carry
a "termination_reason" key ("agreement" | "burden_dropout"); environment.py's step() should
promote that into closed_by, falling back to the old "agreement" default when the user_llm
plugin doesn't report one (v1/v2 compatibility -- their user_state never has this key).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo, EpisodeConfig, PolicyOutput, VerificationTemplate
from plugins.fact_validator_llm.base import FactValidatorLLMPlugin
from plugins.medical_llm.base import MedicalLLMPlugin
from plugins.policy.base import PolicyPlugin
from plugins.user_llm.base import UserLLMPlugin


class _StubUserLLM(UserLLMPlugin):
    def __init__(self, done: bool, user_state_dict: dict | None):
        self._done = done
        self._user_state_dict = user_state_dict

    def name(self) -> str:
        return "stub-user-llm"

    def load(self) -> None:
        pass

    def generate_user_utterance(self, case_info, dialogue_history, turn_id=0, user_profile=None):
        return "doctor says something", self._done, self._user_state_dict


class _StubMedicalLLM(MedicalLLMPlugin):
    def name(self) -> str:
        return "stub-medical-llm"

    def load(self) -> None:
        pass

    def generate_medical_response(self, case_info, dialogue_history, action_prompt, current_user_utterance):
        return "ai says something", None, None


class _StubFactValidator(FactValidatorLLMPlugin):
    def name(self) -> str:
        return "stub-fact-validator"

    def load(self) -> None:
        pass

    def validate_facts(self, case_info, dialogue_history, current_user_utterance):
        return VerificationTemplate()


class _StubPolicy(PolicyPlugin):
    def name(self) -> str:
        return "stub-policy"

    def load(self) -> None:
        pass

    def select_action(self, case_info, dialogue_history, current_user_utterance, verification_template):
        return PolicyOutput(stage="INFORM", locution="assert", locution_type="fact", action_id="INFORM.assert", action_prompt="x")


def _case_info() -> CaseInfo:
    return CaseInfo(case_id="t", scenario="s", options={}, correct_option="A", answer="A")


def _make_env(user_llm: UserLLMPlugin) -> MedicalHACEnvironment:
    return MedicalHACEnvironment(
        user_llm=user_llm, medical_llm=_StubMedicalLLM({}), fact_validator_llm=_StubFactValidator({}),
        policy=_StubPolicy({}, action_space={}), config={"experiment": {"max_turns": 8}},
    )


def _policy_output() -> PolicyOutput:
    return PolicyOutput(stage="INFORM", locution="assert", locution_type="fact", action_id="INFORM.assert", action_prompt="x")


def test_closed_by_uses_reported_termination_reason_burden_dropout():
    env = _make_env(_StubUserLLM(done=True, user_state_dict={"termination_reason": "burden_dropout"}))
    env.reset(_case_info(), EpisodeConfig(), max_turns=8)
    result = env.step(_policy_output())
    assert result.done is True
    assert result.metadata["closed_by"] == "burden_dropout"


def test_closed_by_uses_reported_termination_reason_agreement():
    env = _make_env(_StubUserLLM(done=True, user_state_dict={"termination_reason": "agreement"}))
    env.reset(_case_info(), EpisodeConfig(), max_turns=8)
    result = env.step(_policy_output())
    assert result.done is True
    assert result.metadata["closed_by"] == "agreement"


def test_closed_by_falls_back_to_agreement_when_no_termination_reason_v1_compat():
    # v1/v2-style user_state: done=True but no "termination_reason" key at all.
    env = _make_env(_StubUserLLM(done=True, user_state_dict={"belief": "A"}))
    env.reset(_case_info(), EpisodeConfig(), max_turns=8)
    result = env.step(_policy_output())
    assert result.done is True
    assert result.metadata["closed_by"] == "agreement"


def test_closed_by_is_max_turns_when_turn_cap_hit_regardless_of_user_state():
    env = _make_env(_StubUserLLM(done=False, user_state_dict={"termination_reason": None}))
    env.reset(_case_info(), EpisodeConfig(), max_turns=1)
    result = env.step(_policy_output())
    assert result.done is True
    assert result.metadata["closed_by"] == "max_turns"


def test_closed_by_is_none_when_not_done():
    env = _make_env(_StubUserLLM(done=False, user_state_dict={"termination_reason": None}))
    env.reset(_case_info(), EpisodeConfig(), max_turns=8)
    result = env.step(_policy_output())
    assert result.done is False
    assert result.metadata["closed_by"] is None
