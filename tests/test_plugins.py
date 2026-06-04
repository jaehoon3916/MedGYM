import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.schemas import CaseInfo, DialogueHistory, UserState
from plugins.user_llm.mock_user import MockUserLLM
from plugins.medical_llm.mock_medical import MockMedicalLLM
from plugins.extractor_llm.mock_extractor import MockExtractorLLM
from plugins.extractor_llm.rule_extractor import RuleExtractorLLM
from plugins.policy.rule_policy import RulePolicy
from core.config import load_action_space


@pytest.fixture
def case_info():
    return CaseInfo(
        case_id="test_001",
        scenario="Patient with shortness of breath and 40 pack-year smoking history.",
        question="What is the most likely diagnosis?",
        options={"A": "COPD", "B": "Asthma", "C": "Fibrosis", "D": "CHF"},
        correct_answer="A",
    )


@pytest.fixture
def empty_history(case_info):
    return DialogueHistory(case_id=case_info.case_id)


@pytest.fixture
def action_space():
    return load_action_space()


def test_mock_user_llm(case_info, empty_history):
    plugin = MockUserLLM({})
    plugin.load()
    response = plugin.generate_user_utterance(case_info, empty_history, "Hello")
    assert isinstance(response, str) and len(response) > 0


def test_mock_user_llm_cycles(case_info, empty_history):
    plugin = MockUserLLM({})
    plugin.load()
    responses = [plugin.generate_user_utterance(case_info, empty_history, "x") for _ in range(10)]
    assert len(set(responses)) > 1  # cycles through multiple responses


def test_mock_medical_llm_accept(case_info, empty_history):
    plugin = MockMedicalLLM({})
    plugin.load()
    response = plugin.generate_medical_response(case_info, empty_history, "ACCEPT the hypothesis")
    assert "agree" in response.lower() or "accept" in response.lower()


def test_mock_medical_llm_challenge(case_info, empty_history):
    plugin = MockMedicalLLM({})
    plugin.load()
    response = plugin.generate_medical_response(case_info, empty_history, "CHALLENGE the hypothesis")
    assert "challenge" in response.lower()


def test_mock_extractor(case_info, empty_history):
    plugin = MockExtractorLLM({})
    plugin.load()
    state = plugin.extract_user_state(case_info, empty_history)
    assert isinstance(state, UserState)
    assert state.certainty == "uncertain"


def test_rule_extractor_uncertain(case_info, empty_history):
    plugin = RuleExtractorLLM({})
    plugin.load()
    empty_history.add_turn("user", "I'm not sure what the diagnosis could be.")
    state = plugin.extract_user_state(case_info, empty_history)
    assert state.certainty == "uncertain"


def test_rule_extractor_agree(case_info, empty_history):
    plugin = RuleExtractorLLM({})
    plugin.load()
    empty_history.add_turn("user", "I agree with your assessment.")
    state = plugin.extract_user_state(case_info, empty_history)
    assert state.stance_toward_medical_llm == "agree"


def test_rule_policy_uncertain_asks_evidence(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    user_state = UserState(certainty="uncertain", summary="uncertain")
    output = policy.select_action(case_info, empty_history, user_state)
    assert output.action_id == "ASK_EVIDENCE"


def test_rule_policy_skeptical_summarizes(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    user_state = UserState(
        certainty="certain",
        stance_toward_medical_llm="skeptical",
        summary="skeptical",
    )
    output = policy.select_action(case_info, empty_history, user_state)
    assert output.action_id == "SUMMARIZE"


def test_rule_policy_output_has_prompt(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    user_state = UserState(summary="test")
    output = policy.select_action(case_info, empty_history, user_state)
    assert len(output.action_prompt) > 0
