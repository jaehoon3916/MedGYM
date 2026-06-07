import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate
from plugins.user_llm.mock_user import MockUserLLM
from plugins.medical_llm.mock_medical import MockMedicalLLM
from plugins.fact_validator_llm.mock_fact_validator import MockFactValidatorLLM
from plugins.policy.rule_policy import RulePolicy
from core.config import load_action_space


@pytest.fixture
def case_info():
    return CaseInfo(
        case_id="test_001",
        scenario="Patient with shortness of breath and 40 pack-year smoking history.",
        options={"A": "COPD", "B": "Asthma", "C": "Fibrosis", "D": "CHF"},
        correct_option="A",
        answer="COPD",
        distractors=["Asthma", "Fibrosis", "CHF"],
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


def test_mock_fact_validator(case_info, empty_history):
    plugin = MockFactValidatorLLM({})
    plugin.load()
    vt = plugin.validate_facts(case_info, empty_history, "I think the diagnosis is COPD.")
    assert isinstance(vt, VerificationTemplate)
    assert vt.overall_relation == "insufficient"
    assert vt.confidence == "low"


def test_rule_policy_contradicted_asserts_fact(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="contradicted", confidence="high")
    output = policy.select_action(case_info, empty_history, "wrong claim", vt)
    assert output.stage == "INFORM"
    assert output.locution == "assert"
    assert output.action_id == "INFORM.assert"


def test_rule_policy_insufficient_asks_justify(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="insufficient", confidence="low")
    output = policy.select_action(case_info, empty_history, "unclear claim", vt)
    assert output.stage == "INFORM"
    assert output.locution == "ask_justify"
    assert output.action_id == "INFORM.ask_justify"


def test_rule_policy_supported_recommends(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="supported", confidence="high")
    output = policy.select_action(case_info, empty_history, "correct claim", vt)
    assert output.stage == "RECOMMEND"
    assert output.locution == "assert"
    assert output.action_id == "RECOMMEND.assert"


def test_rule_policy_output_has_prompt(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate()
    output = policy.select_action(case_info, empty_history, "some claim", vt)
    assert len(output.action_prompt) > 0
    assert output.stage and output.locution
