import json
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate
from plugins.policy.rule_policy import RulePolicy
from plugins.policy.medcobe_naive_policy import MedCobeNaivePolicy
from plugins.policy.react_policy import ReactPolicy
from plugins.policy.medcobe_feedback_policy import MedCobeFeedbackPolicy
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


def test_rule_policy_contradicted_considers(case_info, empty_history, action_space):  # noqa
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="contradicted", confidence="high")
    output = policy.select_action(case_info, empty_history, "wrong claim", vt)
    assert output.stage == "CONSIDER"
    assert output.locution == "assert"
    assert output.action_id == "CONSIDER.assert"


def test_rule_policy_insufficient_asks_justify(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="insufficient", confidence="low")
    output = policy.select_action(case_info, empty_history, "unclear claim", vt)
    assert output.stage == "INFORM"
    assert output.locution == "ask_justify"
    assert output.action_id == "INFORM.ask_justify"


def test_rule_policy_supported_asserts_fact(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="supported", confidence="high")
    output = policy.select_action(case_info, empty_history, "correct claim", vt)
    assert output.stage == "INFORM"
    assert output.locution == "assert"
    assert output.action_id == "INFORM.assert"


def test_rule_policy_output_has_prompt(case_info, empty_history, action_space):
    policy = RulePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate()
    output = policy.select_action(case_info, empty_history, "some claim", vt)
    assert len(output.action_prompt) > 0
    assert output.stage and output.locution


def test_medcobe_naive_policy_is_blind_to_verification_template(case_info, empty_history, action_space):
    policy = MedCobeNaivePolicy({}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="contradicted", confidence="high", reasoning="some secret reasoning")
    output = policy.select_action(case_info, empty_history, "wrong claim", vt)
    assert output.action_id == "INFORM.assert"
    for leaked in ("contradicted", "high", "some secret reasoning"):
        assert leaked not in output.action_prompt


def test_react_policy_uses_chosen_action(case_info, empty_history, action_space):
    policy = ReactPolicy({"model": "test-model"}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate(overall_relation="supported", confidence="high", reasoning="secret")
    raw = json.dumps({"thought": "doctor seems off", "action": "argue", "reason": "evidence mismatch"})
    with patch.object(policy, "_chat", return_value=raw):
        output = policy.select_action(case_info, empty_history, "claim", vt)
    assert output.action_id == "INFORM.assert"
    assert output.metadata["react_action"] == "ARGUE"
    assert "ARGUE" in output.action_prompt
    assert "supported" not in output.action_prompt  # blind to verification_template


def test_react_policy_falls_back_on_bad_json(case_info, empty_history, action_space):
    policy = ReactPolicy({"model": "test-model"}, action_space=action_space)
    policy.load()
    vt = VerificationTemplate()
    with patch.object(policy, "_chat", return_value="not json"):
        output = policy.select_action(case_info, empty_history, "claim", vt)
    assert output.metadata["react_action"] == "DEFER"


def test_medcobe_feedback_policy_falls_back_without_calibration(case_info, empty_history, action_space, tmp_path):
    policy = MedCobeFeedbackPolicy(
        {"target_model": "no-such-model", "calibration_dir": str(tmp_path)}, action_space=action_space,
    )
    policy.load()
    vt = VerificationTemplate()
    output = policy.select_action(case_info, empty_history, "claim", vt)
    assert output.metadata["sycophancy"] == 0.0
    assert output.metadata["obstruction"] == 0.0
    assert "[Behavior Guideline]" in output.action_prompt


def test_medcobe_feedback_policy_loads_calibration_file(case_info, empty_history, action_space, tmp_path):
    (tmp_path / "calibrated-model.json").write_text(json.dumps({
        "model": "calibrated-model", "sycophancy": 0.5, "obstruction": 0.0, "stats": {},
        "behavior_guideline": "ignored -- regenerated from sycophancy/obstruction, not cached text",
    }))
    policy = MedCobeFeedbackPolicy(
        {"target_model": "calibrated-model", "calibration_dir": str(tmp_path)}, action_space=action_space,
    )
    policy.load()
    vt = VerificationTemplate()
    output = policy.select_action(case_info, empty_history, "claim", vt)
    assert output.metadata["sycophancy"] == 0.5
    assert "tendency to accept clinician beliefs" in output.action_prompt
