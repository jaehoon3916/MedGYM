import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest
from core.schemas import CaseInfo
from core.environment import MedicalHACEnvironment
from core.config import load_action_space
from plugins.user_llm.mock_user import MockUserLLM
from plugins.medical_llm.mock_medical import MockMedicalLLM
from plugins.fact_validator_llm.mock_fact_validator import MockFactValidatorLLM
from plugins.policy.rule_policy import RulePolicy


@pytest.fixture
def sample_case():
    return CaseInfo(
        case_id="env_test_001",
        scenario="Patient with chronic cough and heavy smoking history.",
        options={"A": "COPD", "B": "Asthma"},
        correct_option="A",
        answer="COPD",
        distractors=["Asthma"],
    )


@pytest.fixture
def env():
    action_space = load_action_space()
    config = {"experiment": {"max_turns": 2}}
    return MedicalHACEnvironment(
        user_llm=MockUserLLM({}),
        medical_llm=MockMedicalLLM({}),
        fact_validator_llm=MockFactValidatorLLM({}),
        policy=RulePolicy({}, action_space=action_space),
        config=config,
    )


def test_reset_returns_empty_history(env, sample_case):
    history = env.reset(sample_case)
    assert history.case_id == sample_case.case_id
    assert len(history.turns) == 0


def test_single_step(env, sample_case):
    env.reset(sample_case)
    result = env.step()
    assert result.turn_id == 0
    assert result.medical_response
    assert result.user_utterance
    assert "." in result.selected_action  # "STAGE.locution" format
    assert result.action_prompt
    assert result.verification_template is not None


def test_step_updates_history(env, sample_case):
    env.reset(sample_case)
    env.step()
    # After 1 step: user utterance added first, then medical response
    assert len(env._history.turns) == 2
    assert env._history.turns[0].speaker == "user"
    assert env._history.turns[1].speaker == "medical"


def test_run_episode_two_turns(env, sample_case, tmp_path):
    output_path = tmp_path / "rollout.jsonl"
    results = env.run_episode(sample_case, max_turns=2, output_path=str(output_path))
    assert len(results) == 2
    assert output_path.exists()


def test_rollout_jsonl_format(env, sample_case, tmp_path):
    output_path = tmp_path / "rollout.jsonl"
    env.run_episode(sample_case, max_turns=2, output_path=str(output_path))

    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == 2

    record = json.loads(lines[0])
    required_keys = {
        "case_id", "turn_id", "case_info", "dialogue_history",
        "verification_template", "selected_action", "action_prompt",
        "medical_response", "user_utterance", "reward",
        "model_name", "timestamp",
    }
    assert required_keys.issubset(record.keys())


def test_turn_ids_increment(env, sample_case):
    results = env.run_episode(sample_case, max_turns=3)
    assert [r.turn_id for r in results] == [0, 1, 2]
