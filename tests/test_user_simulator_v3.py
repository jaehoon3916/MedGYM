"""Pure-function and wiring tests for UserSimulatorV3 -- zero LLM calls.

See /home/kjy/.claude/plans/wild-tumbling-glade.md section 7 for the rationale behind each
test (recomputing overall_workload rather than trusting the model's own arithmetic,
all-or-nothing parsing, the fallback's midpoint invariant, placeholder round-tripping).
"""
import sys
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import yaml

from core.config import build_plugins
from core.schemas import CaseInfo, DialogueHistory, DialogueTurn, EpisodeConfig
from plugins.user_llm.user_simulator.v3 import (
    UserSimulatorV3, _load_prompts, _parse_opening, _parse_followup_no_options,
)
from plugins.user_llm.user_simulator.v3_burden import (
    _TLX_DIMS,
    _burden_judge_fallback,
    _coerce_tlx_score,
    _format_burden_breakdown,
    _normalize_overall_workload,
    _parse_burden_judge,
)


# ── _normalize_overall_workload ─────────────────────────────────────────────────────────

def test_normalize_overall_workload_endpoints():
    assert _normalize_overall_workload(1.0) == 0.0
    assert _normalize_overall_workload(5.0) == 1.0
    assert _normalize_overall_workload(3.0) == 0.5


def test_normalize_overall_workload_monotonic():
    vals = [_normalize_overall_workload(x) for x in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert vals == sorted(vals)
    assert len(set(vals)) == len(vals)


# ── _coerce_tlx_score ────────────────────────────────────────────────────────────────────

def test_coerce_tlx_score_in_range_passthrough():
    assert _coerce_tlx_score(1) == 1
    assert _coerce_tlx_score(5) == 5
    assert _coerce_tlx_score(3) == 3


def test_coerce_tlx_score_clamps_out_of_range():
    assert _coerce_tlx_score(7) == 5
    assert _coerce_tlx_score(0) == 1
    assert _coerce_tlx_score(-3) == 1


def test_coerce_tlx_score_rounds_fractional():
    assert _coerce_tlx_score(3.6) == 4


def test_coerce_tlx_score_rejects_non_numeric():
    assert _coerce_tlx_score(None) is None
    assert _coerce_tlx_score("n/a") is None
    assert _coerce_tlx_score([]) is None


# ── _parse_burden_judge ──────────────────────────────────────────────────────────────────

def _valid_tlx_json(**overrides) -> str:
    data = {
        "mental_demand": 3, "performance": 4, "effort": 3, "frustration": 4,
        "overall_workload": 999.9,  # deliberately wrong -- code must recompute, not trust this
        "short_rationale": {
            "mental_demand": "a", "performance": "c", "effort": "d", "frustration": "e",
        },
    }
    data.update(overrides)
    return json.dumps(data)


def test_parse_burden_judge_recomputes_overall_workload_not_trusting_llm():
    parsed = _parse_burden_judge(_valid_tlx_json())
    assert parsed is not None
    expected = round((3 + 4 + 3 + 4) / 4, 1)
    assert parsed["overall_workload"] == expected
    assert parsed["overall_workload"] != 999.9
    assert "physical_demand" not in parsed
    assert "temporal_demand" not in parsed
    for dim in _TLX_DIMS:
        assert dim in parsed


def test_parse_burden_judge_all_or_nothing_on_missing_dimension():
    data = json.loads(_valid_tlx_json())
    del data["frustration"]
    assert _parse_burden_judge(json.dumps(data)) is None


def test_parse_burden_judge_rejects_empty_string():
    assert _parse_burden_judge("") is None


# ── _burden_judge_fallback ───────────────────────────────────────────────────────────────

def test_burden_judge_fallback_midpoint_invariant():
    fb = _burden_judge_fallback()
    assert all(fb[dim] == 3 for dim in _TLX_DIMS)
    assert fb["overall_workload"] == 3.0
    assert "physical_demand" not in fb
    assert "temporal_demand" not in fb
    assert _normalize_overall_workload(fb["overall_workload"]) == 0.5


# ── _format_burden_breakdown ─────────────────────────────────────────────────────────────

def test_format_burden_breakdown_dominant_driver():
    tlx = {"mental_demand": 2, "performance": 2, "effort": 2,
           "frustration": 5, "overall_workload": 2.75}
    line = _format_burden_breakdown(tlx)
    assert "Dominant driver: frustration" in line


def test_format_burden_breakdown_tie_break_uses_dims_order():
    # mental_demand and effort tied at 4 -- mental_demand is earlier in _TLX_DIMS, must win.
    tlx = {"mental_demand": 4, "performance": 2, "effort": 4,
           "frustration": 2, "overall_workload": 3.0}
    line = _format_burden_breakdown(tlx)
    assert "Dominant driver: mental_demand" in line


# ── Prompt template loading / placeholder round-trip ────────────────────────────────────

_ALL_TEMPLATE_KEYS = (
    "burden_judge_system", "burden_judge_user",
    "utterance_turn0_system", "utterance_turn0_user",
    "utterance_turn0_system_no_options", "utterance_turn0_user_no_options",
    "utterance_followup_system", "utterance_followup_user",
    "utterance_followup_system_no_options", "utterance_followup_user_no_options",
)


def test_load_prompts_has_all_referenced_keys():
    tmpl = _load_prompts()
    for key in _ALL_TEMPLATE_KEYS:
        assert key in tmpl, f"missing template key: {key}"


def test_all_templates_format_without_keyerror():
    tmpl = _load_prompts()
    ctx = dict(
        scenario="s", image_caption="c", question="q", options_text="A. x\nB. y",
        dialogue_text="d", persona_instruction="p", sparsity_instruction="sp",
        cumulative_burden=0.42, burden_breakdown_line="mental_demand=3 (...)",
        forced_withdrawal_line="", ai_utterance="ai said something",
    )
    for key in _ALL_TEMPLATE_KEYS:
        tmpl[key].format(**ctx)  # raises KeyError on a typo'd/missing placeholder


# ── core.config wiring ───────────────────────────────────────────────────────────────────

def test_build_plugins_resolves_v3_without_network_call():
    config = {
        "experiment": {"ablation_mode": "naive"},
        "plugins": {
            "user_llm": {"type": "v3", "base_url": "http://localhost:8001/v1", "model": "stub-model"},
            "medical_llm": {"type": "vllm", "base_url": "http://localhost:8001/v1", "model": "stub-model"},
            "fact_validator_llm": {"type": "vllm", "base_url": "http://localhost:8001/v1", "model": "stub-model"},
            "final_judge": {"enabled": False},
            "policy": {"type": "naive"},
        },
    }
    user_llm, _medical_llm, _fact_validator_llm, _policy, final_judge = build_plugins(config)
    assert isinstance(user_llm, UserSimulatorV3)
    assert user_llm._persona == "burned_out_resident"
    assert final_judge is None


def test_user_simulator_v3_name():
    sim = UserSimulatorV3({"model": "stub-model"})
    assert "user-simulator-v3-" in sim.name()


# ── _followup_turn: termination_reason / burden-threshold dropout ──────────────────────

def _burden_json(level: int) -> str:
    return json.dumps({
        "mental_demand": level, "performance": level, "effort": level, "frustration": level,
        "short_rationale": {d: "x" for d in ("mental_demand", "performance", "effort", "frustration")},
    })


def _followup_json(decision: str) -> str:
    return json.dumps({
        "response": "ok", "decision": decision, "confidence": 0.5, "argument_validity": "v",
    })


def _one_turn_history() -> DialogueHistory:
    history = DialogueHistory(case_id="t")
    history.turns.append(DialogueTurn(speaker="user", text="doctor opening"))
    history.turns.append(DialogueTurn(speaker="medical", text="ai turn"))
    return history


def _case_info() -> CaseInfo:
    return CaseInfo(case_id="t", scenario="s", options={}, correct_option="A", answer="A")


def _fake_chat_factory(burden_level: int, decision: str):
    def fake_chat(messages, temperature=None, response_format=None):
        if temperature == 0.0:
            return _burden_json(burden_level)
        return _followup_json(decision)
    return fake_chat


def test_followup_turn_burden_dropout_sets_termination_reason():
    sim = UserSimulatorV3({
        "model": "stub-model", "burden_samples_per_item": 1, "show_options": False,
        "burden_dropout_threshold": 5.0,
    })
    sim.reset_episode(EpisodeConfig())
    with patch.object(sim, "_chat", side_effect=_fake_chat_factory(5, "CONTINUE")):
        utterance, done, state = sim._followup_turn(_case_info(), _one_turn_history())
    assert done is True
    assert state["termination_reason"] == "burden_dropout"
    assert state["decision"] == "CONTINUE"  # LLM didn't comply -- threshold forces done anyway


def test_followup_turn_agreement_sets_termination_reason():
    sim = UserSimulatorV3({
        "model": "stub-model", "burden_samples_per_item": 1, "show_options": False,
    })  # burden_dropout_threshold defaults to inf -- never triggers
    sim.reset_episode(EpisodeConfig())
    with patch.object(sim, "_chat", side_effect=_fake_chat_factory(2, "END")):
        utterance, done, state = sim._followup_turn(_case_info(), _one_turn_history())
    assert done is True
    assert state["termination_reason"] == "agreement"


def test_followup_turn_continue_has_no_termination_reason():
    sim = UserSimulatorV3({
        "model": "stub-model", "burden_samples_per_item": 1, "show_options": False,
    })
    sim.reset_episode(EpisodeConfig())
    with patch.object(sim, "_chat", side_effect=_fake_chat_factory(2, "CONTINUE")):
        utterance, done, state = sim._followup_turn(_case_info(), _one_turn_history())
    assert done is False
    assert state["termination_reason"] is None


# ── show_options=false: belief must be captured as free text, never silently discarded ──────

def test_parse_opening_no_options_keeps_free_text_belief_instead_of_discarding_it():
    # Regression test for the real bug: the no-options turn0 prompt already asks for a
    # free-text belief, but the old code forced it through _letter() (which only accepts a
    # bare A/B/C/D) and silently turned any real answer into None.
    raw = json.dumps({"response": "hi", "belief": "Apocrine hidrocystoma", "confidence": 0.7, "reasoning": "r"})
    _utterance, belief, _reasoning, _confidence = _parse_opening(raw, lettered=False)
    assert belief == "Apocrine hidrocystoma"


def test_parse_opening_options_still_requires_a_letter():
    raw = json.dumps({"response": "hi", "belief": "Apocrine hidrocystoma", "confidence": 0.7, "reasoning": "r"})
    _utterance, belief, _reasoning, _confidence = _parse_opening(raw, lettered=True)
    assert belief is None  # not a valid A-D letter, correctly rejected in lettered mode


def test_parse_followup_no_options_captures_free_text_belief():
    raw = json.dumps({
        "argument_validity": "v", "belief": "Likely pulmonary embolism, will order CTPA.",
        "confidence": 0.6, "response": "ok", "decision": "CONTINUE",
    })
    _utterance, belief, decision, _confidence, _av = _parse_followup_no_options(raw)
    assert belief == "Likely pulmonary embolism, will order CTPA."
    assert decision == "CONTINUE"


def test_followup_turn_no_options_populates_belief_with_free_text_not_none():
    sim = UserSimulatorV3({
        "model": "stub-model", "burden_samples_per_item": 1, "show_options": False,
    })
    sim.reset_episode(EpisodeConfig())

    def fake_chat(messages, temperature=None, response_format=None):
        if temperature == 0.0:
            return _burden_json(2)
        return json.dumps({
            "argument_validity": "v", "belief": "Pneumonia, will start antibiotics.",
            "confidence": 0.6, "response": "ok", "decision": "CONTINUE",
        })

    with patch.object(sim, "_chat", side_effect=fake_chat):
        _utterance, _done, state = sim._followup_turn(_case_info(), _one_turn_history())
    assert state["belief"] == "Pneumonia, will start antibiotics."


# ── scripts/verify_load_judge_v3.py: oracle generation+recovery (ladder) ────────────────
#
# These import the verify script as a plain module (scripts/ has no __init__.py, but a
# directory of .py files added to sys.path imports fine as implicit namespace modules).

import verify_load_judge_v3 as _vljv3  # noqa: E402


def test_ladder_profiles_are_uniform_and_cover_1_to_5():
    assert len(_vljv3._LADDER_PROFILES) == 5
    levels = sorted(p["level"] for p in _vljv3._LADDER_PROFILES)
    assert levels == [1, 2, 3, 4, 5]
    for p in _vljv3._LADDER_PROFILES:
        assert set(p["target"].keys()) == set(_TLX_DIMS)
        assert all(v == p["level"] for v in p["target"].values()), p  # uniform across all 5 dims


def test_burden_oracle_yaml_templates_format_without_keyerror():
    tmpl = _vljv3._load_oracle_prompts()
    assert "generator_system" in tmpl and "generator_user" in tmpl
    ctx = dict(
        mental_demand=3, performance=3, effort=3, frustration=3,
        scenario="s", dialogue_text="d", persona_instruction="p",
    )
    tmpl["generator_system"].format(**ctx)
    tmpl["generator_user"].format(**ctx)


def test_build_oracle_generator_prompt_messages_shape():
    tmpl = _vljv3._load_oracle_prompts()
    messages = _vljv3.build_oracle_generator_prompt(
        tmpl=tmpl, scenario="s", dialogue_text="d", persona_instruction="p",
        target={d: 3 for d in _TLX_DIMS},
    )
    assert [m["role"] for m in messages] == ["system", "user"]


def _valid_oracle_generation_json(**overrides) -> str:
    data = {
        "ai_utterance": "Let's walk through the differential together.",
        "design_rationale": {d: f"rationale for {d}" for d in _TLX_DIMS},
    }
    data.update(overrides)
    return json.dumps(data)


def test_parse_oracle_generation_valid():
    parsed = _vljv3._parse_oracle_generation(_valid_oracle_generation_json())
    assert parsed is not None
    assert parsed["ai_utterance"]
    assert set(parsed["design_rationale"].keys()) == set(_TLX_DIMS)


def test_parse_oracle_generation_all_or_nothing():
    assert _vljv3._parse_oracle_generation("") is None
    assert _vljv3._parse_oracle_generation(json.dumps({"ai_utterance": ""})) is None
    assert _vljv3._parse_oracle_generation(json.dumps({"ai_utterance": "x"})) is None  # no design_rationale
    missing_dim = json.loads(_valid_oracle_generation_json())
    del missing_dim["design_rationale"]["frustration"]
    assert _vljv3._parse_oracle_generation(json.dumps(missing_dim)) is None


def test_compute_oracle_metrics_math():
    # Perfect recovery: observed == target for all 5 rungs -> rho=1.0, MAE=0.0.
    results = [
        {
            "id": p["id"], "level": p["level"], "target": p["target"], "generation_failed": False,
            "observed": dict(p["target"]), "overall_workload_mean": float(p["level"]),
        }
        for p in _vljv3._LADDER_PROFILES
    ]
    metrics = _vljv3.compute_oracle_metrics(results)
    assert metrics["ordinal"]["spearman"] == 1.0
    assert metrics["mae"]["pooled"] == 0.0
    assert metrics["n_generation_failed"] == 0
    assert metrics["all_pass"] is True

    # One generation failure -- excluded from metrics, counted separately.
    results_with_failure = results + [
        {"id": "ladder_extra", "level": 6, "target": {d: 6 for d in _TLX_DIMS},
         "generation_failed": True, "observed": None, "overall_workload_mean": None}
    ]
    metrics2 = _vljv3.compute_oracle_metrics(results_with_failure)
    assert metrics2["n_generation_failed"] == 1
    assert metrics2["mae"]["pooled"] == 0.0  # unaffected by the failed row


def test_verify_load_judge_v3_config_loads():
    config = yaml.safe_load((Path(__file__).parent.parent / "configs" / "verify_load_judge_v3.yaml").read_text())
    assert "judge" in config and "generator" in config and "oracle_persona" in config
    assert config["oracle_persona"] in _vljv3._PERSONAS
