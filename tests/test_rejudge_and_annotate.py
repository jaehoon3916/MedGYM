"""Pure-function and Flask-route tests for Part 4 (real-rollout rejudge + human annotation +
correlation) -- zero LLM calls. The rollout fixture used here
(outputs/scaling_poc_persona_heuristic_delib/burned_out_resident_full/rollouts/6001.jsonl) is
a real past experiment output checked into outputs/, not a synthetic fixture -- if it's ever
moved/cleaned, these tests will need a new path.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

from plugins.user_llm.user_simulator.v3_burden import _TLX_DIMS

import rejudge_rollout as rr
import correlate_human_judge as chj

_ROLLOUT_REL = "scaling_poc_persona_heuristic_delib/burned_out_resident_full/rollouts/6001.jsonl"
_ROLLOUT_PATH = Path(__file__).parent.parent / "outputs" / _ROLLOUT_REL


# ── rejudge_rollout.py ───────────────────────────────────────────────────────────────────

def test_persona_from_folder():
    assert rr._persona_from_folder(_ROLLOUT_PATH) == "burned_out_resident"
    assert rr._persona_from_folder(Path("x/veteran_attending_full/rollouts/1.jsonl")) == "veteran_attending"
    with pytest.raises(ValueError):
        rr._persona_from_folder(Path("x/totally_unknown_folder/rollouts/1.jsonl"))


def test_fmt_history_matches_v3_convention():
    assert rr._fmt_history([]) == "(No conversation yet)"
    history = [
        {"speaker": "user", "text": "I think it's CHF."},
        {"speaker": "medical", "text": "Have you considered PE?"},
    ]
    formatted = rr._fmt_history(history)
    assert formatted == "[Doctor]: I think it's CHF.\n[AI Assistant]: Have you considered PE?"


def test_extract_ai_turns_on_real_rollout():
    assert _ROLLOUT_PATH.exists(), f"fixture rollout missing: {_ROLLOUT_PATH}"
    scenario, history = rr.load_rollout_dialogue(_ROLLOUT_PATH)
    assert scenario
    ai_turns = rr.extract_ai_turns(history)
    assert len(ai_turns) == 8
    assert ai_turns[0]["ai_turn_index"] == 0
    # All AI turns have non-empty utterance text.
    assert all(t["ai_utterance"].strip() for t in ai_turns)
    # 7 of 8 turns have a following user turn recording the old burden; the last AI turn in
    # this episode has none (matches v1.py's score_burden_for_last_turn precedent).
    n_with_old_burden = sum(1 for t in ai_turns if t["old_burden_raw"] is not None)
    assert n_with_old_burden == 7


def test_extract_ai_turns_handles_synthetic_history():
    history = [
        {"speaker": "user", "text": "opening"},
        {"speaker": "medical", "text": "ai reply 1"},
        {"speaker": "user", "text": "doctor reply 1", "user_state": {"cognitive_burden": 2.0}},
        {"speaker": "medical", "text": "ai reply 2"},
        # no trailing user turn -- episode ended right after this AI turn.
    ]
    ai_turns = rr.extract_ai_turns(history)
    assert len(ai_turns) == 2
    assert ai_turns[0]["old_burden_raw"] == 2.0
    assert ai_turns[1]["old_burden_raw"] is None
    assert ai_turns[1]["dialogue_text_before"] == (
        "[Doctor]: opening\n[AI Assistant]: ai reply 1\n[Doctor]: doctor reply 1"
    )


# ── correlate_human_judge.py ─────────────────────────────────────────────────────────────

def test_compute_correlation_perfect_agreement():
    rejudge = {"rows": [
        {"ai_turn_index": i, "new_dims": {d: i + 1 for d in _TLX_DIMS}, "new_overall_workload": float(i + 1)}
        for i in range(5)
    ]}
    annotations = {str(i): {d: i + 1 for d in _TLX_DIMS} for i in range(5)}
    result = chj.compute_correlation(rejudge, annotations)
    assert result["n_matched"] == 5
    assert all(v == 1.0 for v in result["per_dim_spearman"].values())
    assert result["overall_spearman"] == 1.0


def test_compute_correlation_insufficient_points_returns_none():
    rejudge = {"rows": [{"ai_turn_index": 0, "new_dims": {d: 3 for d in _TLX_DIMS}, "new_overall_workload": 3.0}]}
    annotations = {"0": {d: 3 for d in _TLX_DIMS}}
    result = chj.compute_correlation(rejudge, annotations)
    assert result["n_matched"] == 1
    assert result["overall_spearman"] is None
    assert all(v is None for v in result["per_dim_spearman"].values())


def test_compute_correlation_skips_generation_failed_judge_rows():
    rejudge = {"rows": [
        {"ai_turn_index": 0, "new_dims": None, "new_overall_workload": None},  # judge failed this turn
        {"ai_turn_index": 1, "new_dims": {d: 4 for d in _TLX_DIMS}, "new_overall_workload": 4.0},
        {"ai_turn_index": 2, "new_dims": {d: 5 for d in _TLX_DIMS}, "new_overall_workload": 5.0},
    ]}
    annotations = {"0": {d: 1 for d in _TLX_DIMS}, "1": {d: 4 for d in _TLX_DIMS}, "2": {d: 5 for d in _TLX_DIMS}}
    result = chj.compute_correlation(rejudge, annotations)
    assert result["n_matched"] == 2  # turn 0 excluded (judge had no dims)
    assert 0 not in result["matched_turn_indices"]


# ── app.py /annotate routes (Flask test client, no real server, no LLM calls) ───────────

@pytest.fixture
def client():
    import app as appmod
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client()


def test_annotate_get_renders_ai_turns(client):
    resp = client.get(f"/annotate/{_ROLLOUT_REL}")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "AI turn 1" in body
    for dim in _TLX_DIMS:
        assert dim.replace("_", " ") in body


def test_annotate_get_404_for_missing_rollout(client):
    resp = client.get("/annotate/does/not/exist.jsonl")
    assert resp.status_code == 404


def test_annotate_save_then_reload_prefills(client, tmp_path, monkeypatch):
    # Redirect OUTPUTS_DIR-derived annotation writes to a tmp dir so this test doesn't touch
    # the real outputs/ folder (mirrors feedback_no_silent_overwrite -- never pollute real
    # experiment dirs with test artifacts).
    import app as appmod
    fake_annotations = tmp_path / "human_annotations_6001.json"
    monkeypatch.setattr(appmod, "_annotations_path", lambda rel_path: fake_annotations)

    save_resp = client.post(f"/annotate/{_ROLLOUT_REL}/save", data={
        "ai_turn_index": "0", "mental_demand": "3",
        "performance": "3", "effort": "3", "frustration": "2",
    })
    assert save_resp.status_code == 302
    assert fake_annotations.exists()
    saved = json.loads(fake_annotations.read_text())
    assert saved["0"]["mental_demand"] == 3

    reload_resp = client.get(f"/annotate/{_ROLLOUT_REL}")
    assert "scored" in reload_resp.get_data(as_text=True)


def test_annotate_save_rejects_out_of_range_score(client):
    resp = client.post(f"/annotate/{_ROLLOUT_REL}/save", data={
        "ai_turn_index": "0", "mental_demand": "9",
        "performance": "3", "effort": "3", "frustration": "2",
    })
    assert resp.status_code == 400
