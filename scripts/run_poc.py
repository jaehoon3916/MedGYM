#!/usr/bin/env python3
"""
Termination POC: runs v3/v4 as the REAL multi-turn harness, instead of
scripts/run_scaling_poc.py's "always run to max_turns, then re-judge truncated history at
fixed checkpoints" design. This script now selects UserSimulatorV3 or UserSimulatorV4 based on
plugins.user_llm.type. Here the episode actually ends for one of three real reasons:

  1. "agreement"      -- the doctor judges consensus reached (decision == "END").
  2. "burden_dropout"  -- the doctor's accumulated cognitive burden (raw 1-5 sum, see
                          UserSimulatorV3._burden_cumulative) crosses
                          plugins.user_llm.burden_dropout_threshold, forcing withdrawal.
  3. "max_turns"       -- the turn cap is hit with neither of the above happening first.

Accuracy is read DIRECTLY off the doctor's own self-reported `belief` field at the turn that
ended the episode. When plugins.user_llm.show_options is false, the doctor is never shown
lettered options and so `belief` is structurally always None (v3 doesn't even ask for it in
that mode) -- final_judge is then used ONLY as a narrow fallback to map that turn's free-text
history onto an option letter (mirrors scripts/run_scaling_poc.py's identical fallback for its
turn-0 "doctor-alone" belief). When show_options is true, final_judge is never called at all.
For "max_turns", the doctor never got a chance to react to the very last AI turn (env.step()
doesn't call the user_llm again once the cap is hit), so the most recently recorded belief
(reacting to the second-to-last AI turn) is used as-is.

Each case's full transcript is saved via the same RolloutLogger format core/environment.py
already uses (output_path=... to run_episode()), so existing viewers (app.py/annotate_app.py)
work unchanged. A separate results.json records, per case: closed_by, n_turns_actual,
terminal_belief, is_correct, final_cumulative_burden.

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    python scripts/run_poc.py --config configs/termination_poc.yaml

If plugins.user_llm.persona (or info_condition/information_sparsity) in that config is a
LIST instead of a single string, main() below auto-sweeps: one run() per combination,
no separate sweep script needed.
"""
from __future__ import annotations

import json
import hashlib
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import build_plugins, load_yaml
from core.environment import MedicalHACEnvironment
from core.json_utils import safe_json_load
from core.schemas import CaseInfo, DialogueHistory, DialogueTurn, UserState
from core.token_tracker import tracker
from plugins.final_judge_llm.base import FinalJudgeLLMPlugin
from plugins.user_llm.user_simulator.v3 import UserSimulatorV3
from plugins.user_llm.user_simulator.v4 import UserSimulatorV4

from scripts.run_dialogue import load_dotenv  # avoids needing `source .env` before running
from scripts.run_scaling_poc import _load_balanced_cases, _resolve_info_condition, _confirm

_CLOSED_BY_REASONS = ("agreement", "burden_dropout", "max_turns")


def _burden_to_close(
    burden_by_user_turn: dict[int, float],
    natural_end_turn: int | None,
    n_turns_actual: int,
) -> tuple[float, float, int]:
    """Un-confounded burden: burden accrued only up to the doctor's natural close.

    ``final_cumulative_burden`` sums every forced turn (force_full_turns=true keeps the
    episode running after the doctor said END, padding it out to max_turns with cheap
    confirmations). That deflates the headline metric for policies that close early, so an
    early-closing policy looks "gentle" when it is really just coasting on padding. Truncating
    at ``natural_end_turn`` removes the padding; policies that never close (natural_end_turn is
    None) fall back to the full episode and get no discount.

    Returns (burden_to_close, burden_per_substantive_turn, close_turn).
    """
    close_turn = natural_end_turn if natural_end_turn is not None else n_turns_actual
    total = sum(burden_by_user_turn.get(k, 0.0) for k in range(1, close_turn + 1))
    per_turn = total / close_turn if close_turn > 0 else 0.0
    return round(total, 4), round(per_turn, 4), close_turn


def _record_from_rollout(
    rollout_path: Path,
    case_info: CaseInfo,
    checkpoints: list[int],
    final_judge: FinalJudgeLLMPlugin | None,
) -> dict | None:
    """Reconstruct a run_episode_with_termination record from a saved rollout JSONL.

    Returns None if the file is missing or malformed (caller will re-run the episode).
    """
    try:
        lines = rollout_path.read_text().splitlines()
        if not lines:
            return None
        last = json.loads(lines[-1])
        raw_turns: list[dict] = last.get("dialogue_history", [])
        if not raw_turns:
            return None
        turns = [
            DialogueTurn(
                speaker=t["speaker"],
                text=t.get("text", ""),
                action=t.get("action"),
                user_state=UserState(**t["user_state"]) if t.get("user_state") else None,
            )
            for t in raw_turns
        ]
        n_turns_actual = int(last.get("turn_id", len(lines) - 1)) + 1
        closed_by = last.get("closed_by") or "max_turns"
    except Exception:
        return None

    # -- analysis (mirrors run_episode_with_termination logic post-run_episode) --
    last_user_state = _last_user_state(turns)
    terminal_belief = _resolve_belief(
        last_user_state.get("belief"), final_judge, case_info, case_info.case_id, turns
    )
    is_correct = terminal_belief is not None and terminal_belief == case_info.correct_option

    burden_by_user_turn: dict[int, float] = {}
    medical_count = 0
    for t in turns:
        if t.speaker == "medical":
            medical_count += 1
        elif t.speaker == "user" and medical_count > 0:
            burden_by_user_turn[medical_count] = _state(t).get("cognitive_burden", 0.0)

    traj = _belief_trajectory(turns)
    checkpoint_results: dict[int, dict] = {}
    for c in [0] + list(checkpoints):
        c_eff = min(c, n_turns_actual)
        raw_belief = _checkpoint_belief(traj, c_eff)
        turns_prefix = turns[:1] if c_eff == 0 else turns[: 2 * c_eff]
        belief = _resolve_belief(raw_belief, final_judge, case_info, case_info.case_id, turns_prefix)
        cum_burden = sum(burden_by_user_turn.get(k, 0.0) for k in range(1, c_eff + 1))
        checkpoint_results[c] = {
            "self_reported_belief": belief,
            "is_correct": belief is not None and belief == case_info.correct_option,
            "cumulative_burden": round(cum_burden, 4),
            "avg_burden": round(cum_burden / c_eff, 4) if c_eff > 0 else 0.0,
        }

    alone_correct = checkpoint_results[0]["is_correct"]
    natural_end_turn: int | None = None
    natural_end_correct: bool | None = None
    mc = 0
    for t in turns:
        if t.speaker == "medical":
            mc += 1
        elif t.speaker == "user" and mc > 0:
            if _state(t).get("decision") == "END" and natural_end_turn is None:
                natural_end_turn = mc
                cp_key = min(mc, max(checkpoint_results.keys()))
                natural_end_correct = checkpoint_results.get(cp_key, {}).get("is_correct")

    final_burden = last_user_state.get("cognitive_burden_cumulative")
    burden_n_ok = sum(_state(t).get("cognitive_burden_n_ok", 0) for t in turns if t.speaker == "user")
    burden_n_attempted = sum(_state(t).get("cognitive_burden_n_attempted", 0) for t in turns if t.speaker == "user")
    burden_to_close, burden_per_substantive_turn, _ = _burden_to_close(
        burden_by_user_turn, natural_end_turn, n_turns_actual
    )

    record: dict = {
        "case_id": case_info.case_id,
        "specialty": case_info.metadata.get("specialty"),
        "alone_correct": alone_correct,
        "n_turns_actual": n_turns_actual,
        "closed_by": closed_by,
        "terminal_belief": terminal_belief,
        "is_correct": is_correct,
        "final_cumulative_burden": final_burden,
        "burden_to_close": burden_to_close,
        "burden_per_substantive_turn": burden_per_substantive_turn,
        "burden_judge_calls_ok": burden_n_ok,
        "burden_judge_calls_attempted": burden_n_attempted,
        "checkpoints": {str(k): v for k, v in checkpoint_results.items()},
        "natural_end_turn": natural_end_turn,
        "natural_end_correct": natural_end_correct,
        "burden_by_user_turn": {str(k): round(v, 4) for k, v in burden_by_user_turn.items()},
    }
    return record


def _state(turn: DialogueTurn) -> dict:
    return turn.user_state.model_dump() if turn.user_state is not None else {}


def _last_user_state(turns: list[DialogueTurn]) -> dict:
    return next((_state(t) for t in reversed(turns) if t.speaker == "user"), {})


def _belief_trajectory(turns: list[DialogueTurn]) -> list[tuple[int, str | None]]:
    """(n_medical_turns_so_far, self_reported_belief) per user turn, in order -- the first
    entry is always (0, opening_belief) since the opening user turn precedes any medical
    turn. Belief comes straight from user_state -- None whenever show_options=false (v3
    never asks for it then); see _resolve_belief for the final_judge fallback in that case."""
    traj: list[tuple[int, str | None]] = []
    medical_count = 0
    for t in turns:
        if t.speaker == "medical":
            medical_count += 1
        elif t.speaker == "user":
            traj.append((medical_count, _state(t).get("belief")))
    return traj


def _checkpoint_belief(traj: list[tuple[int, str | None]], c_eff: int) -> str | None:
    """Belief as of (at most) c_eff completed medical turns -- the most recent trajectory
    entry not past c_eff. If the episode ended at max_turns right after the cap-hitting
    medical turn (no further user reaction), this naturally collapses to the same belief
    used as terminal_belief, since c_eff caps at n_turns_actual."""
    candidates = [b for n, b in traj if n <= c_eff]
    return candidates[-1] if candidates else None


_VALID_LETTERS = ("A", "B", "C", "D")


def _resolve_belief(
    belief: str | None, final_judge: FinalJudgeLLMPlugin | None, case_info: CaseInfo,
    case_id: str, turns_prefix: list[DialogueTurn],
) -> str | None:
    """show_options=true: `belief` is already a single A-D letter from v3 -- use as-is,
    final_judge never called.

    show_options=false: v3 still captures `belief` every turn, but as FREE TEXT (the doctor's
    current clinical impression in their own words -- see utterance_followup_system_no_options
    / utterance_turn0_system_no_options's Output Format) -- there's nothing lettered to read
    directly. Map THAT specific free-text statement (not the whole dialogue) onto an option
    letter via final_judge, using a single synthetic turn containing just the belief text --
    precise, not a re-judgment of the full truncated history.

    If `belief` is None (a genuine parse failure -- v3 exhausted its retries and never
    produced any belief text at all), fall back to judging the actual prefix of dialogue
    turns instead, mirroring scripts/run_scaling_poc.py's turn-0 doctor-alone fallback."""
    if belief is not None and belief.strip().upper() in _VALID_LETTERS:
        return belief.strip().upper()
    if final_judge is None:
        return None
    if belief is not None:
        # Free-text belief -- judge it directly, not the surrounding conversation.
        synthetic_history = DialogueHistory(case_id=case_id, turns=[DialogueTurn(speaker="user", text=belief)])
    elif turns_prefix:
        synthetic_history = DialogueHistory(case_id=case_id, turns=turns_prefix)
    else:
        return None
    fj = final_judge.judge(case_info, synthetic_history)
    option = fj.concluded_option.strip().upper()
    return option if option in _VALID_LETTERS else None


# ── Episode (real termination, belief-based checkpoints) ───────────────────────────────────

def _build_user_simulator(user_llm_cfg: dict):
    user_type = str(user_llm_cfg.get("type", "v3")).strip().lower()
    if user_type == "v4":
        return UserSimulatorV4(user_llm_cfg)
    if user_type == "v3":
        return UserSimulatorV3(user_llm_cfg)
    raise ValueError(
        f"Unsupported plugins.user_llm.type={user_type!r} in run_poc.py; expected 'v3' or 'v4'."
    )


def run_episode_with_termination(
    case_info: CaseInfo,
    user_llm_cfg: dict,
    medical_llm,
    fact_validator_llm,
    policy,
    final_judge: FinalJudgeLLMPlugin | None,
    config: dict,
    max_turns: int,
    output_dir: Path,
    checkpoints: list[int],
    agenda_planner=None,
    resolution_tracker=None,
) -> dict:
    # Fresh instance per case (not build_plugins's shared one) -- v3/v4 carry per-episode
    # mutable state, so a shared instance would race across ThreadPoolExecutor workers.
    user_llm = _build_user_simulator(user_llm_cfg)
    # final_judge=None here on purpose -- the environment's own _finalize() is unused; THIS
    # script calls final_judge.judge() itself, only as a belief fallback (see _resolve_belief).
    if agenda_planner is not None:
        from core.agenda_environment import AgendaEnvironment
        ai_alone_result = _simulate_ai_alone_with_reasoning(case_info, config["plugins"]["medical_llm"])
        env = AgendaEnvironment(
            user_llm, medical_llm, fact_validator_llm, policy, config, final_judge=None,
            agenda_planner=agenda_planner, resolution_tracker=resolution_tracker,
        )
        env.set_ai_alone_result(ai_alone_result["selected_option"], ai_alone_result["reasoning"])
    else:
        ai_alone_result = None
        env = MedicalHACEnvironment(user_llm, medical_llm, fact_validator_llm, policy, config, final_judge=None)
    rollout_path = output_dir / "rollouts" / f"{case_info.case_id}.jsonl"

    # Fast path: rollout already saved → reconstruct record without re-running the episode.
    if rollout_path.exists():
        rec = _record_from_rollout(rollout_path, case_info, checkpoints, final_judge)
        if rec is not None:
            return rec

    steps = env.run_episode(case_info, max_turns=max_turns, output_path=rollout_path, episode_config=None)

    history = env.history
    turns = history.turns if history is not None else []
    n_turns_actual = len(steps)
    closed_by = steps[-1].metadata.get("closed_by") if steps else "agreement"

    last_user_state = _last_user_state(turns)
    terminal_belief = _resolve_belief(
        last_user_state.get("belief"), final_judge, case_info, case_info.case_id, turns,
    )
    is_correct = terminal_belief is not None and terminal_belief == case_info.correct_option

    burden_n_ok = sum(_state(t).get("cognitive_burden_n_ok", 0) for t in turns if t.speaker == "user")
    burden_n_attempted = sum(_state(t).get("cognitive_burden_n_attempted", 0) for t in turns if t.speaker == "user")

    # Per-turn incremental burden (same convention as run_scaling_poc.py: user turn k carries
    # burden from medical turn k-1). v3 exposes cognitive_burden (incremental) per user turn.
    burden_by_user_turn: dict[int, float] = {}
    for k in range(1, n_turns_actual + 1):
        idx = 2 * k
        if idx >= len(turns):
            break
        burden_by_user_turn[k] = _state(turns[idx]).get("cognitive_burden", 0.0)

    # "What if we'd stopped here" curve -- belief-based, with the same show_options=false
    # final_judge fallback as terminal_belief above (never called when show_options=true).
    traj = _belief_trajectory(turns)
    checkpoint_results: dict[int, dict] = {}
    for c in [0] + list(checkpoints):
        c_eff = min(c, n_turns_actual)
        raw_belief = _checkpoint_belief(traj, c_eff)
        # Same prefix convention as run_scaling_poc.py: c_eff==0 -> opening turn only,
        # else -> the first 2*c_eff turns (c_eff completed (user, medical) pairs).
        turns_prefix = turns[:1] if c_eff == 0 else turns[: 2 * c_eff]
        belief = _resolve_belief(raw_belief, final_judge, case_info, case_info.case_id, turns_prefix)
        cum_burden = sum(burden_by_user_turn.get(k, 0.0) for k in range(1, c_eff + 1))
        checkpoint_results[c] = {
            "self_reported_belief": belief,
            "is_correct": belief is not None and belief == case_info.correct_option,
            "cumulative_burden": round(cum_burden, 4),
            "avg_burden": round(cum_burden / c_eff, 4) if c_eff > 0 else 0.0,
        }

    alone_correct = checkpoint_results[0]["is_correct"]

    # natural_end_turn: the number of completed AI turns at the first user turn where
    # decision == "END" (the moment the doctor would have naturally stopped). Only meaningful
    # when force_full_turns=true -- otherwise closed_by=="agreement" already captures this
    # and the episode DID stop there. None = doctor never said END (always CONTINUE or
    # burden_dropout/max_turns terminated before any agreement).
    natural_end_turn: int | None = None
    natural_end_correct: bool | None = None
    medical_count = 0
    for t in turns:
        if t.speaker == "medical":
            medical_count += 1
        elif t.speaker == "user" and medical_count > 0:
            if _state(t).get("decision") == "END" and natural_end_turn is None:
                natural_end_turn = medical_count
                cp_key = min(medical_count, max(checkpoint_results.keys()))
                natural_end_correct = checkpoint_results.get(cp_key, {}).get("is_correct")

    burden_to_close, burden_per_substantive_turn, _ = _burden_to_close(
        burden_by_user_turn, natural_end_turn, n_turns_actual
    )
    record: dict = {
        "case_id": case_info.case_id,
        "specialty": case_info.metadata.get("specialty"),
        "alone_correct": alone_correct,
        "n_turns_actual": n_turns_actual,
        "closed_by": closed_by,
        "terminal_belief": terminal_belief,
        "is_correct": is_correct,
        "final_cumulative_burden": last_user_state.get("cognitive_burden_cumulative"),
        "burden_to_close": burden_to_close,
        "burden_per_substantive_turn": burden_per_substantive_turn,
        "burden_judge_calls_ok": burden_n_ok,
        "burden_judge_calls_attempted": burden_n_attempted,
        "natural_end_turn": natural_end_turn,
        "natural_end_correct": natural_end_correct,
        "checkpoints": {str(c): v for c, v in checkpoint_results.items()},
    }
    # Agenda-arm extras (only present when agenda_planner was configured).
    if ai_alone_result is not None:
        record["human_alone_correct"] = alone_correct
        record["ai_alone_correct"] = ai_alone_result["alone_correct"]
        agenda = getattr(env, "agenda", None)
        if agenda is not None:
            items = [
                {"id": it.id, "issue": it.issue, "status": it.status}
                for it in agenda.items
            ]
            record["agenda"] = items
            record["n_resolved"] = sum(1 for it in agenda.items if it.status == "resolved")
            record["n_unresolved_at_termination"] = sum(
                1 for it in agenda.items if it.status == "unresolved"
            )
    return record


# ── AI-alone baseline ────────────────────────────────────────────────────────────────────

_AI_ALONE_SYSTEM = (
    "You are a medical AI. Given a clinical case, answer the multiple-choice question "
    "by selecting the single best option."
)
_AI_ALONE_USER = """\
[Clinical Case]
{scenario}

[Image]
{image_caption}

[Question]
{question}

[Options]
{options_text}

Respond with JSON only: {{"selected_option": "<single letter A-D>", "reasoning": "<1-2 sentences>"}}"""

_AI_ALONE_USER_WITH_REASONING = """\
[Clinical Case]
{scenario}

[Image]
{image_caption}

[Question]
{question}

[Options]
{options_text}

Respond with JSON only: {{"selected_option": "<single letter A-D>", "reasoning": "<3-5 sentences: your clinical reasoning and why you chose this option over the main alternatives>"}}"""


def _simulate_ai_alone_with_reasoning(case_info: CaseInfo, medical_llm_cfg: dict) -> dict:
    """One-shot MCQ with full reasoning text (for Disagreement Analyzer input).
    Returns {selected_option, reasoning, alone_correct}.
    """
    base_url = medical_llm_cfg.get("base_url", "http://localhost:8001/v1")
    api_key = medical_llm_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
    model = medical_llm_cfg["model"]
    client = OpenAI(base_url=base_url, api_key=api_key)
    options_text = "\n".join(f"  {k}. {v}" for k, v in case_info.options.items())
    messages = [
        {"role": "system", "content": _AI_ALONE_SYSTEM},
        {"role": "user", "content": _AI_ALONE_USER_WITH_REASONING.format(
            scenario=case_info.scenario,
            image_caption=case_info.caption or "N/A",
            question=case_info.question,
            options_text=options_text,
        )},
    ]
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    data = safe_json_load(text)
    selected = str(data.get("selected_option", "")).strip().upper()
    if selected not in _VALID_LETTERS:
        selected = ""
    return {
        "selected_option": selected,
        "reasoning": str(data.get("reasoning", "")).strip(),
        "alone_correct": selected == str(case_info.correct_option).strip().upper(),
    }


def _simulate_ai_alone(case_info: CaseInfo, medical_llm_cfg: dict) -> bool:
    """One-shot: medical model answers the MCQ without any dialogue. Returns True if correct."""
    base_url = medical_llm_cfg.get("base_url", "http://localhost:8001/v1")
    api_key = medical_llm_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
    model = medical_llm_cfg["model"]
    client = OpenAI(base_url=base_url, api_key=api_key)
    options_text = "\n".join(f"  {k}. {v}" for k, v in case_info.options.items())
    messages = [
        {"role": "system", "content": _AI_ALONE_SYSTEM},
        {"role": "user", "content": _AI_ALONE_USER.format(
            scenario=case_info.scenario,
            image_caption=case_info.caption or "N/A",
            question=case_info.question,
            options_text=options_text,
        )},
    ]
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    data = safe_json_load(text)
    selected = str(data.get("selected_option", "")).strip().upper()
    if selected not in _VALID_LETTERS:
        selected = ""
    return selected == str(case_info.correct_option).strip().upper()


def _case_cache_hash(case_info: CaseInfo) -> str:
    payload = {
        "case_id": case_info.case_id,
        "scenario": case_info.scenario,
        "caption": case_info.caption,
        "question": case_info.question,
        "options": case_info.options,
        "correct_option": case_info.correct_option,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _ai_alone_cache_key(case_info: CaseInfo, medical_llm_cfg: dict) -> str:
    model = str(medical_llm_cfg["model"])
    return f"{model}::oneshot_mcq_v1::{case_info.case_id}::{_case_cache_hash(case_info)}"


def _default_ai_alone_cache_path(medical_llm_cfg: dict) -> Path:
    model_slug = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(medical_llm_cfg["model"])
    )
    return _ROOT / "outputs" / "_cache" / f"ai_alone_{model_slug}.json"


def _load_ai_alone_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        print(f"  WARNING: ai_alone cache is invalid JSON, ignoring: {cache_path}")
        return {"version": 1, "entries": {}}
    if "entries" not in data or not isinstance(data["entries"], dict):
        return {"version": 1, "entries": data if isinstance(data, dict) else {}}
    return data


def _save_ai_alone_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def _seed_ai_alone_cache_from_results(
    cache: dict,
    cases: list[CaseInfo],
    medical_llm_cfg: dict,
) -> int:
    """Backfill the shared ai_alone cache from old run_poc results.json files.

    Old records only store the boolean, not the selected option. That is still enough for the
    headline accuracy and for downstream complementarity fields that consume ai_alone_correct.
    """
    entries = cache.setdefault("entries", {})
    cases_by_id = {c.case_id: c for c in cases}
    target_model = str(medical_llm_cfg["model"])
    seeded = 0
    for results_path in (_ROOT / "outputs").glob("**/results.json"):
        try:
            data = json.loads(results_path.read_text())
        except Exception:
            continue
        if str(data.get("run_meta", {}).get("model", "")) != target_model:
            continue
        for record in data.get("records", []):
            cid = record.get("case_id")
            if cid not in cases_by_id or "ai_alone_correct" not in record:
                continue
            case_info = cases_by_id[cid]
            key = _ai_alone_cache_key(case_info, medical_llm_cfg)
            if key in entries:
                continue
            entries[key] = {
                "case_id": cid,
                "model": target_model,
                "prompt": "oneshot_mcq_v1",
                "case_hash": _case_cache_hash(case_info),
                "correct": bool(record["ai_alone_correct"]),
                "selected_option": None,
                "source": str(results_path.relative_to(_ROOT)),
            }
            seeded += 1
    return seeded


def _load_or_measure_ai_alone(
    cases: list[CaseInfo],
    medical_llm_cfg: dict,
    concurrency: int,
    cache_path: Path,
) -> dict[str, bool]:
    cache = _load_ai_alone_cache(cache_path)
    entries = cache.setdefault("entries", {})
    seeded = _seed_ai_alone_cache_from_results(cache, cases, medical_llm_cfg)
    if seeded:
        _save_ai_alone_cache(cache_path, cache)
        print(f"  Seeded ai_alone cache from old results: {seeded} case(s).")

    ai_alone_by_case: dict[str, bool] = {}
    pending: list[CaseInfo] = []
    for case_info in cases:
        key = _ai_alone_cache_key(case_info, medical_llm_cfg)
        cached = entries.get(key)
        if cached is not None and "correct" in cached:
            ai_alone_by_case[case_info.case_id] = bool(cached["correct"])
        else:
            pending.append(case_info)

    if pending:
        print(
            f"  ai_alone cache: {len(cases) - len(pending)}/{len(cases)} hit(s), "
            f"{len(pending)} call(s) needed."
        )
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            alone_futures = {
                ex.submit(_simulate_ai_alone, case_info, medical_llm_cfg): case_info
                for case_info in pending
            }
            for fut in tqdm(as_completed(alone_futures), total=len(alone_futures),
                            desc="ai_alone", unit="case"):
                case_info = alone_futures[fut]
                correct = bool(fut.result())
                ai_alone_by_case[case_info.case_id] = correct
                entries[_ai_alone_cache_key(case_info, medical_llm_cfg)] = {
                    "case_id": case_info.case_id,
                    "model": str(medical_llm_cfg["model"]),
                    "prompt": "oneshot_mcq_v1",
                    "case_hash": _case_cache_hash(case_info),
                    "correct": correct,
                    "selected_option": None,
                    "source": "fresh_api_call",
                }
        _save_ai_alone_cache(cache_path, cache)
        print(f"  Saved ai_alone cache: {cache_path}")
    else:
        print(f"  ai_alone cache: all {len(cases)} case(s) hit. No API calls.")

    return ai_alone_by_case


# ── Aggregation ──────────────────────────────────────────────────────────────────────────

_TRAJECTORY_CLASSES = ("always_correct", "self_corrected", "locked_wrong", "regressed")
_ALONE_BUCKETS = (True, False)


def _classify_trajectory(first_correct: bool, last_correct: bool) -> str:
    if first_correct and last_correct:
        return "always_correct"
    if (not first_correct) and last_correct:
        return "self_corrected"
    if (not first_correct) and (not last_correct):
        return "locked_wrong"
    return "regressed"


def aggregate(records: list[dict], checkpoints: list[int], ai_alone_correct: list[bool] | None = None) -> dict:
    n = len(records)
    all_cps = [0] + list(checkpoints)

    # ── same structure as run_scaling_poc.py ─────────────────────────────────
    by_bucket: dict[bool, dict[int, dict[str, list]]] = {
        b: {c: {"correct": [], "burden": []} for c in all_cps} for b in _ALONE_BUCKETS
    }
    trajectory_counts = {b: {cls: 0 for cls in _TRAJECTORY_CLASSES} for b in _ALONE_BUCKETS}
    burden_by_traj: dict[str, list[float]] = {cls: [] for cls in _TRAJECTORY_CLASSES}

    for r in records:
        bucket = r["alone_correct"]
        cps = r["checkpoints"]
        for c in all_cps:
            cp = cps[str(c)]
            by_bucket[bucket][c]["correct"].append(cp["is_correct"])
            by_bucket[bucket][c]["burden"].append(cp["cumulative_burden"])

        first_correct = cps[str(all_cps[0])]["is_correct"]
        last_correct = cps[str(all_cps[-1])]["is_correct"]
        cls = _classify_trajectory(first_correct, last_correct)
        trajectory_counts[bucket][cls] += 1
        burden_by_traj[cls].append(cps[str(all_cps[-1])]["cumulative_burden"])

    doctor_alone_accuracy = round(
        sum(1 for r in records if r["alone_correct"]) / n, 4
    ) if n else 0.0

    curve = {
        bucket: {
            c: {
                "accuracy": round(sum(by_bucket[bucket][c]["correct"]) / len(by_bucket[bucket][c]["correct"]), 4)
                if by_bucket[bucket][c]["correct"] else 0.0,
                "avg_cumulative_burden": round(statistics.mean(by_bucket[bucket][c]["burden"]), 4)
                if by_bucket[bucket][c]["burden"] else 0.0,
                "n": len(by_bucket[bucket][c]["correct"]),
            }
            for c in all_cps
        }
        for bucket in _ALONE_BUCKETS
    }
    burden_by_trajectory = {
        cls: round(statistics.mean(vals), 4) if vals else None
        for cls, vals in burden_by_traj.items()
    }

    # ── termination-specific ─────────────────────────────────────────────────
    by_closed_by: dict[str, dict] = {}
    for reason in _CLOSED_BY_REASONS:
        rows = [r for r in records if r["closed_by"] == reason]
        burdens = [r["final_cumulative_burden"] for r in rows if r["final_cumulative_burden"] is not None]
        b2c = [r["burden_to_close"] for r in rows if r.get("burden_to_close") is not None]
        by_closed_by[reason] = {
            "n": len(rows),
            "rate": round(len(rows) / len(records), 4) if records else 0.0,
            "accuracy": round(sum(1 for r in rows if r["is_correct"]) / len(rows), 4) if rows else None,
            "avg_n_turns": round(statistics.mean(r["n_turns_actual"] for r in rows), 2) if rows else None,
            "avg_final_cumulative_burden": round(statistics.mean(burdens), 2) if burdens else None,
            "avg_burden_to_close": round(statistics.mean(b2c), 2) if b2c else None,
        }

    avg_n_turns = round(statistics.mean(r["n_turns_actual"] for r in records), 2) if records else 0.0

    # Un-confounded burden: burden accrued up to the doctor's natural close, and per
    # substantive (pre-close) turn. These strip the post-close padding that force_full_turns
    # bakes into final_cumulative_burden -- report these, not the 8-turn cumulative, when
    # comparing policies, or early-closing policies get credited for cheap coasting turns.
    # .get() guards results.json files written before these fields existed.
    _b2c = [r["burden_to_close"] for r in records if r.get("burden_to_close") is not None]
    _bpt = [r["burden_per_substantive_turn"] for r in records
            if r.get("burden_per_substantive_turn") is not None]
    avg_burden_to_close = round(statistics.mean(_b2c), 4) if _b2c else None
    avg_burden_per_substantive_turn = round(statistics.mean(_bpt), 4) if _bpt else None

    burden_calls_ok = sum(r.get("burden_judge_calls_ok", 0) for r in records)
    burden_calls_attempted = sum(r.get("burden_judge_calls_attempted", 0) for r in records)

    # ai_alone_accuracy: medical model one-shot performance (no dialogue). Optional -- only
    # present when run() was asked to measure it (measure_ai_alone: true in config).
    ai_alone_accuracy = (
        round(sum(ai_alone_correct) / len(ai_alone_correct), 4) if ai_alone_correct else None
    )

    # post-agreement scaling: among cases where the doctor would have naturally stopped
    # (natural_end_turn is not None), how many recovered (wrong at natural_end → correct at
    # terminal) or regressed (correct at natural_end → wrong at terminal) due to continued turns.
    # Only meaningful when force_full_turns=true -- otherwise natural_end_turn == n_turns_actual.
    nat_end_records = [r for r in records if r.get("natural_end_turn") is not None]
    if nat_end_records:
        recovered_after_agreement = round(
            sum(1 for r in nat_end_records
                if not r["natural_end_correct"] and r["is_correct"]) / len(nat_end_records), 4
        )
        regressed_after_agreement = round(
            sum(1 for r in nat_end_records
                if r["natural_end_correct"] and not r["is_correct"]) / len(nat_end_records), 4
        )
        # "agreement" reinterpreted via the FIRST END token (natural_end_turn). This is the
        # unified agreement signal across force_full_turns true/false: when force_full_turns=false
        # natural_end_turn == n_turns_actual, so this matches by_closed_by["agreement"] exactly;
        # when true (END suppressed, closed_by collapses to max_turns) it recovers the turn the
        # doctor WOULD have stopped at. plot_end_turn uses this for the agreement bar.
        natural_end_agreement = {
            "n": len(nat_end_records),
            "rate": round(len(nat_end_records) / len(records), 4) if records else 0.0,
            "avg_n_turns": round(statistics.mean(r["natural_end_turn"] for r in nat_end_records), 2),
            "accuracy": round(
                sum(1 for r in nat_end_records if r["natural_end_correct"]) / len(nat_end_records), 4
            ),
        }
    else:
        recovered_after_agreement = None
        regressed_after_agreement = None
        natural_end_agreement = None

    result: dict = {
        "all_checkpoints": all_cps,
        "doctor_alone_accuracy": doctor_alone_accuracy,
        "curve": curve,
        "trajectory_counts": trajectory_counts,
        "burden_by_trajectory": burden_by_trajectory,
        "by_closed_by": by_closed_by,
        "avg_n_turns": avg_n_turns,
        "avg_burden_to_close": avg_burden_to_close,
        "avg_burden_per_substantive_turn": avg_burden_per_substantive_turn,
        "burden_judge_calls_ok": burden_calls_ok,
        "burden_judge_calls_attempted": burden_calls_attempted,
    }
    if ai_alone_accuracy is not None:
        result["ai_alone_accuracy"] = ai_alone_accuracy
    if recovered_after_agreement is not None:
        result["recovered_after_agreement"] = recovered_after_agreement
        result["regressed_after_agreement"] = regressed_after_agreement
        result["n_with_natural_end"] = len(nat_end_records)
        result["natural_end_agreement"] = natural_end_agreement
    # Agenda-arm complementarity (present only when records have human/ai_alone_correct).
    if records and "human_alone_correct" in records[0]:
        import statistics as _stats
        compl = [
            (1 if r["is_correct"] else 0)
            - max(1 if r["human_alone_correct"] else 0, 1 if r["ai_alone_correct"] else 0)
            for r in records
        ]
        result["human_alone_accuracy"] = round(
            sum(1 for r in records if r["human_alone_correct"]) / n, 4
        )
        result["agenda_ai_alone_accuracy"] = round(
            sum(1 for r in records if r["ai_alone_correct"]) / n, 4
        )
        result["complementarity"] = round(_stats.mean(compl), 4)
        result["avg_n_resolved"] = round(
            _stats.mean(r.get("n_resolved", 0) for r in records), 2
        )
        result["avg_n_unresolved_at_termination"] = round(
            _stats.mean(r.get("n_unresolved_at_termination", 0) for r in records), 2
        )
    return result


def print_report(scores: dict, condition_label: str) -> None:
    all_cps = scores["all_checkpoints"]
    print()
    print("=" * 86)
    print(f"  Termination POC: real episode termination + belief-based accuracy ({condition_label})")
    print("=" * 86)
    print(f"\n  doctor_alone_accuracy (turn-0 self-judgment, no AI input): {scores['doctor_alone_accuracy']:.4f}")
    if scores.get("ai_alone_accuracy") is not None:
        print(f"  ai_alone_accuracy    (one-shot MCQ, no dialogue):        {scores['ai_alone_accuracy']:.4f}")
    if scores.get("recovered_after_agreement") is not None:
        n_nat = scores["n_with_natural_end"]
        print(f"  post-agreement scaling (n={n_nat} cases with natural_end_turn):")
        print(f"    recovered_after_agreement (wrong→correct after forced turns): {scores['recovered_after_agreement']:.4f}")
        print(f"    regressed_after_agreement (correct→wrong after forced turns):  {scores['regressed_after_agreement']:.4f}")
    ok, attempted = scores["burden_judge_calls_ok"], scores["burden_judge_calls_attempted"]
    if attempted and ok < attempted:
        print(f"  *** burden judge: {attempted - ok}/{attempted} calls FAILED to parse ***")
    else:
        print(f"  burden judge: {ok}/{attempted} calls parsed OK")

    for bucket in _ALONE_BUCKETS:
        label = "alone_correct" if bucket else "alone_incorrect"
        n = scores["curve"][bucket][all_cps[0]]["n"]
        print(f"\n  {label}  (n={n})")
        print("  checkpoint:      " + "  ".join(f"{c:>6}" for c in all_cps))
        print("  accuracy:        " + "  ".join(f"{scores['curve'][bucket][c]['accuracy']:>6.3f}" for c in all_cps))
        print("  avg_cum_burden:  " + "  ".join(
            f"{scores['curve'][bucket][c]['avg_cumulative_burden']:>6.3f}" for c in all_cps))

    print("\n  by closed_by:")
    print(f"  {'reason':<16}{'n':>5}{'accuracy':>10}{'avg_turns':>11}{'avg_final_burden':>18}")
    for reason in _CLOSED_BY_REASONS:
        row = scores["by_closed_by"][reason]
        acc = f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "n/a"
        turns_s = f"{row['avg_n_turns']:.2f}" if row["avg_n_turns"] is not None else "n/a"
        burden = f"{row['avg_final_cumulative_burden']:.2f}" if row["avg_final_cumulative_burden"] is not None else "n/a"
        print(f"  {reason:<16}{row['n']:>5}{acc:>10}{turns_s:>11}{burden:>18}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────────────────

def run(config: dict) -> dict:
    """Run one termination-POC condition from an already-loaded config dict. Returns the
    aggregated `scores` so run_sweep() can collect them without re-reading results.json."""
    load_dotenv()
    tracker.reset()  # see run_scaling_poc.py's run() for why this must be per-call
    exp = config["experiment"]

    for field in ("persona", "information_sparsity", "info_condition"):
        if isinstance(config["plugins"]["user_llm"].get(field), list):
            raise ValueError(
                f"plugins.user_llm.{field} is a list -- run() handles exactly one value per "
                "call; main() below expands lists into multiple run() calls automatically, so "
                "call main()/this script's CLI entry point instead of run() directly."
            )
    _resolve_info_condition(config)

    output_dir = _ROOT / "outputs" / exp["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    max_turns = int(exp["max_turns"])
    checkpoints = list(exp.get("checkpoints", []))

    data_dir = exp.get("data_dir")
    data_path = exp.get("data_path")
    assert not (data_dir and data_path), (
        "experiment.data_dir and experiment.data_path are mutually exclusive"
    )
    if data_dir is not None:
        n_cases = exp.get("n_cases")
        assert n_cases is not None, "experiment.n_cases is required with data_dir"
        raw_cases, specialty_counts = _load_balanced_cases(_ROOT / data_dir, int(n_cases))
        print(f"Balanced sample across specialties: {specialty_counts}")
    else:
        case_path = _ROOT / data_path
        raw = json.loads(case_path.read_text())
        raw_cases = raw if isinstance(raw, list) else [raw]
        n_cases = exp.get("n_cases")
        if n_cases is not None:
            raw_cases = raw_cases[: int(n_cases)]
        specialty_counts = None
    cases = [CaseInfo(**c) for c in raw_cases]
    print(f"Loaded {len(cases)} case(s) from {data_dir or data_path}")

    results_path = output_dir / "results.json"
    run_meta = {
        "model": config["plugins"]["medical_llm"]["model"],
        "max_turns": max_turns,
        "data_path": data_path,
        "data_dir": data_dir,
        "specialty_counts": specialty_counts,
        "policy_type": config["plugins"]["policy"]["type"],
        "checkpoints": checkpoints,
        "burden_dropout_threshold": config["plugins"]["user_llm"].get("burden_dropout_threshold"),
        "user_llm_show_options": config["plugins"]["user_llm"].get("show_options", True),
        "user_llm_persona": config["plugins"]["user_llm"].get("persona", "veteran_attending"),
        "user_llm_information_sparsity": config["plugins"]["user_llm"].get("information_sparsity", "dense"),
        "medical_llm_show_case_info": config["plugins"]["medical_llm"].get("show_case_info", True),
    }
    # Resume support: records_partial.jsonl is written incrementally (one line per case).
    # On crash/restart, load completed records from it and skip those cases.
    partial_path = output_dir / "records_partial.jsonl"
    completed_records: list[dict] = []
    completed_keys: set[str] = set()
    if results_path.exists():
        saved = json.loads(results_path.read_text())
        if saved.get("run_meta") != run_meta:
            print(f"\n  *** {results_path}의 run_meta가 이번 설정과 다릅니다 -- 캐시 없이 전체를 새로 덮어씁니다. ***")
            _confirm("  계속 진행하시겠습니까?")
            partial_path.unlink(missing_ok=True)
    if partial_path.exists():
        for line in partial_path.read_text().splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                # Compat: old records used "checkpoint_results" key; aggregate() expects "checkpoints".
                if "checkpoint_results" in r and "checkpoints" not in r:
                    r["checkpoints"] = r.pop("checkpoint_results")
                completed_records.append(r)
                completed_keys.add(r["case_id"])
        if completed_keys:
            print(f"  Resuming: {len(completed_keys)}/{len(cases)} cases already done, skipping.")

    # final_judge is NOT forced off anymore -- needed as the show_options=false belief
    # fallback (see _resolve_belief). Respect whatever plugins.final_judge.enabled says;
    # it's simply unused (never called) when show_options=true.
    _, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    from core.config import build_agenda_plugins
    agenda_planner, resolution_tracker = build_agenda_plugins(config)
    user_llm_cfg = config["plugins"]["user_llm"]

    condition_label = (
        f"policy={config['plugins']['policy']['type']}, "
        f"doctor_persona={config['plugins']['user_llm'].get('persona', 'veteran_attending')}, "
        f"burden_dropout_threshold={user_llm_cfg.get('burden_dropout_threshold', 'inf')}"
    )
    print(f"Running {len(cases)} case(s) ({condition_label}, max_turns={max_turns})...")

    records: list[dict] = list(completed_records)
    pending = [case_info for case_info in cases if case_info.case_id not in completed_keys]
    with ThreadPoolExecutor(max_workers=config.get("concurrency", 5)) as ex:
        futures = {
            ex.submit(
                run_episode_with_termination, case_info, user_llm_cfg, medical_llm,
                fact_validator_llm, policy, final_judge, config, max_turns, output_dir,
                checkpoints,
                agenda_planner, resolution_tracker,
            ): case_info.case_id
            for case_info in pending
        }
        with open(partial_path, "a") as _pf:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="episodes", unit="case"):
                rec = fut.result()
                records.append(rec)
                _pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                _pf.flush()

    # ai_alone: medical model answers each case without dialogue (one-shot MCQ). Opt-in via
    # experiment.measure_ai_alone: true -- off by default so existing configs are unaffected.
    ai_alone_correct: list[bool] | None = None
    if exp.get("measure_ai_alone", False):
        print("  Loading/measuring ai_alone accuracy (one-shot MCQ, no dialogue)...")
        medical_llm_cfg = config["plugins"]["medical_llm"]
        cache_path = (
            _ROOT / exp["ai_alone_cache"]
            if exp.get("ai_alone_cache")
            else _default_ai_alone_cache_path(medical_llm_cfg)
        )
        ai_alone_by_case = _load_or_measure_ai_alone(
            cases=cases,
            medical_llm_cfg=medical_llm_cfg,
            concurrency=config.get("concurrency", 5),
            cache_path=cache_path,
        )
        # preserve original case order
        ai_alone_correct = [ai_alone_by_case[c.case_id] for c in cases]
        # write per-case result back into records so it's saved in results.json
        for rec in records:
            if rec["case_id"] in ai_alone_by_case:
                rec["ai_alone_correct"] = ai_alone_by_case[rec["case_id"]]
        ai_alone_acc = round(sum(ai_alone_correct) / len(ai_alone_correct), 4)
        print(f"  ai_alone_accuracy: {ai_alone_acc:.4f}")

    scores = aggregate(records, checkpoints, ai_alone_correct=ai_alone_correct)
    print_report(scores, condition_label)

    print("  Token usage:")
    tracker.print_summary()
    print()

    with open(results_path, "w") as f:
        json.dump({"run_meta": run_meta, "scores": scores, "records": records}, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {results_path}")
    partial_path.unlink(missing_ok=True)  # clean up incremental cache on successful completion

    tracker.accumulate_to_ledger(
        _ROOT / exp.get("token_ledger", "token_usage_ledger.json"),
        run_meta={"script": "run_poc", "n_cases": len(cases), "max_turns": max_turns},
    )
    return scores


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _print_sweep_summary(all_scores: dict[str, dict], conditions: list[str]) -> None:
    print()
    print("=" * 96)
    print("  Termination POC persona/info-condition sweep")
    print("=" * 96)
    print(f"\n  {'condition':<40}{'doctor_alone_acc':>18}{'n (true/false)':>16}")
    for cond in conditions:
        s = all_scores[cond]
        last_cp = str(s["all_checkpoints"][-1])
        n_true = s["curve"][True][int(last_cp)]["n"]
        n_false = s["curve"][False][int(last_cp)]["n"]
        print(f"  {cond:<40}{s['doctor_alone_accuracy']:>18.4f}{f'{n_true}/{n_false}':>16}")
    print()


def _run_plot(output_path: Path) -> None:
    import subprocess
    plot_script = _ROOT / "plot" / "code" / "plot_scaling_poc.py"
    print(f"\n  Plotting {output_path} ...")
    result = subprocess.run(
        [sys.executable, str(plot_script), "--mode", "overall", str(output_path)],
        cwd=_ROOT,
    )
    if result.returncode != 0:
        print("  *** plot failed (see above) ***")


def _condition_from_output_leaf(name: str) -> str:
    persona, sep, info_condition = name.rpartition("_")
    return f"{persona}/{info_condition}" if sep and persona and info_condition else name


def _load_summary_from_condition_results(base_name: str) -> dict[str, dict]:
    """Recover summary entries from completed condition subdirs.

    This keeps top-level summary.json cumulative across separate one-persona sweep runs. A
    condition is included only after its results.json exists; records_partial.jsonl alone is
    deliberately ignored because the run did not finish aggregation yet.
    """
    summary_dir = _ROOT / "outputs" / base_name
    existing: dict[str, dict] = {}
    if not summary_dir.exists():
        return existing
    for result_path in sorted(summary_dir.glob("*/results.json")):
        try:
            data = json.loads(result_path.read_text())
        except Exception:
            continue
        scores = data.get("scores")
        if isinstance(scores, dict):
            existing[_condition_from_output_leaf(result_path.parent.name)] = scores
    return existing


def run_sweep(base_config: dict) -> dict[str, dict]:
    """Expands plugins.user_llm.persona/info_condition list(s) into one run() call per
    combination (cross product), prints the combined report, and saves summary.json."""
    import copy
    personas = _as_list(base_config["plugins"]["user_llm"].get("persona", "veteran_attending"))
    info_conditions = _as_list(base_config["plugins"]["user_llm"].get("info_condition", "full"))
    base_name = base_config["experiment"]["name"]

    all_scores: dict[str, dict] = {}
    conditions: list[str] = []
    for persona in personas:
        for info_condition in info_conditions:
            cond = f"{persona}/{info_condition}"
            conditions.append(cond)
            cfg = copy.deepcopy(base_config)
            cfg["plugins"]["user_llm"]["persona"] = persona
            cfg["plugins"]["user_llm"]["info_condition"] = info_condition
            # Ground-truth persona label for this condition, passed through to the policy too --
            # harmless for policies that ignore it; consumed by e.g. ActionSpaceV3OraclePolicy's
            # reveal_user_state mode (needs the persona to look up its BS/C Bayes params).
            cfg["plugins"]["policy"]["persona"] = persona
            cfg["experiment"]["name"] = f"{base_name}/{persona}_{info_condition}"
            print(f"\n{'#' * 96}\n# Condition: {cond}\n{'#' * 96}")
            all_scores[cond] = run(cfg)

    _print_sweep_summary(all_scores, conditions)

    summary_dir = _ROOT / "outputs" / base_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    merged_scores: dict[str, dict] = {}
    summary_path = summary_dir / "summary.json"
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text())
            if isinstance(previous, dict):
                merged_scores.update(previous)
        except json.JSONDecodeError:
            print(f"  WARNING: ignoring invalid existing summary: {summary_path}")
    merged_scores.update(_load_summary_from_condition_results(base_name))
    merged_scores.update(all_scores)
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(merged_scores, f, indent=2, ensure_ascii=False)
    print(f"  Combined summary saved to {summary_dir / 'summary.json'}")

    _run_plot(summary_dir)
    return merged_scores


def main(config_path: str, concurrency: int | None = None) -> None:
    config: dict = load_yaml(config_path)
    if concurrency is not None:
        config["concurrency"] = concurrency

    user_llm_cfg = config["plugins"]["user_llm"]
    if any(isinstance(user_llm_cfg.get(f), list) for f in ("persona", "information_sparsity", "info_condition")):
        run_sweep(config)
        return

    scores = run(config)
    output_dir = _ROOT / "outputs" / config["experiment"]["name"]
    _run_plot(output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.concurrency)
