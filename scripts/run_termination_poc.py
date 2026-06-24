#!/usr/bin/env python3
"""
Termination POC: runs v3 (the two-stage burden-judge + utterance-generator doctor simulator,
see plugins/user_llm/user_simulator/v3.py) as the REAL multi-turn harness, instead of
scripts/run_scaling_poc.py's "always run to max_turns, then re-judge truncated history at
fixed checkpoints" design. Here the episode actually ends for one of three real reasons:

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
    python scripts/run_termination_poc.py --config configs/termination_poc.yaml
"""
from __future__ import annotations

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import build_plugins, load_episode_configs, load_yaml
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo, DialogueHistory, DialogueTurn
from core.token_tracker import tracker
from plugins.final_judge_llm.base import FinalJudgeLLMPlugin
from plugins.user_llm.user_simulator.v3 import UserSimulatorV3

from scripts.run_dialogue import load_dotenv  # avoids needing `source .env` before running
from scripts.run_scaling_poc import _load_balanced_cases, _resolve_info_condition, _confirm

_CLOSED_BY_REASONS = ("agreement", "burden_dropout", "max_turns")


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
    episode_config_name: str | None,
    episode_config,
    checkpoints: list[int],
) -> dict:
    # Fresh instance per case (not build_plugins's shared one) -- v3 carries per-episode
    # mutable state (_burden_cumulative, _last_belief), so a shared instance would race across
    # ThreadPoolExecutor workers. Mirrors how run_scaling_poc.py does the same for v1.
    user_llm = UserSimulatorV3(user_llm_cfg)
    # final_judge=None here on purpose -- the environment's own _finalize() is unused; THIS
    # script calls final_judge.judge() itself, only as a belief fallback (see _resolve_belief).
    env = MedicalHACEnvironment(user_llm, medical_llm, fact_validator_llm, policy, config, final_judge=None)
    # Disambiguate the rollout filename when sweeping multiple initial_user_state presets
    # over the same cases (cross product) -- otherwise they'd collide on one path.
    rollout_name = f"{case_info.case_id}_{episode_config_name}" if episode_config_name else case_info.case_id
    rollout_path = output_dir / "rollouts" / f"{rollout_name}.jsonl"
    steps = env.run_episode(case_info, max_turns=max_turns, output_path=rollout_path, episode_config=episode_config)

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
        checkpoint_results[c] = {
            "self_reported_belief": belief,
            "is_correct": belief is not None and belief == case_info.correct_option,
        }

    return {
        "case_id": case_info.case_id,
        "initial_user_state": episode_config_name,
        "specialty": case_info.metadata.get("specialty"),
        "n_turns_actual": n_turns_actual,
        "closed_by": closed_by,
        "terminal_belief": terminal_belief,
        "is_correct": is_correct,
        "final_cumulative_burden": last_user_state.get("cognitive_burden_cumulative"),
        "burden_judge_calls_ok": burden_n_ok,
        "burden_judge_calls_attempted": burden_n_attempted,
        "checkpoints": {str(c): v for c, v in checkpoint_results.items()},
    }


# ── Aggregation ──────────────────────────────────────────────────────────────────────────

def aggregate(records: list[dict], checkpoints: list[int]) -> dict:
    n = len(records)
    accuracy = round(sum(1 for r in records if r["is_correct"]) / n, 4) if n else 0.0

    by_closed_by: dict[str, dict] = {}
    for reason in _CLOSED_BY_REASONS:
        rows = [r for r in records if r["closed_by"] == reason]
        burdens = [r["final_cumulative_burden"] for r in rows if r["final_cumulative_burden"] is not None]
        by_closed_by[reason] = {
            "n": len(rows),
            "accuracy": round(sum(1 for r in rows if r["is_correct"]) / len(rows), 4) if rows else None,
            "avg_n_turns": round(statistics.mean(r["n_turns_actual"] for r in rows), 2) if rows else None,
            "avg_final_cumulative_burden": round(statistics.mean(burdens), 2) if burdens else None,
        }

    # Belief-based "what if we'd stopped here" curve, pooled across all records (see
    # run_episode_with_termination's checkpoint_results / _resolve_belief).
    all_cps = [0] + list(checkpoints)
    curve = {}
    for c in all_cps:
        cp_correct = [r["checkpoints"][str(c)]["is_correct"] for r in records if str(c) in r.get("checkpoints", {})]
        curve[c] = {
            "accuracy": round(sum(cp_correct) / len(cp_correct), 4) if cp_correct else 0.0,
            "n": len(cp_correct),
        }

    burden_calls_ok = sum(r.get("burden_judge_calls_ok", 0) for r in records)
    burden_calls_attempted = sum(r.get("burden_judge_calls_attempted", 0) for r in records)
    return {
        "n_cases": n,
        "accuracy": accuracy,
        "by_closed_by": by_closed_by,
        "checkpoint_curve": curve,
        "burden_judge_calls_ok": burden_calls_ok,
        "burden_judge_calls_attempted": burden_calls_attempted,
    }


def print_report(scores: dict, condition_label: str) -> None:
    print()
    print("=" * 86)
    print(f"  Termination POC: real episode termination + belief-based accuracy ({condition_label})")
    print("=" * 86)
    print(f"\n  overall accuracy (n={scores['n_cases']}): {scores['accuracy']:.4f}")
    ok, attempted = scores["burden_judge_calls_ok"], scores["burden_judge_calls_attempted"]
    if attempted and ok < attempted:
        print(f"  *** burden judge: {attempted - ok}/{attempted} calls FAILED to parse ***")
    else:
        print(f"  burden judge: {ok}/{attempted} calls parsed OK")

    cps = sorted(scores["checkpoint_curve"].keys())
    print("\n  belief-based checkpoint curve (final_judge only as show_options=false fallback, "
          "c_eff = min(c, n_turns_actual)):")
    print("  checkpoint:  " + "  ".join(f"{c:>6}" for c in cps))
    print("  accuracy:    " + "  ".join(f"{scores['checkpoint_curve'][c]['accuracy']:>6.3f}" for c in cps))

    print("\n  by closed_by:")
    print(f"  {'reason':<16}{'n':>5}{'accuracy':>10}{'avg_turns':>11}{'avg_final_burden':>18}")
    for reason in _CLOSED_BY_REASONS:
        row = scores["by_closed_by"][reason]
        acc = f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "n/a"
        turns = f"{row['avg_n_turns']:.2f}" if row["avg_n_turns"] is not None else "n/a"
        burden = f"{row['avg_final_cumulative_burden']:.2f}" if row["avg_final_cumulative_burden"] is not None else "n/a"
        print(f"  {reason:<16}{row['n']:>5}{acc:>10}{turns:>11}{burden:>18}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────────────────

def run(config: dict) -> dict:
    """Run one termination-POC condition from an already-loaded config dict. Returns the
    aggregated `scores` so callers sweeping multiple conditions (e.g.
    scripts/run_termination_poc_sweep.py) can collect them without re-reading results.json."""
    load_dotenv()
    tracker.reset()  # see run_scaling_poc.py's run() for why this must be per-call
    exp = config["experiment"]

    for field in ("persona", "information_sparsity", "info_condition"):
        if isinstance(config["plugins"]["user_llm"].get(field), list):
            raise ValueError(
                f"plugins.user_llm.{field} is a list -- run() handles exactly one value per "
                "call. Use scripts/run_termination_poc_sweep.py to sweep lists."
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
        "initial_user_state": exp.get("initial_user_state"),
        "checkpoints": checkpoints,
        "burden_dropout_threshold": config["plugins"]["user_llm"].get("burden_dropout_threshold"),
        "user_llm_show_options": config["plugins"]["user_llm"].get("show_options", True),
        "user_llm_persona": config["plugins"]["user_llm"].get("persona", "burned_out_resident"),
        "user_llm_information_sparsity": config["plugins"]["user_llm"].get("information_sparsity", "dense"),
        "medical_llm_show_case_info": config["plugins"]["medical_llm"].get("show_case_info", True),
    }
    if results_path.exists() and json.loads(results_path.read_text()).get("run_meta") != run_meta:
        print(f"\n  *** {results_path}의 run_meta가 이번 설정과 다릅니다 -- 캐시 없이 전체를 새로 덮어씁니다. ***")
        _confirm("  계속 진행하시겠습니까?")

    # final_judge is NOT forced off anymore -- needed as the show_options=false belief
    # fallback (see _resolve_belief). Respect whatever plugins.final_judge.enabled says;
    # it's simply unused (never called) when show_options=true.
    _, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    user_llm_cfg = config["plugins"]["user_llm"]

    # initial_user_state: None -> [(None, None)] (single default EpisodeConfig); a name/list/
    # "all" -> one (name, EpisodeConfig) per preset, cross-producted with cases below. v3 only
    # records these fields into user_state metadata (doesn't branch prompt logic on them, like
    # v1) -- wired through anyway for template consistency + metadata fidelity.
    episode_configs = load_episode_configs(exp.get("initial_user_state"))

    condition_label = (
        f"policy={config['plugins']['policy']['type']}, "
        f"doctor_persona={config['plugins']['user_llm'].get('persona', 'burned_out_resident')}, "
        f"burden_dropout_threshold={user_llm_cfg.get('burden_dropout_threshold', 'inf')}"
    )
    print(f"Running {len(cases)} case(s) x {len(episode_configs)} initial_user_state preset(s) "
          f"({condition_label}, max_turns={max_turns})...")

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.get("concurrency", 5)) as ex:
        futures = {
            ex.submit(
                run_episode_with_termination, case_info, user_llm_cfg, medical_llm,
                fact_validator_llm, policy, final_judge, config, max_turns, output_dir,
                ep_name, ep_cfg, checkpoints,
            ): (case_info.case_id, ep_name)
            for case_info in cases
            for ep_name, ep_cfg in episode_configs
        }
        for fut in as_completed(futures):
            records.append(fut.result())

    scores = aggregate(records, checkpoints)
    print_report(scores, condition_label)

    print("  Token usage:")
    tracker.print_summary()
    print()

    with open(results_path, "w") as f:
        json.dump({"run_meta": run_meta, "scores": scores, "records": records}, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {results_path}")

    tracker.accumulate_to_ledger(
        _ROOT / exp.get("token_ledger", "token_usage_ledger.json"),
        run_meta={"script": "run_termination_poc", "n_cases": len(cases), "max_turns": max_turns},
    )
    return scores


def main(config_path: str, concurrency: int | None = None) -> None:
    config: dict = load_yaml(config_path)
    if concurrency is not None:
        config["concurrency"] = concurrency
    run(config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.concurrency)
