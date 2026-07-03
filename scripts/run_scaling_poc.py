#!/usr/bin/env python3
"""
Scaling POC: does a longer AI-doctor dialogue self-correct an initial error, or lock it in?

Runs one episode per case (policy is whatever `plugins.policy.type` is configured to --
oracle and naive are both used as A/B conditions) at the full `max_turns` budget, using the
primitive v1 doctor simulator (plugins/user_llm/user_simulator/v1.py, ported from MedCOBE_EMNLP). The
resulting dialogue is then re-scored by the production `final_judge` on the history TRUNCATED
at each turn checkpoint, to get an "if we'd stopped here" verdict. This gives the
within-episode scaling curve at ~1x episode cost instead of running max_turns in {1,2,4,8} as
4 separate (and noisier, cross-run) episodes.

Unlike v2 (the heavier simulator), v1's initial belief is NOT injected -- the doctor
independently judges the case on turn 0 (no hint), so there is no forced correct/incorrect
"mode": whether a case's doctor started right or wrong emerges from that self-judgment, read
straight off turn 0's self-reported `belief`. This also gives a natural "doctor-alone
accuracy" baseline (parallel to `ai_alone_accuracy` in scripts/run_poc_multiturn.py).

cognitive_burden is scored by the v1 plugin itself (per AI turn, attached to the following
doctor turn's user_state) -- this script just reads it back off each turn instead of
re-deriving it via a second, duplicate set of judge calls.

Policy is set via plugins.policy.type in the config (e.g. oracle = ctx-aware argmax ceiling,
naive = no action-space/r_align influence at all) -- not hardcoded by this script.

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    python scripts/run_scaling_poc.py --config configs/scaling_poc.yaml
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import build_plugins, load_yaml
from core.environment import MedicalHACEnvironment
from core.schemas import CaseInfo, DialogueHistory, DialogueTurn
from core.token_tracker import tracker
from plugins.user_llm.user_simulator.v1 import UserSimulatorV1

from scripts.run_dialogue import load_dotenv  # avoids needing `source .env` before running

_VALID_INFO_CONDITIONS = ("full", "dense", "sparse")


def _resolve_info_condition(config: dict) -> None:
    """Translate plugins.user_llm.info_condition (full|dense|sparse), if set, into the two
    low-level flags the plugins actually read -- medical_llm.show_case_info and
    user_llm.information_sparsity -- so callers can sweep one named 3-value axis instead of
    juggling two separately-owned flags (see persona/information_sparsity.yaml's 3-condition
    note: full = AI gets case_info.scenario directly, no real info asymmetry; dense/sparse =
    AI does NOT get it directly, doctor relays everything vs. withholds). Mutates config in
    place. No-op if info_condition isn't set (older configs that set show_case_info /
    information_sparsity directly keep working unchanged)."""
    cond = config["plugins"]["user_llm"].get("info_condition")
    if cond is None:
        return
    if cond not in _VALID_INFO_CONDITIONS:
        raise ValueError(f"plugins.user_llm.info_condition must be one of {_VALID_INFO_CONDITIONS}, got {cond!r}")
    if cond == "full":
        config["plugins"]["medical_llm"]["show_case_info"] = True
        # Doesn't matter to the AI's knowledge in "full" mode (just doctor verbosity style) --
        # "dense" is the natural default rather than leaving it unset.
        config["plugins"]["user_llm"]["information_sparsity"] = "dense"
    else:
        config["plugins"]["medical_llm"]["show_case_info"] = False
        config["plugins"]["user_llm"]["information_sparsity"] = cond


def _load_balanced_cases(data_dir: Path, n_cases: int) -> tuple[list[dict], dict[str, int]]:
    """Load every data_dir/jama_raw_*.json ("specialty" files -- specialty name = filename
    minus the jama_raw_ prefix and .json suffix) and pick n_cases total, divided as evenly as
    possible across all specialty files instead of pulling all of them from one (the old
    single-data_path behavior always meant "cardiology only", since that's the only file any
    config pointed at). Each returned raw case dict gets metadata.specialty stamped on it.

    Deterministic, no shuffling -- matches the existing single-file behavior's plain
    raw_cases[:n_cases] slice (experiment.seed isn't used for sampling here either way).
    Files are processed in alphabetical order by specialty name so the remainder-from-division
    and any shortfall redistribution are reproducible, not glob/OS-order dependent.
    """
    files = sorted(data_dir.glob("jama_raw_*.json"))
    if not files:
        raise FileNotFoundError(f"No jama_raw_*.json files found under {data_dir}")
    specialties = [f.stem.removeprefix("jama_raw_") for f in files]
    pools: dict[str, list[dict]] = {}
    for path, specialty in zip(files, specialties):
        raw = json.loads(path.read_text())
        pools[specialty] = raw if isinstance(raw, list) else [raw]

    n = len(specialties)
    base, remainder = divmod(n_cases, n)
    # Alphabetically-first `remainder` specialties get one extra -- see plan's example:
    # n_cases=20, 7 specialties -> first 6 get 3, last (otolaryngology) gets 2.
    quota = {s: base + (1 if i < remainder else 0) for i, s in enumerate(specialties)}

    picked: dict[str, list[dict]] = {s: [] for s in specialties}
    shortfall = 0
    for specialty in specialties:
        available = pools[specialty]
        take = min(quota[specialty], len(available))
        picked[specialty] = available[:take]
        shortfall += quota[specialty] - take

    # Redistribute any shortfall (a specialty's pool smaller than its quota) round-robin to
    # specialties that still have spare, unpicked cases, in the same alphabetical order.
    while shortfall > 0:
        progressed = False
        for specialty in specialties:
            if shortfall == 0:
                break
            already = len(picked[specialty])
            available = pools[specialty]
            if already < len(available):
                picked[specialty].append(available[already])
                shortfall -= 1
                progressed = True
        if not progressed:
            print(f"  *** Warning: requested {n_cases} cases but only "
                  f"{sum(len(v) for v in pools.values())} exist across all specialties -- "
                  f"using all of them. ***")
            break

    raw_cases: list[dict] = []
    specialty_counts: dict[str, int] = {}
    for specialty in specialties:
        cases = picked[specialty]
        specialty_counts[specialty] = len(cases)
        for c in cases:
            c = dict(c)
            c["metadata"] = {**c.get("metadata", {}), "specialty": specialty}
            raw_cases.append(c)
    return raw_cases, specialty_counts


def _state(turn: DialogueTurn) -> dict:
    return turn.user_state.model_dump() if turn.user_state is not None else {}


def _belief_correct(belief, gold) -> bool:
    """True iff the persona's self-reported belief names the gold option (A–D letter match).

    Formats the persona's OWN belief to a letter; it does not decide anything for the persona.
    """
    s = str(belief or "").strip().upper()
    g = str(gold or "").strip().upper()
    if s in ("A", "B", "C", "D"):
        letter = s
    else:
        m = re.search(r"(?<![A-Z])([A-D])(?![A-Z])", s)   # standalone A–D token
        letter = m.group(1) if m else ""
    return letter != "" and letter == g


# ── Episode + checkpoint scoring ────────────────────────────────────────────────────────

def run_episode_with_checkpoints(
    case_info: CaseInfo,
    user_llm_cfg: dict,
    medical_llm,
    fact_validator_llm,
    policy,
    final_judge,
    config: dict,
    max_turns: int,
    checkpoints: list[int],
    output_dir: Path,
) -> dict:
    user_llm = UserSimulatorV1(user_llm_cfg)
    env = MedicalHACEnvironment(user_llm, medical_llm, fact_validator_llm, policy, config, final_judge)
    # Also write a RolloutLogger-format .jsonl per case, alongside the custom results.json
    # below -- app.py's rollout viewer only globs outputs/**/*.jsonl, so without this the
    # scaling POC's dialogues are invisible there even though results.json has everything.
    rollout_path = output_dir / "rollouts" / f"{case_info.case_id}.jsonl"
    steps = env.run_episode(case_info, max_turns=max_turns, output_path=rollout_path)

    history = env.history
    turns = history.turns if history is not None else []
    n_turns_actual = len(steps)  # number of completed (doctor, AI) action pairs

    closed_by = steps[-1].metadata.get("closed_by") if steps else "agreement"

    # turns[0] is always the doctor's independent opening judgment (no AI input yet) --
    # this IS the "doctor-alone" data point. Normally read straight off its self-reported
    # belief; when user_llm.show_options is false the doctor never sees lettered options and
    # so never tags one -- fall back to final_judge mapping the free-text opening statement
    # onto the real options, mirroring how medical_llm's correctness is judged (it never
    # tags a belief either).
    opening_belief = _state(turns[0]).get("belief") if turns else None
    if opening_belief is not None:
        alone_correct = opening_belief == case_info.correct_option
    elif turns:
        opening_judgement = final_judge.judge(
            case_info, DialogueHistory(case_id=case_info.case_id, turns=turns[:1]),
        )
        alone_correct = (
            opening_judgement.concluded_option.strip().upper()
            == str(case_info.correct_option).strip().upper()
        )
    else:
        alone_correct = False

    # user_k's state carries the cognitive_burden score for medical_{k-1}
    # (see UserSimulatorV1._followup_turn) -- cumulative_burden(c) sums user_1..user_c.
    # Exception: if the episode ended by hitting max_turns right after the LAST AI turn, no
    # user_{n_turns_actual} turn was ever generated (the doctor plugin never got called
    # again), so that final AI turn's burden was never scored as a side effect -- score it
    # directly instead of leaving it silently missing.
    burden_by_user_turn: dict[int, float] = {}
    burden_n_ok_total = 0
    burden_n_attempted_total = 0
    for k in range(1, n_turns_actual + 1):
        if 2 * k >= len(turns):
            continue
        st = _state(turns[2 * k])
        burden_by_user_turn[k] = st.get("cognitive_burden", 0.0)
        burden_n_ok_total += st.get("cognitive_burden_n_ok", 0)
        burden_n_attempted_total += st.get("cognitive_burden_n_attempted", 0)
    if n_turns_actual not in burden_by_user_turn and n_turns_actual > 0:
        last_burden, last_n_ok, last_n_attempted = user_llm.score_burden_for_last_turn(case_info, history)
        burden_by_user_turn[n_turns_actual] = last_burden
        burden_n_ok_total += last_n_ok
        burden_n_attempted_total += last_n_attempted
    # r_align per AI turn (steps[i].reward = align_reward for medical_i) + the fact-validator
    # relation it was scored against -- kept so r_align's validity as a proxy for final
    # correctness can be checked post-hoc (scripts/analyze_r_align_validity.py), independent
    # of whether the oracle policy (which argmaxes r_align, not final correctness) actually
    # leads anywhere useful.
    r_align_by_turn = {i: s.reward for i, s in enumerate(steps)}
    relation_by_turn = {i: s.verification_template.overall_relation for i, s in enumerate(steps)}

    checkpoint_results: dict[int, dict] = {
        0: {
            "is_correct": alone_correct, "cumulative_burden": 0.0, "avg_burden": 0.0,
            "cumulative_r_align": 0.0, "mean_r_align": 0.0, "self_reported_belief": opening_belief,
        },
    }
    for c in checkpoints:
        c_eff = min(c, n_turns_actual)  # episode may have closed before reaching checkpoint c
        if c_eff == 0:
            checkpoint_results[c] = dict(checkpoint_results[0])
            continue
        truncated_turns = turns[: 2 * c_eff]
        # Final decision at this checkpoint = the clinician persona's OWN last self-reported belief
        # in the truncated history (the position it holds after the deliberation up to here) — not a
        # judge re-reading the transcript. Structurally dialogue-dependent, and no extra LLM call.
        self_belief = next(
            (_state(t).get("belief") for t in reversed(truncated_turns) if t.speaker == "user"),
            opening_belief,
        )
        is_correct = _belief_correct(self_belief, case_info.correct_option)
        cum_burden = sum(burden_by_user_turn[k] for k in range(1, c_eff + 1))
        r_aligns_so_far: list[float] = []
        for i in range(c_eff):
            r = r_align_by_turn[i]
            if r is not None:
                r_aligns_so_far.append(r)
        cum_r_align = sum(r_aligns_so_far)
        checkpoint_results[c] = {
            "is_correct": is_correct,
            "cumulative_burden": round(cum_burden, 4),
            "avg_burden": round(cum_burden / c_eff, 4),
            "cumulative_r_align": round(cum_r_align, 4),
            "mean_r_align": round(cum_r_align / c_eff, 4),
            "self_reported_belief": self_belief,
        }

    return {
        "case_id": case_info.case_id,
        "specialty": case_info.metadata.get("specialty"),  # None unless loaded via _load_balanced_cases
        "alone_correct": alone_correct,
        "n_turns_actual": n_turns_actual,
        "closed_by": closed_by,
        "burden_judge_calls_ok": burden_n_ok_total,
        "burden_judge_calls_attempted": burden_n_attempted_total,
        "checkpoints": {str(c): v for c, v in checkpoint_results.items()},
        "turns": [
            {
                "speaker": t.speaker,
                "text": t.text,
                "action": t.action,
                "user_state": _state(t) or None,
                # medical turn i is at index 2i+1; r_align/relation scored that action.
                "r_align": r_align_by_turn.get((idx - 1) // 2) if t.speaker == "medical" else None,
                "relation": relation_by_turn.get((idx - 1) // 2) if t.speaker == "medical" else None,
            }
            for idx, t in enumerate(turns)
        ],
    }


# ── Aggregation ──────────────────────────────────────────────────────────────────────────

_TRAJECTORY_CLASSES = ("always_correct", "self_corrected", "locked_wrong", "regressed")
_ALONE_BUCKETS = (True, False)  # alone_correct


def _classify_trajectory(first_correct: bool, last_correct: bool) -> str:
    if first_correct and last_correct:
        return "always_correct"
    if (not first_correct) and last_correct:
        return "self_corrected"
    if (not first_correct) and (not last_correct):
        return "locked_wrong"
    return "regressed"


def aggregate(records: list[dict], checkpoints: list[int]) -> dict:
    all_cps = [0] + list(checkpoints)
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
        sum(1 for r in records if r["alone_correct"]) / len(records), 4
    ) if records else 0.0

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
    burden_calls_ok = sum(r.get("burden_judge_calls_ok", 0) for r in records)
    burden_calls_attempted = sum(r.get("burden_judge_calls_attempted", 0) for r in records)

    _reasons = ["agreement", "burden_dropout", "max_turns"]
    by_closed_by: dict[str, dict] = {}
    for reason in _reasons:
        rows = [r for r in records if r.get("closed_by") == reason]
        by_closed_by[reason] = {
            "n": len(rows),
            "rate": round(len(rows) / len(records), 4) if records else 0.0,
            "avg_n_turns": round(statistics.mean(r["n_turns_actual"] for r in rows), 2) if rows else None,
        }
    avg_n_turns = round(statistics.mean(r["n_turns_actual"] for r in records), 2) if records else 0.0

    return {
        "all_checkpoints": all_cps,
        "doctor_alone_accuracy": doctor_alone_accuracy,
        "curve": curve,
        "trajectory_counts": trajectory_counts,
        "burden_by_trajectory": burden_by_trajectory,
        "burden_judge_calls_ok": burden_calls_ok,
        "burden_judge_calls_attempted": burden_calls_attempted,
        "by_closed_by": by_closed_by,
        "avg_n_turns": avg_n_turns,
    }


def print_report(scores: dict, condition_label: str) -> None:
    all_cps = scores["all_checkpoints"]
    print()
    print("=" * 86)
    print(f"  Scaling POC: self-correction vs. lock-in across turn checkpoints ({condition_label})")
    print("=" * 86)
    print(f"\n  doctor_alone_accuracy (turn-0 self-judgment, no AI input): {scores['doctor_alone_accuracy']:.4f}")
    ok, attempted = scores["burden_judge_calls_ok"], scores["burden_judge_calls_attempted"]
    if attempted and ok < attempted:
        print(f"  *** burden judge: {attempted - ok}/{attempted} calls FAILED to parse (silently scored as missing data, not 0) ***")
    else:
        print(f"  burden judge: {ok}/{attempted} calls parsed OK")
    for bucket in _ALONE_BUCKETS:
        label = "alone_correct" if bucket else "alone_incorrect"
        n = scores["curve"][bucket][all_cps[0]]["n"]
        print(f"\n  {label}  (n={n})")
        header = "  checkpoint:      " + "  ".join(f"{c:>6}" for c in all_cps)
        print(header)
        acc_row = "  accuracy:        " + "  ".join(f"{scores['curve'][bucket][c]['accuracy']:>6.3f}" for c in all_cps)
        burden_row = "  avg_cum_burden:  " + "  ".join(
            f"{scores['curve'][bucket][c]['avg_cumulative_burden']:>6.3f}" for c in all_cps
        )
        print(acc_row)
        print(burden_row)
    print("\n  trajectory counts (first checkpoint -> last checkpoint):")
    for bucket in _ALONE_BUCKETS:
        label = "alone_correct" if bucket else "alone_incorrect"
        tc = scores["trajectory_counts"][bucket]
        print(f"    {label:<16}  " + "  ".join(f"{cls}={tc[cls]}" for cls in _TRAJECTORY_CLASSES))
    print("\n  mean final cumulative_burden by trajectory class:")
    for cls in _TRAJECTORY_CLASSES:
        v = scores["burden_by_trajectory"][cls]
        print(f"    {cls:<16}  {v if v is not None else 'n/a'}")
    print()
    print("  self_corrected = wrong at first checkpoint, correct by last (real signal we want)")
    print("  locked_wrong   = wrong at every checkpoint (early error never escaped)")
    print("  regressed      = correct at first checkpoint, wrong by last (longer dialogue hurt)")
    print()


def _confirm(message: str) -> None:
    """Never silently clobber/reuse a previous run's data without asking -- this always runs
    before any LLM call happens, so aborting costs nothing."""
    answer = input(f"{message} [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("  중단합니다 (아무 API 호출도 하지 않았습니다).")
        sys.exit(1)


def _load_cache(results_path: Path, run_meta: dict) -> dict[str, dict]:
    """Reuse a previous results.json's per-case records IF run_meta matches exactly (same
    model/max_turns/checkpoints/data_path). Note this does NOT detect plugin-code changes
    (e.g. a v1.py bug fix) -- the caller still confirms with the user before actually
    skipping any case, since cached results could be stale from before such a fix."""
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if data.get("run_meta") != run_meta:
        return {}
    return {r["case_id"]: r for r in data.get("records", [])}


# ── Main ──────────────────────────────────────────────────────────────────────────────────

def run(config: dict) -> dict:
    """Run one scaling-POC condition from an already-loaded config dict. Returns the
    aggregated `scores` (see aggregate()) so callers sweeping multiple conditions (e.g.
    scripts/run_persona_sweep.py) can collect them without re-reading results.json."""
    load_dotenv()
    # `tracker` is a process-wide singleton -- when run() is called multiple times in one
    # process (run_persona_sweep.py), its summary()/print_summary() would otherwise report
    # the CUMULATIVE total since process start, not this condition's own usage, and
    # accumulate_to_ledger() would double-count every prior condition's calls into the
    # persistent ledger each time. Reset here so each run() call is self-contained.
    tracker.reset()
    exp = config["experiment"]

    for field in ("persona", "information_sparsity", "info_condition"):
        if isinstance(config["plugins"]["user_llm"].get(field), list):
            raise ValueError(
                f"plugins.user_llm.{field} is a list -- run() handles exactly one value per "
                "call. Use scripts/run_persona_sweep.py to sweep lists (it reads these same "
                "fields and calls run() once per combination)."
            )
    _resolve_info_condition(config)

    output_dir = _ROOT / "outputs" / exp["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Dump the exact config this run used, in full -- run_meta below only records a curated
    # subset of fields (the ones cache-invalidation cares about); this is for a human to
    # check "what conditions did this run actually use" without having to reverse-engineer it
    # from run_meta. Written upfront (before any LLM call) so it's there even if interrupted.
    with open(output_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    max_turns = int(exp["max_turns"])
    checkpoints = list(exp["checkpoints"])

    data_dir = exp.get("data_dir")
    data_path = exp.get("data_path")
    assert not (data_dir and data_path), (
        "experiment.data_dir and experiment.data_path are mutually exclusive -- "
        "data_dir balances n_cases across every specialty file in that directory, "
        "data_path loads exactly one specialty file."
    )
    if data_dir is not None:
        n_cases = exp.get("n_cases")
        assert n_cases is not None, (
            "experiment.n_cases is required with data_dir (it's the total to divide evenly "
            "across specialties)"
        )
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
        "checkpoints": checkpoints,
        "data_path": data_path,
        "data_dir": data_dir,
        "specialty_counts": specialty_counts,  # None when using data_path (single-specialty mode, unchanged)
        "policy_type": config["plugins"]["policy"]["type"],
        # show_options changes the DOCTOR's (user_llm) prompt/behavior materially (it's what
        # this run is A/B testing) -- must invalidate the cache when toggled, not silently reuse.
        "user_llm_show_options": config["plugins"]["user_llm"].get("show_options", True),
        # persona and information_sparsity are the 2 independent persona axes -- both change
        # the doctor's behavior materially -- same cache-invalidation reasoning as show_options.
        "user_llm_persona": config["plugins"]["user_llm"].get("persona", "veteran_attending"),
        "user_llm_information_sparsity": config["plugins"]["user_llm"].get("information_sparsity", "dense"),
        # Whether the AI receives case_info.scenario directly (see core/prompt_builder.py:
        # build_medical_prompt) -- this is what gives information_sparsity real teeth (see
        # persona/information_sparsity.yaml); false vs true is a materially different episode.
        "medical_llm_show_case_info": config["plugins"]["medical_llm"].get("show_case_info", True),
    }

    if results_path.exists() and json.loads(results_path.read_text()).get("run_meta") != run_meta:
        print(f"\n  *** {results_path}의 run_meta가 이번 설정과 다릅니다 (모델/max_turns/checkpoints/data_path 변경됨)")
        print("  *** 캐시를 못 쓰고 전체를 새로 덮어씁니다. ***")
        _confirm("  계속 진행하시겠습니까?")

    cache = _load_cache(results_path, run_meta)
    cached_records = [cache[c.case_id] for c in cases if c.case_id in cache]
    cases_to_run = [c for c in cases if c.case_id not in cache]

    if cached_records:
        print(f"\n  *** 다음 {len(cached_records)}개 케이스는 이미 결과가 있어서 스킵됩니다 (코드가 바뀐 뒤의 옛 결과일 수도 있음): ***")
        for r in cached_records:
            print(f"    - case_id={r['case_id']}")
        _confirm("  스킵됩니다. 진행하시겠습니까?")

    _, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    assert final_judge is not None, "plugins.final_judge.enabled must be true for this experiment"
    user_llm_cfg = config["plugins"]["user_llm"]

    condition_label = (
        f"policy={config['plugins']['policy']['type']}, user_llm={config['plugins']['user_llm']['type']}, "
        f"doctor_show_options={config['plugins']['user_llm'].get('show_options', True)}, "
        f"doctor_persona={config['plugins']['user_llm'].get('persona', 'veteran_attending')}, "
        f"doctor_information_sparsity={config['plugins']['user_llm'].get('information_sparsity', 'dense')}, "
        f"medical_show_case_info={config['plugins']['medical_llm'].get('show_case_info', True)}"
    )
    print(f"Running {len(cases_to_run)} new episode(s), reusing {len(cached_records)} "
          f"({condition_label}, max_turns={max_turns}, checkpoints={checkpoints})...")

    records: list[dict] = list(cached_records)
    with ThreadPoolExecutor(max_workers=config.get("concurrency", 5)) as ex:
        futures = {
            ex.submit(
                run_episode_with_checkpoints, case_info, user_llm_cfg, medical_llm,
                fact_validator_llm, policy, final_judge, config, max_turns, checkpoints, output_dir,
            ): case_info.case_id
            for case_info in cases_to_run
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

    try:
        from plot.code.plot_scaling_poc import plot_overall
        plot_overall(output_dir)
    except Exception as e:
        print(f"  [plot] overall 플롯 생성 실패 (결과는 저장됨): {e}")

    tracker.accumulate_to_ledger(
        _ROOT / exp.get("token_ledger", "token_usage_ledger.json"),
        run_meta={"script": "run_scaling_poc", "n_cases": len(cases), "max_turns": max_turns},
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
    parser.add_argument("--concurrency", type=int, default=None,
                         help="Override the config's top-level concurrency (ThreadPoolExecutor max_workers)")
    args = parser.parse_args()
    main(args.config, args.concurrency)
