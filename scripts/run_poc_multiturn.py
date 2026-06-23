#!/usr/bin/env python3
"""
POC: HAC Policy -> Multi-turn Dialogue Outcome, via the production Env.

Drives the same multi-turn engine the GRPO main experiment uses
(core/environment.py MedicalHACEnvironment + plugins/*) for a 3-condition comparison:
  - baseline:      policy=naive,  medical_llm.frame_style=command
  - HAC-command:   policy=rule,   medical_llm.frame_style=command
  - HAC-reference:  policy=rule,   medical_llm.frame_style=reference

(policy=rule, not oracle: RewardOraclePolicy does full ctx-aware argmax over align_reward's
BASE table every turn -- that's solving the exact decision problem a trained policy is
meant to learn, so using it here as "the HAC policy" would be testing a ceiling, not a
deployable rule-based baseline. RulePolicy's relation->action map was corrected instead
(see plugins/policy/rule_policy.py) so it no longer leaves reward on the table for no
reason, while staying context-blind/deployable.)

Each condition runs n_cases x {error, correct} episodes (EpisodeConfig.initial_fact) via
env.run_episode(), up to experiment.max_turns. AI-alone baseline (no dialogue) computed
separately for complementarity.

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    python scripts/run_poc_multiturn.py --config configs/poc_multiturn.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import build_plugins
from core.environment import MedicalHACEnvironment
from core.json_utils import safe_json_load
from core.schemas import CaseInfo
from core.token_tracker import tracker
from core.schemas import EpisodeConfig
from plugins.user_llm.user_simulator.v2 import UserSimulatorV2

_CONDITIONS = (
    ("baseline", "naive", "command"),
    ("hac_command", "rule", "command"),
    ("hac_reference", "rule", "reference"),
)
_CONDITION_NAMES = tuple(c[0] for c in _CONDITIONS)
_CONDITION_LABELS = {"baseline": "Baseline", "hac_command": "HAC-command", "hac_reference": "HAC-reference"}
_MODES = ("error", "correct")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_cases(data_dir: Path, n: int) -> list[dict]:
    cases: list[dict] = []
    for path in sorted(data_dir.glob("jama_raw_*.json")):
        with open(path) as f:
            cases.extend(json.load(f))
    return cases[:n]


def to_case_info(case: dict) -> CaseInfo:
    return CaseInfo(
        case_id=case["case_id"],
        scenario=case["scenario"],
        options=case["options"],
        correct_option=case["correct_option"],
        answer=case["answer"],
        distractors=case.get("distractors", []),
        caption=case.get("caption"),
        explanation=case.get("explanation"),
    )


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


def _normalize_option(value: Any, options: dict[str, str]) -> str | None:
    letter = str(value or "").strip().upper()
    return letter if letter in options else None


# ── AI-alone baseline (no dialogue; complementarity reference point) ───────────

_ALONE_SYSTEM = "You are a medical AI assistant. Decide the case independently — you are not shown any other opinion."

_ALONE_USER = """\
Case:
{scenario_text}

Image Explanation:
{image_caption}

[Question]
{question}

Options:
{options_text}

Respond in JSON: {{"selected_option": "<option letter>", "reasoning": "<1-2 sentences>"}}\
"""


def simulate_ai_alone(case: dict, client: OpenAI, model: str) -> dict:
    options_text = format_options(case["options"])
    user = _ALONE_USER.format(
        scenario_text=case["scenario"],
        image_caption=case.get("caption") or "N/A",
        question="What is the most appropriate next step in management?",
        options_text=options_text,
    )
    messages = [
        {"role": "system", "content": _ALONE_SYSTEM},
        {"role": "user", "content": user},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    tracker.record(model, messages, text, response.usage)
    data = safe_json_load(text)
    selected = _normalize_option(data.get("selected_option"), case["options"])
    return {
        "selected_option": selected,
        "reasoning": str(data.get("reasoning", "")),
        "alone_correct": selected == case["correct_option"],
    }


# ── Condition plugin wiring ──────────────────────────────────────────────────────

def build_condition_plugins(base_config: dict, policy_type: str, frame_style: str):
    """Deep-copy the base config, override policy/frame_style, build the (stateless) shared
    plugins. user_llm is intentionally NOT reused — each episode gets its own (see run_episode_task),
    since it carries per-episode state and Env/episodes run concurrently across threads."""
    cfg = copy.deepcopy(base_config)
    cfg["plugins"]["policy"]["type"] = policy_type
    cfg["plugins"]["medical_llm"]["frame_style"] = frame_style
    _, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(cfg)
    return cfg, medical_llm, fact_validator_llm, policy, final_judge


def run_episode_task(
    case: dict,
    mode: str,
    user_llm_cfg: dict,
    medical_llm,
    fact_validator_llm,
    policy,
    final_judge,
    config: dict,
    max_turns: int,
) -> dict:
    case_info = to_case_info(case)
    user_llm = UserSimulatorV2(user_llm_cfg)
    env = MedicalHACEnvironment(user_llm, medical_llm, fact_validator_llm, policy, config, final_judge)
    episode_config = EpisodeConfig(initial_fact="correct" if mode == "correct" else "incorrect")
    steps = env.run_episode(case_info, max_turns=max_turns, episode_config=episode_config)

    if steps:
        last_meta = steps[-1].metadata
        final_judgement = last_meta.get("final_judgement")
        closed_by = last_meta.get("closed_by")
    else:
        # User simulator reached agreement on its very first turn — no policy step was taken.
        final_judgement = env._finalize()
        closed_by = "agreement"

    rewards = [s.reward for s in steps if s.reward is not None]
    return {
        "case_id": case_info.case_id,
        "mode": mode,
        "n_turns": len(steps),
        "closed_by": closed_by,
        "final_judgement": final_judgement,
        "mean_r_align": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "steps": [
            {
                "turn": s.turn_id,
                "user_utterance": s.user_utterance,
                "medical_response": s.medical_response,
                "action_id": s.selected_action,
                "action_prompt": s.action_prompt,
                "r_align": s.reward,
            }
            for s in steps
        ],
        "_complete": True,
    }


# ── Cache (reuse a previous results.json under the same model/max_turns/data condition) ──

def _load_cache(results_path: Path, run_meta: dict) -> tuple[dict[tuple[str, str, str], dict], dict[str, dict]]:
    if not results_path.exists():
        return {}, {}
    try:
        data = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, {}
    if data.get("run_meta") != run_meta:
        return {}, {}
    rec_cache = {
        (r["condition"], r["case_id"], r["mode"]): r
        for r in data.get("records", []) if r.get("_complete")
    }
    return rec_cache, data.get("alone", {})


# ── Scoring ───────────────────────────────────────────────────────────────────

def _mean(values: list[bool]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def compute_scores(records: list[dict], alone_by_case: dict[str, dict]) -> dict:
    by_cond: dict[str, dict[str, Any]] = {
        c: {"all": [], "error": [], "correct": [], "turns": [], "agreement": 0, "max_turns": 0}
        for c in _CONDITION_NAMES
    }
    for r in records:
        d = by_cond[r["condition"]]
        fj = r.get("final_judgement")
        is_correct = bool(fj["is_correct"]) if fj else False
        d["all"].append(is_correct)
        d["error" if r["mode"] == "error" else "correct"].append(is_correct)
        d["turns"].append(r["n_turns"])
        if r["closed_by"] == "agreement":
            d["agreement"] += 1
        else:
            d["max_turns"] += 1

    ai_alone_accuracy = _mean([a["alone_correct"] for a in alone_by_case.values()])
    result: dict[str, Any] = {"ai_alone_accuracy": ai_alone_accuracy}
    for cond in _CONDITION_NAMES:
        d = by_cond[cond]
        n = len(d["all"])
        team_accuracy = _mean(d["all"])
        result[cond] = {
            "team_accuracy": team_accuracy,
            "team_accuracy_error_mode": _mean(d["error"]),
            "team_accuracy_correct_mode": _mean(d["correct"]),
            "complementarity_gain": round(team_accuracy - ai_alone_accuracy, 4),
            "avg_turns": round(sum(d["turns"]) / n, 2) if n else 0.0,
            "agreement_rate": round(d["agreement"] / n, 4) if n else 0.0,
            "n": n,
        }
    return result


def print_report(scores: dict) -> None:
    print()
    print("=" * 86)
    print("  Multi-turn POC: Baseline vs HAC (command vs reference) — production Env")
    print("=" * 86)
    print(f"  AI-alone accuracy: {scores['ai_alone_accuracy']:.4f}")
    print()
    print(f"  {'Condition':<14} {'team_acc':>9} {'err_mode':>9} {'corr_mode':>9} {'gain':>8} {'avg_turns':>9} {'agree%':>7}  n")
    print(f"  {'-'*14} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*7}  -")
    for cond in _CONDITION_NAMES:
        s = scores[cond]
        print(f"  {_CONDITION_LABELS[cond]:<14} {s['team_accuracy']:>9.4f} {s['team_accuracy_error_mode']:>9.4f} "
              f"{s['team_accuracy_correct_mode']:>9.4f} {s['complementarity_gain']:>+8.4f} {s['avg_turns']:>9.2f} "
              f"{s['agreement_rate']*100:>6.1f}%  {s['n']}")
    print()
    print("  err_mode = 의사가 틀렸을 때 team이 정답으로 수렴한 비율 (자가교정)")
    print("  corr_mode = 의사가 맞았을 때 team이 정답을 지킨 비율")
    print("  gain = team_acc - AI-alone accuracy, agree% = max_turns 대신 합의로 끝난 비율")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    config: dict = yaml.safe_load(Path(config_path).read_text())

    data_dir = _ROOT / config["data_dir"]
    output_dir = _ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    max_turns = int(config["experiment"]["max_turns"])
    model = config["plugins"]["medical_llm"]["model"]

    cases = load_cases(data_dir, config["n_cases"])
    print(f"Loaded {len(cases)} cases from {data_dir}")

    run_meta = {"model": model, "max_turns": max_turns, "data_dir": config["data_dir"]}
    results_path = output_dir / "results.json"
    rec_cache, alone_cache = _load_cache(results_path, run_meta)
    if rec_cache:
        print(f"Found cached run with matching model/max_turns/data_dir ({len(rec_cache)} records) — reusing where complete.")

    # AI-alone baseline
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    alone_client = OpenAI(api_key=api_key, base_url=config["plugins"]["medical_llm"]["base_url"])

    def alone_for_case(case: dict) -> tuple[str, dict]:
        cid = case["case_id"]
        cached = alone_cache.get(cid)
        if cached and cached.get("selected_option") is not None:
            return cid, cached
        return cid, simulate_ai_alone(case, alone_client, model)

    n_alone_reused = sum(
        1 for c in cases
        if alone_cache.get(c["case_id"], {}).get("selected_option") is not None
    )
    print(f"AI-alone: {len(cases) - n_alone_reused} new, {n_alone_reused} reused from cache...")
    alone_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=config.get("alone_concurrency", 5)) as ex:
        futures = [ex.submit(alone_for_case, c) for c in cases]
        for fut in as_completed(futures):
            cid, res = fut.result()
            alone_results[cid] = res

    # Per-condition episodes
    all_records: list[dict] = []
    for condition, policy_type, frame_style in _CONDITIONS:
        cfg, medical_llm, fact_validator_llm, policy, final_judge = build_condition_plugins(
            config, policy_type, frame_style
        )
        user_llm_cfg = cfg["plugins"]["user_llm"]

        plan: list[tuple] = []
        for case in cases:
            for mode in _MODES:
                key = (condition, case["case_id"], mode)
                cached = rec_cache.get(key)
                plan.append((case, mode, cached))

        to_run = [(case, mode) for case, mode, cached in plan if cached is None]
        print(f"[{condition}] {len(to_run)} new episodes, {len(plan) - len(to_run)} reused from cache "
              f"(policy={policy_type}, frame_style={frame_style})...")

        new_by_key: dict[tuple[str, str], dict] = {}
        with ThreadPoolExecutor(max_workers=config.get("concurrency", 5)) as ex:
            futures = {
                ex.submit(
                    run_episode_task, case, mode, user_llm_cfg, medical_llm,
                    fact_validator_llm, policy, final_judge, cfg, max_turns,
                ): (case["case_id"], mode)
                for case, mode in to_run
            }
            for fut in as_completed(futures):
                case_id, mode = futures[fut]
                rec = fut.result()
                rec["condition"] = condition
                new_by_key[(case_id, mode)] = rec

        for case, mode, cached in plan:
            if cached is not None:
                all_records.append(cached)
            else:
                all_records.append(new_by_key[(case["case_id"], mode)])

    scores = compute_scores(all_records, alone_results)
    print_report(scores)

    print("  Token usage:")
    tracker.print_summary()
    print()

    with open(results_path, "w") as f:
        json.dump(
            {"run_meta": run_meta, "scores": scores, "alone": alone_results, "records": all_records},
            f, indent=2, ensure_ascii=False,
        )
    print(f"  Results saved to {results_path}")

    tracker.accumulate_to_ledger(
        _ROOT / "token_usage_ledger.json",
        run_meta={"script": "run_poc_multiturn", "n_cases": len(cases), "model": model, "max_turns": max_turns},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
