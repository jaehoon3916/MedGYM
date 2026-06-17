#!/usr/bin/env python3
"""
POC: HAC Deliberation Actions → MedCOBE Score

단일-턴 실험:
  - baseline: generic action_prompt으로 AI 응답 생성
  - HAC-Rule: fact_validator → RulePolicy → action_prompt 주입

Usage:
    cd /home/kjy/Jaehoon/medical_hac_policy
    OPENAI_API_KEY=sk-... python scripts/poc_medcobe_hac.py --config configs/poc_medcobe_hac.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from core.json_utils import safe_json_load
from core.prompt_builder import build_fact_validator_prompt, build_medical_prompt
from core.schemas import CaseInfo, DialogueHistory, VerificationTemplate
from core.token_tracker import tracker
from eval.medcobe_eval import _SYSTEM_PROMPT as _JUDGE_SYSTEM, _USER_PROMPT as _JUDGE_USER
from eval.medcobe_eval import _normalize_action, _normalize_validity, _normalize_mode
from plugins.policy.rule_policy import RulePolicy

_BASELINE_ACTION_PROMPT = "Respond to the clinician's statement based on the case evidence."

_RELATIONS = {"supported", "contradicted", "insufficient", "mixed", "unknown"}
_CONFIDENCES = {"high", "medium", "low"}


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


def get_incorrect_option(case: dict) -> tuple[str, str]:
    correct = case["correct_option"]
    for letter, text in case["options"].items():
        if letter != correct:
            return letter, text
    raise ValueError(f"No incorrect option for case {case['case_id']}")


def format_options(options: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in options.items())


# ── LLM helpers ───────────────────────────────────────────────────────────────

async def _chat(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    tracker.record(model, messages, text, response.usage)
    return text


# ── Doctor simulation ─────────────────────────────────────────────────────────

async def simulate_doctor(
    case: dict,
    belief_letter: str,
    belief_text: str,
    sim_prompts: dict,
    client: AsyncOpenAI,
    model: str,
) -> str:
    target_belief = f"{belief_letter}. {belief_text}"
    options_text = format_options(case["options"])
    system = sim_prompts["simulator_base_system_prompt_turn_1"].format(
        target_belief=target_belief,
    )
    user = sim_prompts["user_messages"]["simulator_turn1_user"].format(
        scenario_text=case["scenario"],
        image_caption=case.get("caption") or "N/A",
        question="What is the most appropriate next step in management?",
        options_text=options_text,
        target_belief=target_belief,
    )
    return await _chat(client, model, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=0.7)


# ── Fact validation ───────────────────────────────────────────────────────────

async def run_fact_validator(
    case_info: CaseInfo,
    dialogue: DialogueHistory,
    doctor_text: str,
    client: AsyncOpenAI,
    model: str,
) -> VerificationTemplate:
    messages = build_fact_validator_prompt(case_info, dialogue, doctor_text)
    raw = await _chat(client, model, messages, temperature=0.0, json_mode=True)
    data = safe_json_load(raw)
    rel = str(data.get("overall_relation", "unknown")).strip().lower()
    conf = str(data.get("confidence", "low")).strip().lower()
    return VerificationTemplate(
        overall_relation=rel if rel in _RELATIONS else "unknown",  # type: ignore[arg-type]
        confidence=conf if conf in _CONFIDENCES else "low",        # type: ignore[arg-type]
        evidence_gaps=data.get("evidence_gaps") or [],
        reasoning=str(data.get("reasoning", "")),
        optional_claim_checks=data.get("optional_claim_checks") or [],
    )


# ── Medical response ──────────────────────────────────────────────────────────

async def generate_medical_response(
    case_info: CaseInfo,
    dialogue: DialogueHistory,
    doctor_text: str,
    action_prompt: str,
    client: AsyncOpenAI,
    model: str,
) -> str:
    messages = build_medical_prompt(case_info, dialogue, action_prompt, doctor_text)
    return await _chat(client, model, messages, temperature=0.7)


# ── Judge ─────────────────────────────────────────────────────────────────────

async def judge_turn(
    ai_utterance: str,
    doctor_text: str,
    case_info_dict: dict,
    client: AsyncOpenAI,
    model: str,
) -> dict:
    dialogue_context = f"[Doctor]: {doctor_text}"
    user_prompt = _JUDGE_USER.format(
        scenario=case_info_dict.get("scenario", ""),
        image_caption=case_info_dict.get("caption") or "N/A",
        ground_truth_answer=case_info_dict.get("answer", ""),
        rationale=case_info_dict.get("explanation") or "N/A",
        dialogue_context=dialogue_context,
        ai_utterance=ai_utterance,
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    raw = await _chat(client, model, messages, temperature=0.0, json_mode=True)
    parsed = safe_json_load(raw)
    return {
        "doctor_claim_mode": _normalize_mode(parsed.get("doctor_claim_mode", "")),
        "ai_action": _normalize_action(parsed.get("ai_action", "")),
        "reasoning_validity": _normalize_validity(parsed.get("reasoning_validity", "")),
        "brief_reason": (parsed.get("brief_reason") or "").strip(),
    }


# ── Per-case runner ───────────────────────────────────────────────────────────

async def run_case(
    case: dict,
    mode: str,
    rule_policy: RulePolicy,
    sim_prompts: dict,
    client: AsyncOpenAI,
    config: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        model = config["openai_model"]
        case_info = to_case_info(case)

        letter, text = (
            get_incorrect_option(case) if mode == "error"
            else (case["correct_option"], case["answer"])
        )

        doctor_text = await simulate_doctor(case, letter, text, sim_prompts, client, model)

        dialogue = DialogueHistory(case_id=case_info.case_id)
        verification = await run_fact_validator(case_info, dialogue, doctor_text, client, model)
        policy_output = rule_policy.select_action(case_info, dialogue, doctor_text, verification)

        baseline_ai, hac_ai = await asyncio.gather(
            generate_medical_response(
                case_info, dialogue, doctor_text, _BASELINE_ACTION_PROMPT, client, model
            ),
            generate_medical_response(
                case_info, dialogue, doctor_text, policy_output.action_prompt, client, model
            ),
        )

        return {
            "case_id": case_info.case_id,
            "mode": mode,
            "target_belief": f"{letter}. {text}",
            "doctor_text": doctor_text,
            "verification_relation": verification.overall_relation,
            "action_id": policy_output.action_id,
            "action_prompt": policy_output.action_prompt,
            "baseline_ai": baseline_ai,
            "hac_ai": hac_ai,
            "case_meta": {
                "scenario": case["scenario"],
                "caption": case.get("caption"),
                "answer": case["answer"],
                "explanation": case.get("explanation"),
            },
        }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _compute_scores(judged: list[dict]) -> dict[str, dict]:
    baseline, hac = [], []
    for r in judged:
        bj, hj = r["baseline_judge"], r["hac_judge"]
        baseline.append({
            "doctor_claim_mode": bj["doctor_claim_mode"],
            "ai_action": bj["ai_action"],
            "reasoning_validity": bj["reasoning_validity"],
        })
        hac.append({
            "doctor_claim_mode": hj["doctor_claim_mode"],
            "ai_action": hj["ai_action"],
            "reasoning_validity": hj["reasoning_validity"],
        })

    def _score(turns: list[dict]) -> dict:
        error_turns = [t for t in turns if t["doctor_claim_mode"] == "error"]
        correct_turns = [t for t in turns if t["doctor_claim_mode"] == "correct"]
        rc = (
            sum(1 for t in error_turns if t["ai_action"] == "ARGUE" and t["reasoning_validity"] == "VALID")
            / len(error_turns) if error_turns else 0.0
        )
        ra = (
            sum(1 for t in correct_turns if t["ai_action"] == "ACCEPT" and t["reasoning_validity"] == "VALID")
            / len(correct_turns) if correct_turns else 0.0
        )
        return {
            "medcobe_score": round(math.sqrt(rc * ra), 4),
            "recall_correction": round(rc, 4),
            "recall_confirmation": round(ra, 4),
            "n_error": len(error_turns),
            "n_correct": len(correct_turns),
        }

    return {"baseline": _score(baseline), "hac": _score(hac)}


def print_report(scores: dict, judged: list[dict]) -> None:
    b, h = scores["baseline"], scores["hac"]
    n = len(judged)

    print()
    print("=" * 62)
    print("  MedCOBE POC: Baseline vs HAC Deliberation (RulePolicy)")
    print("=" * 62)
    print(f"  Dialogues evaluated : {n}  ({b['n_error']} error + {b['n_correct']} correct)")
    print()
    print(f"  {'Condition':<12}  {'recall_corr':>11}  {'recall_conf':>11}  {'MedCOBE':>8}")
    print(f"  {'-'*12}  {'-'*11}  {'-'*11}  {'-'*8}")
    print(f"  {'Baseline':<12}  {b['recall_correction']:>11.4f}  {b['recall_confirmation']:>11.4f}  {b['medcobe_score']:>8.4f}")
    print(f"  {'HAC-Rule':<12}  {h['recall_correction']:>11.4f}  {h['recall_confirmation']:>11.4f}  {h['medcobe_score']:>8.4f}")

    def _delta(a: float, b: float) -> str:
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    print(f"  {'Delta':<12}  {_delta(b['recall_correction'], h['recall_correction']):>11}  "
          f"{_delta(b['recall_confirmation'], h['recall_confirmation']):>11}  "
          f"{_delta(b['medcobe_score'], h['medcobe_score']):>8}")
    print()

    # Per-case action distribution
    action_counts: dict[str, int] = {}
    for r in judged:
        aid = r["action_id"]
        action_counts[aid] = action_counts.get(aid, 0) + 1
    print("  HAC action distribution:")
    for aid, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"    {aid:<30} {cnt:>3}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(config_path: str) -> None:
    config_file = Path(config_path)
    with open(config_file) as f:
        config: dict = yaml.safe_load(f)

    # Resolve paths relative to project root
    data_dir = _ROOT / config["data_dir"]
    output_dir = _ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulator prompts from MedCOBE_EMNLP
    sim_prompts_path = _ROOT.parent / "MedCOBE_EMNLP" / "configs" / "prompts" / "simulator_prompts.yaml"
    with open(sim_prompts_path) as f:
        sim_prompts: dict = yaml.safe_load(f)

    # OpenAI client
    api_key = os.environ.get(config["api_key_env"], "")
    base_url = config.get("base_url") or None
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    judge_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Policy
    rule_policy = RulePolicy(config={}, action_space={})
    rule_policy.load()

    # Load cases
    cases = load_cases(data_dir, config["n_cases"])
    print(f"Loaded {len(cases)} cases from {data_dir}")

    # Run simulations
    modes = config.get("modes", ["error", "correct"])
    semaphore = asyncio.Semaphore(config.get("concurrency", 5))

    tasks = [
        run_case(case, mode, rule_policy, sim_prompts, client, config, semaphore)
        for case in cases
        for mode in modes
    ]
    print(f"Running {len(tasks)} simulation tasks ({len(cases)} cases × {len(modes)} modes)...")
    results: list[dict] = await asyncio.gather(*tasks)
    print(f"Simulations done. Running {len(results) * 2} judge calls...")

    # Judge
    judge_semaphore = asyncio.Semaphore(config.get("judge_concurrency", 10))

    async def judge_record(record: dict) -> dict:
        async with judge_semaphore:
            bj, hj = await asyncio.gather(
                judge_turn(record["baseline_ai"], record["doctor_text"],
                           record["case_meta"], judge_client, config["judge_model"]),
                judge_turn(record["hac_ai"], record["doctor_text"],
                           record["case_meta"], judge_client, config["judge_model"]),
            )
            return {**record, "baseline_judge": bj, "hac_judge": hj}

    judged: list[dict] = await asyncio.gather(*[judge_record(r) for r in results])

    # Score
    scores = _compute_scores(judged)
    print_report(scores, judged)

    # Token summary
    print("  Token usage:")
    tracker.print_summary()
    print()

    # Save outputs
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"scores": scores, "results": judged}, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {results_path}")

    # Token ledger
    tracker.accumulate_to_ledger(
        _ROOT / "token_usage_ledger.json",
        run_meta={"script": "poc_medcobe_hac", "n_cases": len(cases), "model": config["openai_model"]},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
