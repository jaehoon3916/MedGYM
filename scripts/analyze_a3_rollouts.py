#!/usr/bin/env python3
"""Turn-level analysis for action_space_v3 / A3 rollouts.

This script is meant to answer the reward-design question:

  "When the policy picked EXTEND or RECOMMEND, what actually happened to the doctor's
   belief state and cognitive burden?"

It does not call any LLMs. It reconstructs before/after user states from saved rollout JSONL
files and writes:

  - turn_rows.csv: one row per AI turn with before/after belief, burden, AI belief mapping.
  - action_summary.csv: EXTEND vs RECOMMEND aggregate effect.
  - persona_action_summary.csv: the same split by persona/condition.
  - correctness_action_summary.csv: action effect split by whether the doctor was correct before
    the AI turn.
  - persona_correctness_action_summary.csv: the same split by condition/persona.
  - run_summary.csv: one row per condition from results.json, when available.

Usage:
  # point at any poc_run.py output dir (naive_a3, oracle_a3, meta_oracle_a3, ...) --
  # glob/out-dir are derived from it automatically:
  python scripts/analyze_a3_rollouts.py --run-dir outputs/poc_0704_oracle_a3

  # or override glob/out-dir explicitly (still supported, takes precedence over --run-dir):
  python scripts/analyze_a3_rollouts.py \
    --glob "outputs/poc_0704_naive_a3/*_full/rollouts/*.jsonl" \
    --out-dir outputs/poc_0704_naive_a3/a3_analysis
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "with", "in", "to", "is", "are", "for", "on", "at",
    "by", "as", "from", "this", "that", "likely", "most", "consistent", "given", "due",
    "diagnosis", "condition", "option", "choice", "answer", "patient", "clinical",
}

_TURN_FIELDS = [
    "condition", "case_id", "turn_id", "action", "policy_control",
    "doctor_before", "doctor_after", "correct_option",
    "before_correct", "after_correct", "delta_correct", "recovered", "regressed",
    "confidence_before", "confidence_after", "delta_confidence",
    "burden", "burden_cumulative_after", "decision_after", "termination_reason_after",
    "mental_demand", "performance", "effort", "frustration", "burden_n_ok",
    "medical_belief_text", "medical_belief_mapped", "medical_belief_correct",
    "medical_confidence", "response_option_mapped", "response_mentions_option",
    "doctor_moved_to_medical_belief",
]

_CORRECTNESS_FIELDS = [
    "doctor_before_status", "action", "condition",
    "n_turns", "after_correct_rate", "mean_delta_correct", "mean_delta_confidence",
    "mean_burden", "mean_mental_demand", "mean_performance", "mean_effort", "mean_frustration",
    "medical_belief_mapped_rate", "medical_belief_correct_rate_when_mapped",
    "response_mentions_option_rate", "doctor_moved_to_medical_belief_rate", "end_decision_rate",
    "stayed_correct_rate", "regressed_rate", "recovered_rate", "stayed_wrong_rate",
]


def _condition_from_path(path: Path) -> str:
    # outputs/<run>/<condition>/rollouts/<case>.jsonl
    return path.parent.parent.name if path.parent.name == "rollouts" else path.parent.name


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _words(text: Any) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _map_text_to_option(text: Any, options: dict[str, str]) -> str | None:
    """No-LLM heuristic for mapping free text to a lettered option.

    First accept explicit option text substrings, then fall back to content-word overlap.
    Returns None on ties/no evidence.
    """
    if not text:
        return None
    text_norm = _norm(text)
    exact = [
        letter for letter, option_text in options.items()
        if _norm(option_text) and _norm(option_text) in text_norm
    ]
    if len(exact) == 1:
        return exact[0]

    text_words = _words(text)
    scored: list[tuple[int, str]] = []
    for letter, option_text in options.items():
        score = len(text_words & _words(option_text))
        if score > 0:
            scored.append((score, letter))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _response_mentions_option(response: str, letter: str | None, options: dict[str, str]) -> bool:
    if not response or not letter:
        return False
    response_norm = _norm(response)
    option_norm = _norm(options.get(letter, ""))
    if option_norm and option_norm in response_norm:
        return True
    return bool(re.search(rf"\b(?:option|choice|answer)\s*(?:is\s*)?{re.escape(letter)}\b", response, re.I))


def _latest_medical_index(turns: list[dict]) -> int | None:
    for idx in range(len(turns) - 1, -1, -1):
        if turns[idx].get("speaker") == "medical":
            return idx
    return None


def _nearest_user_state(turns: list[dict], start: int, direction: int) -> dict | None:
    idx = start
    while 0 <= idx < len(turns):
        turn = turns[idx]
        if turn.get("speaker") == "user" and turn.get("user_state"):
            return turn["user_state"]
        idx += direction
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int_bool(value: bool | None) -> int | None:
    return None if value is None else int(bool(value))


def extract_turn_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        condition = _condition_from_path(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            turns = rec.get("dialogue_history") or []
            med_idx = _latest_medical_index(turns)
            if med_idx is None:
                continue
            medical_turn = turns[med_idx]
            before = _nearest_user_state(turns, med_idx - 1, -1)
            after = _nearest_user_state(turns, med_idx + 1, 1)
            if not before or not after:
                # The cap-hitting final medical turn has no following doctor reaction, so it
                # cannot measure action effect.
                continue

            case = rec.get("case_info") or {}
            options = case.get("options") or {}
            correct = str(case.get("correct_option") or "").strip().upper()
            doctor_before = str(before.get("belief") or "").strip().upper() or None
            doctor_after = str(after.get("belief") or "").strip().upper() or None
            before_correct = doctor_before == correct if doctor_before and correct else None
            after_correct = doctor_after == correct if doctor_after and correct else None

            metadata = medical_turn.get("metadata") or {}
            response = medical_turn.get("text") or rec.get("medical_response") or ""
            medical_belief_text = metadata.get("belief")
            medical_belief_mapped = _map_text_to_option(medical_belief_text, options)
            response_option_mapped = _map_text_to_option(response, options)
            mapped_for_mention = response_option_mapped or medical_belief_mapped

            dims = after.get("cognitive_burden_dims") or {}
            action = str(rec.get("selected_action") or medical_turn.get("action") or "").split(".")[0].upper()
            policy_control = str((rec.get("policy") or {}).get("control") or action).upper()
            delta_correct = (
                int(after_correct) - int(before_correct)
                if before_correct is not None and after_correct is not None else None
            )
            row = {
                "condition": condition,
                "case_id": rec.get("case_id") or case.get("case_id"),
                "turn_id": rec.get("turn_id"),
                "action": action,
                "policy_control": policy_control,
                "doctor_before": doctor_before,
                "doctor_after": doctor_after,
                "correct_option": correct,
                "before_correct": _as_int_bool(before_correct),
                "after_correct": _as_int_bool(after_correct),
                "doctor_before_status": (
                    "correct" if before_correct is True else
                    "wrong" if before_correct is False else
                    "unknown"
                ),
                "delta_correct": delta_correct,
                "recovered": int(before_correct is False and after_correct is True),
                "regressed": int(before_correct is True and after_correct is False),
                "confidence_before": _as_float(before.get("confidence")),
                "confidence_after": _as_float(after.get("confidence")),
                "delta_confidence": (
                    _as_float(after.get("confidence")) - _as_float(before.get("confidence"))
                    if _as_float(after.get("confidence")) is not None
                    and _as_float(before.get("confidence")) is not None else None
                ),
                "burden": _as_float(after.get("cognitive_burden")),
                "burden_cumulative_after": _as_float(after.get("cognitive_burden_cumulative")),
                "decision_after": after.get("decision"),
                "termination_reason_after": after.get("termination_reason"),
                "mental_demand": _as_float(dims.get("mental_demand")),
                "performance": _as_float(dims.get("performance")),
                "effort": _as_float(dims.get("effort")),
                "frustration": _as_float(dims.get("frustration")),
                "burden_n_ok": after.get("cognitive_burden_n_ok"),
                "medical_belief_text": medical_belief_text,
                "medical_belief_mapped": medical_belief_mapped,
                "medical_belief_correct": int(medical_belief_mapped == correct) if medical_belief_mapped else None,
                "medical_confidence": _as_float(metadata.get("confidence")),
                "response_option_mapped": response_option_mapped,
                "response_mentions_option": int(_response_mentions_option(response, mapped_for_mention, options)),
                "doctor_moved_to_medical_belief": int(
                    bool(medical_belief_mapped)
                    and doctor_before != medical_belief_mapped
                    and doctor_after == medical_belief_mapped
                ),
            }
            rows.append(row)
    return rows


def _mean(values: list[Any]) -> float | None:
    xs = [_as_float(v) for v in values]
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 4) if xs else None


def _rate(rows: list[dict], field: str) -> float | None:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return round(sum(float(v) for v in vals) / len(vals), 4) if vals else None


def summarize(rows: list[dict], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(f) for f in group_fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda kv: tuple("" if v is None else str(v) for v in kv[0])):
        wrong_before = [r for r in group if r.get("before_correct") == 0]
        right_before = [r for r in group if r.get("before_correct") == 1]
        mapped = [r for r in group if r.get("medical_belief_mapped")]
        item = {field: value for field, value in zip(group_fields, key)}
        item.update({
            "n_turns": len(group),
            "n_wrong_before": len(wrong_before),
            "n_right_before": len(right_before),
            "mean_delta_correct": _mean([r.get("delta_correct") for r in group]),
            "recover_if_wrong": _rate(wrong_before, "recovered"),
            "regress_if_right": _rate(right_before, "regressed"),
            "mean_delta_confidence": _mean([r.get("delta_confidence") for r in group]),
            "mean_burden": _mean([r.get("burden") for r in group]),
            "mean_mental_demand": _mean([r.get("mental_demand") for r in group]),
            "mean_performance": _mean([r.get("performance") for r in group]),
            "mean_effort": _mean([r.get("effort") for r in group]),
            "mean_frustration": _mean([r.get("frustration") for r in group]),
            "medical_belief_mapped_rate": round(len(mapped) / len(group), 4) if group else None,
            "medical_belief_correct_rate_when_mapped": _rate(mapped, "medical_belief_correct"),
            "response_mentions_option_rate": _rate(group, "response_mentions_option"),
            "doctor_moved_to_medical_belief_rate": _rate(group, "doctor_moved_to_medical_belief"),
            "end_decision_rate": _rate(
                [{**r, "_end": int(r.get("decision_after") == "END")} for r in group], "_end"
            ),
        })
        out.append(item)
    return out


def summarize_by_before_correctness(rows: list[dict], include_condition: bool = False) -> list[dict[str, Any]]:
    group_fields = ["doctor_before_status", "action"]
    if include_condition:
        group_fields = ["condition", *group_fields]

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("doctor_before_status") == "unknown":
            continue
        grouped[tuple(row.get(f) for f in group_fields)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        item = {field: value for field, value in zip(group_fields, key)}
        before_status = item["doctor_before_status"]
        item.update({
            "n_turns": len(group),
            "after_correct_rate": _rate(group, "after_correct"),
            "mean_delta_correct": _mean([r.get("delta_correct") for r in group]),
            "mean_delta_confidence": _mean([r.get("delta_confidence") for r in group]),
            "mean_burden": _mean([r.get("burden") for r in group]),
            "mean_mental_demand": _mean([r.get("mental_demand") for r in group]),
            "mean_performance": _mean([r.get("performance") for r in group]),
            "mean_effort": _mean([r.get("effort") for r in group]),
            "mean_frustration": _mean([r.get("frustration") for r in group]),
            "medical_belief_mapped_rate": round(
                sum(1 for r in group if r.get("medical_belief_mapped")) / len(group), 4
            ) if group else None,
            "medical_belief_correct_rate_when_mapped": _rate(
                [r for r in group if r.get("medical_belief_mapped")], "medical_belief_correct"
            ),
            "response_mentions_option_rate": _rate(group, "response_mentions_option"),
            "doctor_moved_to_medical_belief_rate": _rate(group, "doctor_moved_to_medical_belief"),
            "end_decision_rate": _rate(
                [{**r, "_end": int(r.get("decision_after") == "END")} for r in group], "_end"
            ),
            "stayed_correct_rate": _rate(
                [{**r, "_v": int(r.get("after_correct") == 1)} for r in group], "_v"
            ) if before_status == "correct" else None,
            "regressed_rate": _rate(group, "regressed") if before_status == "correct" else None,
            "recovered_rate": _rate(group, "recovered") if before_status == "wrong" else None,
            "stayed_wrong_rate": _rate(
                [{**r, "_v": int(r.get("after_correct") == 0)} for r in group], "_v"
            ) if before_status == "wrong" else None,
        })
        out.append(item)
    return out


def load_run_summary(paths: list[Path]) -> list[dict[str, Any]]:
    result_paths = sorted({p.parent.parent / "results.json" for p in paths if (p.parent.parent / "results.json").exists()})
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        data = json.loads(path.read_text())
        scores = data.get("scores") or {}
        records = data.get("records") or []
        nat = [r for r in records if r.get("natural_end_turn") is not None]
        rows.append({
            "condition": path.parent.name,
            "n_cases": len(records),
            "doctor_alone_accuracy": scores.get("doctor_alone_accuracy"),
            "ai_alone_accuracy": scores.get("ai_alone_accuracy"),
            "terminal_accuracy": _rate(
                [{**r, "_correct": int(bool(r.get("is_correct")))} for r in records], "_correct"
            ),
            "natural_end_n": len(nat),
            "natural_end_rate": round(len(nat) / len(records), 4) if records else None,
            "mean_natural_end_turn": _mean([r.get("natural_end_turn") for r in nat]),
            "natural_end_accuracy": _rate(
                [{**r, "_correct": int(bool(r.get("natural_end_correct")))} for r in nat], "_correct"
            ),
            "mean_burden_to_close": _mean([r.get("burden_to_close") for r in records]),
            "mean_final_cumulative_burden": _mean([r.get("final_cumulative_burden") for r in records]),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir", default="outputs/poc_0704_naive_a3",
        help="Root of one poc_run.py output (contains <persona>_full/rollouts/*.jsonl). "
             "Derives --glob/--out-dir when those are not given explicitly, so pointing this "
             "at a different run (e.g. outputs/poc_0704_oracle_a3) is enough on its own.",
    )
    ap.add_argument("--glob", default=None, help="Overrides the glob derived from --run-dir.")
    ap.add_argument("--out-dir", default=None, help="Overrides the out-dir derived from --run-dir.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    glob_pattern = args.glob or str(run_dir / "*_full" / "rollouts" / "*.jsonl")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "a3_analysis"

    paths = [Path(p) for p in sorted(glob.glob(glob_pattern))]
    if not paths:
        raise SystemExit(f"No rollout files matched: {glob_pattern}")
    rows = extract_turn_rows(paths)
    action_summary = summarize(rows, ["action"])
    persona_action_summary = summarize(rows, ["condition", "action"])
    correctness_action_summary = summarize_by_before_correctness(rows)
    persona_correctness_action_summary = summarize_by_before_correctness(rows, include_condition=True)
    run_summary = load_run_summary(paths)

    write_csv(out_dir / "turn_rows.csv", rows, _TURN_FIELDS)
    write_csv(out_dir / "action_summary.csv", action_summary)
    write_csv(out_dir / "persona_action_summary.csv", persona_action_summary)
    write_csv(out_dir / "correctness_action_summary.csv", correctness_action_summary, _CORRECTNESS_FIELDS)
    write_csv(
        out_dir / "persona_correctness_action_summary.csv",
        persona_correctness_action_summary,
        _CORRECTNESS_FIELDS,
    )
    write_csv(out_dir / "run_summary.csv", run_summary)

    print(f"Analyzed {len(rows)} turn(s) from {len(paths)} rollout file(s).")
    print(f"Wrote {out_dir / 'turn_rows.csv'}")
    print()
    print("Action summary:")
    for row in action_summary:
        print(
            f"  {row['action']:<10} n={row['n_turns']:<4} "
            f"recover={row['recover_if_wrong']} regress={row['regress_if_right']} "
            f"burden={row['mean_burden']} med_correct={row['medical_belief_correct_rate_when_mapped']} "
            f"moved_to_med={row['doctor_moved_to_medical_belief_rate']}"
        )


if __name__ == "__main__":
    main()
