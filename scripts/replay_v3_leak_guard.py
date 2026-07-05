"""Replay the v3-aware leak guard over saved rollout transcripts.

No API calls. This audits whether answer-naming guidance is flagged only when it cannot be
attributed to the medical LLM's own belief, which is the pre-flight check for belief-shaped GRPO.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.reward import _map_belief_to_option, leak_score, leak_score_v3


def _iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _last_medical_belief(rec: dict) -> str | None:
    if rec.get("medical_belief"):
        return rec.get("medical_belief")
    for turn in reversed(rec.get("dialogue_history") or []):
        if turn.get("speaker") == "medical":
            meta = turn.get("metadata") or {}
            return meta.get("belief") or turn.get("text")
    return rec.get("medical_response")


def _control(rec: dict) -> str:
    action = str(rec.get("selected_action") or "").upper()
    return action.split(".")[0]


def collect(paths: list[Path]) -> list[dict]:
    rows = []
    for root in paths:
        files = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]
        for fp in files:
            if fp.stem.endswith("_calls"):
                continue
            for rec in _iter_jsonl(fp):
                case = rec.get("case_info") or {}
                correct = str(case.get("correct_option") or "").strip().upper()
                answer = case.get("answer") or ""
                options = case.get("options") or {}
                guidance = rec.get("action_prompt") or ""
                med_belief = _last_medical_belief(rec)
                med_option = _map_belief_to_option(med_belief, options)
                raw = leak_score(guidance, correct, answer)
                v3 = leak_score_v3(guidance, correct, answer, med_belief, options)
                rows.append({
                    "path": str(fp),
                    "case_id": rec.get("case_id"),
                    "turn_id": rec.get("turn_id"),
                    "control": _control(rec),
                    "correct_option": correct,
                    "medical_belief_mapped": med_option,
                    "medical_belief_aligned": int(med_option == correct) if med_option else None,
                    "raw_leak": raw,
                    "v3_leak": v3,
                })
    return rows


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        aligned = (
            "aligned" if r["medical_belief_aligned"] == 1
            else "misaligned_or_unmapped"
        )
        groups.setdefault((r["control"], aligned), []).append(r)
    out = []
    for (control, aligned), rs in sorted(groups.items()):
        out.append({
            "control": control,
            "medical_belief": aligned,
            "n": len(rs),
            "raw_leak_rate": _rate(sum(int(r["raw_leak"]) for r in rs), len(rs)),
            "v3_leak_rate": _rate(sum(int(r["v3_leak"]) for r in rs), len(rs)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("outputs/grpo_v3/grpo/rollouts")],
        help="Rollout jsonl file or directory. Defaults to outputs/grpo_v3/grpo/rollouts.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/leak_guard_v3_preflight"))
    args = ap.parse_args()

    rows = collect(args.paths)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "detail.csv"
    summary_path = args.out_dir / "summary.csv"
    if rows:
        with detail_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    summary = summarize(rows)
    if summary:
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)

    counts = Counter((r["control"], r["medical_belief_aligned"], r["raw_leak"], r["v3_leak"]) for r in rows)
    print(f"rows={len(rows)} detail={detail_path} summary={summary_path}")
    for row in summary:
        print(row)
    if not rows:
        print("No rollout rows found.")


if __name__ == "__main__":
    main()
