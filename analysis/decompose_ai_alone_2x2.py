"""Decompose PoC run accuracy by the (doctor-alone x AI-alone) correctness 2x2.

Motivation (2026-07-05 thread): naive / user_state_oracle / react all sit ~0.60 while the
full oracle hits ~0.92. This script shows WHERE the gap lives: the oracle's edge splits into
(a) the recoverable pool D-wrong/AI-right, where the knowledge already exists in the system
and non-oracle policies fail to deploy it, and (b) the hopeless pool D-wrong/AI-wrong, where
the oracle wins purely by injecting ground truth no deployable policy can have. (b) must be
excluded when quoting a reachable ceiling.

Also counts, for the recoverable pool's failures, whether the user simulator's evidence tags
say the AI's own utterances mostly SUPPORTED the doctor's wrong anchor (renderer sycophancy)
vs pushed elsewhere (pushed-but-failed).

Usage:
  python analysis/decompose_ai_alone_2x2.py [run ...]
Writes plot/result/plot_scaling_poc/ai_alone_2x2_decomposition.csv (+ sycophancy CSV).
"""

import collections
import csv
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plot", "result", "plot_scaling_poc")

DEFAULT_RUNS = [
    "poc_0704_naive_a3",
    "poc_0704_oracle_a3",
    "poc_0704_user_state_oracle_a3",
    "poc_0705_react_control_v3",
]


def load_records(run: str):
    for rp in sorted(glob.glob(os.path.join(ROOT, run, "*/results.json"))):
        persona = os.path.basename(os.path.dirname(rp)).replace("_full", "")
        with open(rp) as f:
            for r in json.load(f)["records"]:
                yield persona, r


def natural_correct(r: dict) -> bool:
    # force_full_turns runs never set natural_end_correct when no close fired; the episode's
    # deployed outcome is then the turn-8 state.
    return r["natural_end_correct"] if r["natural_end_correct"] is not None else r["is_correct"]


def decompose(runs: list[str]) -> list[dict]:
    rows = []
    for run in runs:
        cells = collections.defaultdict(lambda: {"n": 0, "nat_ok": 0, "t8_ok": 0, "end_turns": []})
        for _persona, r in load_records(run):
            c = cells[(r["alone_correct"], r["ai_alone_correct"])]
            c["n"] += 1
            c["nat_ok"] += 1 if natural_correct(r) else 0
            c["t8_ok"] += 1 if r["is_correct"] else 0
            c["end_turns"].append(r["natural_end_turn"] if r["natural_end_turn"] is not None else 8)
        for (d, a), c in sorted(cells.items()):
            rows.append({
                "run": run,
                "doctor_alone_correct": d,
                "ai_alone_correct": a,
                "n": c["n"],
                "natural_end_acc": round(c["nat_ok"] / c["n"], 4),
                "turn8_acc": round(c["t8_ok"] / c["n"], 4),
                "mean_natural_end_turn": round(sum(c["end_turns"]) / c["n"], 2),
            })
    return rows


def sycophancy_in_recoverable_failures(run: str) -> dict:
    """In D-wrong/AI-right episodes that ended wrong: per-turn evidence tags on the AI's
    utterances -- did they net-support the doctor's initial (wrong) belief?"""
    res = {(p, r["case_id"]): r for p, r in load_records(run)}
    counts = {"supported_doctor_anchor": 0, "supported_other_option": 0, "no_positive_evidence": 0}
    for f in sorted(glob.glob(os.path.join(ROOT, run, "*/rollouts/*.jsonl"))):
        persona = os.path.basename(os.path.dirname(os.path.dirname(f))).replace("_full", "")
        with open(f) as fh:
            recs = [json.loads(l) for l in fh if l.strip()]
        if not recs:
            continue
        r = res.get((persona, recs[-1]["case_id"]))
        if r is None or r["alone_correct"] or not r["ai_alone_correct"] or natural_correct(r):
            continue
        states = [t["user_state"] for t in recs[-1]["dialogue_history"]
                  if t.get("user_state") and "evidence_tags" in t["user_state"]]
        if not states:
            continue
        anchor = states[0].get("belief")
        net = collections.Counter()
        for us in states:
            ev = us["evidence_tags"]
            if isinstance(ev, dict) and ev:
                best = max(ev, key=lambda k: ev[k])
                if ev[best] > 0:
                    net[best] += 1
        if not net:
            counts["no_positive_evidence"] += 1
        elif net.most_common(1)[0][0] == anchor:
            counts["supported_doctor_anchor"] += 1
        else:
            counts["supported_other_option"] += 1
    return counts


def main() -> None:
    runs = sys.argv[1:] or DEFAULT_RUNS
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = decompose(runs)
    p1 = os.path.join(OUT_DIR, "ai_alone_2x2_decomposition.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {p1} ({len(rows)} rows)")

    p2 = os.path.join(OUT_DIR, "recoverable_failure_sycophancy.csv")
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "supported_doctor_anchor", "supported_other_option", "no_positive_evidence"])
        for run in runs:
            c = sycophancy_in_recoverable_failures(run)
            w.writerow([run, c["supported_doctor_anchor"], c["supported_other_option"],
                        c["no_positive_evidence"]])
            print(run, c)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
