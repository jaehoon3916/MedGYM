"""Per-persona re-estimation of (yield y, regression risk r, cost c) for the 3 control clusters,
to test the POC prediction: the accuracy ceiling p_bar = max_a y_a/(y_a+r_a) shifts by persona.

Method = analysis/action_space_effect.py, but (a) split by persona and (b) sourced from the GRPO
rollouts (grpo_phase1 + grpo_phase2), which are the only multi-persona rollouts we have with the
3-way control realization. Caveat: these come from the *evolving* ours_v2 policy mid-training, so
absolute yields differ from the deliberation_llm audit; what matters here is the RELATIVE shift
across personas, not the absolute calibration.
"""
import json, glob, collections, statistics

CLUSTER = {"INFORM": "PROBE", "PROPOSE": "PROBE",
           "CONSIDER": "CHALLENGE", "REVISE": "CHALLENGE",
           "RECOMMEND": "CONVERGE", "CONFIRM": "CONVERGE"}
RUNS = ["outputs/grpo_phase1/grpo/rollouts/*.jsonl",
        "outputs/grpo_phase2/grpo/rollouts/*.jsonl"]


def belief(turn):
    return str((turn.get("user_state") or {}).get("belief") or "").strip().upper()


def collect():
    # per[(persona, cluster)] = counters
    S = lambda: {"n": 0, "burden": [], "tc3": 0, "tc3_d": 0, "rg": 0, "rg_d": 0}
    per = collections.defaultdict(S)
    for pat in RUNS:
        for f in glob.glob(pat):
            last = None
            for line in open(f):
                last = json.loads(line)
            if not last:
                continue
            persona = (last.get("episode_config") or {}).get("persona") or "unknown"
            correct = str(last["case_info"].get("correct_option") or "").strip().upper()
            dh = last["dialogue_history"]
            for i, t in enumerate(dh):
                if t.get("speaker") != "medical" or not t.get("action"):
                    continue
                stage = t["action"].split(".")[0].upper()
                cl = CLUSTER.get(stage)
                if cl is None:
                    continue
                r = per[(persona, cl)]
                r["n"] += 1
                prev_u = next((dh[j] for j in range(i - 1, -1, -1) if dh[j]["speaker"] == "user"), None)
                nexts = [dh[j] for j in range(i + 1, len(dh)) if dh[j]["speaker"] == "user"]
                if nexts:
                    b = (nexts[0].get("user_state") or {}).get("cognitive_burden")
                    if isinstance(b, (int, float)):
                        r["burden"].append(b)
                if not correct or prev_u is None:
                    continue
                pb = belief(prev_u)
                if not pb:
                    continue
                if pb != correct and nexts:          # yield: wrong -> correct within 3
                    r["tc3_d"] += 1
                    if any(belief(u) == correct for u in nexts[:3]):
                        r["tc3"] += 1
                if pb == correct and nexts:          # risk: correct -> wrong next turn
                    r["rg_d"] += 1
                    nb = belief(nexts[0])
                    if nb and nb != correct:
                        r["rg"] += 1
    return per


def rate(a, b):
    return (a / b) if b else float("nan")


def main():
    per = collect()
    personas = sorted({p for (p, _c) in per})
    clusters = ["PROBE", "CHALLENGE", "CONVERGE"]
    print(f"{'persona':22} {'cluster':10} {'n':>5} {'c(burden)':>9} {'y(W3)':>7} {'r(reg)':>7} {'p*=y/(y+r)':>11}")
    print("-" * 80)
    ceilings = {}
    for persona in personas:
        pstars = []
        for cl in clusters:
            r = per.get((persona, cl))
            if not r or r["n"] == 0:
                continue
            c = statistics.mean(r["burden"]) if r["burden"] else float("nan")
            y = rate(r["tc3"], r["tc3_d"])
            rg = rate(r["rg"], r["rg_d"])
            pstar = y / (y + rg) if (y == y and rg == rg and (y + rg) > 0) else float("nan")
            if pstar == pstar:
                pstars.append(pstar)
            print(f"{persona:22} {cl:10} {r['n']:>5} {c:>9.2f} {y*100:>6.1f}% {rg*100:>6.1f}% {pstar:>11.3f}")
        if pstars:
            ceilings[persona] = max(pstars)
        print()
    print("=" * 80)
    print("POC prediction check — accuracy ceiling p_bar = max_a y_a/(y_a+r_a) per persona:")
    for persona in sorted(ceilings, key=lambda k: -ceilings[k]):
        print(f"  {persona:22} p_bar = {ceilings[persona]:.3f}")


if __name__ == "__main__":
    main()
