"""
Action-space empirical effect analysis.
=======================================
Question: are the 8-fold McBurney deliberation stages behaviorally distinct, and if the
rank collapses, to how many effective clusters -- and along what axes?

Method (all from EXISTING rollouts, ZERO new API calls):
  For every medical turn tagged with a deliberation action A = STAGE.locution, we treat that
  turn as the AI's *move* and read its downstream effect off the clinician (user) turns that
  follow it in the same episode's dialogue_history:
    - cost           : cognitive_burden (per-turn NASA-TLX overall, 1-5) on the NEXT user turn.
    - belief shift    : did the clinician's stated MCQ belief move TO the correct option?
                        measured in a window W (W=1 = next user turn only; W=3 = within next
                        three user turns) to separate immediate vs delayed effect.
    - regression risk : clinician was correct before the move, wrong after (W=1).
  Ground truth (correct_option) comes from case_info in the rollout itself.

Runs pooled (deepseek-v3.2, veteran_attending, n=100 each): plain-blind, estimate, meta_oracle.
Output: prints a table + writes a cost x yield scatter (one point per STAGE, size=usage) to
plot/result/action_space_cluster.png.
"""
import json, glob, collections, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/kjy/Jaehoon/medical_hac_policy"
RUNS = [
    "poc_0630_deliberation_llm",
    "poc_0630_deliberation_llm_estimate",
    "poc_0630_deliberation_llm_meta_oracle",
]
STAGES = ["INFORM", "PROPOSE", "CONSIDER", "REVISE", "RECOMMEND", "CONFIRM"]
CLUSTER = {
    "INFORM": "PROBE", "PROPOSE": "PROBE",
    "CONSIDER": "CHALLENGE", "REVISE": "CHALLENGE",
    "RECOMMEND": "CONVERGE", "CONFIRM": "CONVERGE",
}


def _belief(turn):
    us = turn.get("user_state") or {}
    return str(us.get("belief") or "").strip().upper()


def _last_line(f):
    last = None
    with open(f) as fh:
        for line in fh:
            last = json.loads(line)
    return last


def collect():
    S = lambda: {"n": 0, "burden": [], "tc1": 0, "tc1_d": 0, "tc3": 0, "tc3_d": 0, "rg": 0, "rg_d": 0}
    per = collections.defaultdict(S)
    seen = set()
    for run in RUNS:
        for f in sorted(glob.glob(f"{ROOT}/outputs/{run}/*/rollouts/*.jsonl")):
            rec = _last_line(f)
            if not rec:
                continue
            key = (run, rec["case_id"])
            if key in seen:
                continue
            seen.add(key)
            correct = str(rec["case_info"].get("correct_option") or "").strip().upper()
            dh = rec["dialogue_history"]
            for i, t in enumerate(dh):
                if t.get("speaker") != "medical" or not t.get("action"):
                    continue
                stage = t["action"].split(".")[0].upper()
                if stage not in STAGES:
                    continue
                r = per[stage]
                r["n"] += 1
                prev_u = next((dh[j] for j in range(i - 1, -1, -1) if dh[j]["speaker"] == "user"), None)
                nexts = [dh[j] for j in range(i + 1, len(dh)) if dh[j]["speaker"] == "user"]
                if nexts:
                    b = (nexts[0].get("user_state") or {}).get("cognitive_burden")
                    if isinstance(b, (int, float)):
                        r["burden"].append(b)
                if not correct or prev_u is None:
                    continue
                pb = _belief(prev_u)
                if not pb:
                    continue
                if pb != correct and nexts:
                    r["tc1_d"] += 1
                    if _belief(nexts[0]) == correct:
                        r["tc1"] += 1
                    r["tc3_d"] += 1
                    if any(_belief(u) == correct for u in nexts[:3]):
                        r["tc3"] += 1
                if pb == correct and nexts:
                    r["rg_d"] += 1
                    nb = _belief(nexts[0])
                    if nb and nb != correct:
                        r["rg"] += 1
    return per


def rate(a, b):
    return (a / b) if b else float("nan")


def main():
    per = collect()
    rows = []
    lines = []
    hdr = f"{'STAGE':11s} {'n':>5} {'burden':>7} {'->corr W1':>10} {'->corr W3':>10} {'delay':>7} {'regress':>8}  cluster"
    print("\n" + hdr); print("-" * 100)
    lines += [hdr, "-" * 100]
    for st in STAGES:
        r = per.get(st)
        if not r or r["n"] == 0:
            continue
        burden = statistics.mean(r["burden"]) if r["burden"] else float("nan")
        w1 = rate(r["tc1"], r["tc1_d"]); w3 = rate(r["tc3"], r["tc3_d"]); rg = rate(r["rg"], r["rg_d"])
        rows.append((st, r["n"], burden, w1, w3, rg))
        line = f"{st:11s} {r['n']:>5} {burden:>7.2f} {w1*100:>9.1f}% {w3*100:>9.1f}% {(w3-w1)*100:>+6.1f}% {rg*100:>7.1f}%  {CLUSTER[st]}"
        print(line); lines.append(line)

    print("\n" + "=" * 100)
    lines += ["", "=" * 100, "COLLAPSED to 3 clusters (W3 yield):"]
    print("COLLAPSED to 3 clusters (W3 yield):")
    cl = collections.defaultdict(lambda: {"burden": [], "tc3": 0, "tc3_d": 0, "rg": 0, "rg_d": 0, "n": 0})
    for st in STAGES:
        r = per.get(st)
        if not r:
            continue
        c = CLUSTER[st]
        cl[c]["burden"] += r["burden"]; cl[c]["tc3"] += r["tc3"]; cl[c]["tc3_d"] += r["tc3_d"]
        cl[c]["rg"] += r["rg"]; cl[c]["rg_d"] += r["rg_d"]; cl[c]["n"] += r["n"]
    ch = f"{'cluster':12s} {'n':>6} {'burden':>7} {'->corr W3':>10} {'regress':>8}"
    print(ch); lines.append(ch)
    for c in ["PROBE", "CHALLENGE", "CONVERGE"]:
        d = cl[c]
        line = f"{c:12s} {d['n']:>6} {statistics.mean(d['burden']):>7.2f} {rate(d['tc3'],d['tc3_d'])*100:>9.1f}% {rate(d['rg'],d['rg_d'])*100:>7.1f}%"
        print(line); lines.append(line)

    (ROOT_OUT := f"{ROOT}/analysis/action_space_stats.txt")
    with open(ROOT_OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # scatter
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.axvspan(0.9, 2.05, alpha=0.05, color="green")
    ax.axvspan(2.05, 3.3, alpha=0.06, color="orange")
    for st, n, burden, w1, w3, rg in rows:
        sc = ax.scatter(burden, w3 * 100, s=90 + n / 4.0, c=[rg * 100], cmap=plt.cm.Reds,
                        vmin=0, vmax=12, edgecolors="black", linewidths=1.2, zorder=3)
        ax.annotate(f"{st}\n(n={n})", (burden, w3 * 100), textcoords="offset points",
                    xytext=(9, 5), fontsize=9, fontweight="bold")
    ax.text(1.35, 23, "cheap zone\n(PROBE / CONVERGE)", fontsize=9, color="green", alpha=0.75, ha="center")
    ax.text(2.75, 23, "CHALLENGE\ncostly · high-yield · high-risk", fontsize=9, color="darkorange", ha="center")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("regression risk % (correct→wrong, next turn)")
    ax.set_xlabel("COST — avg clinician cognitive burden on next turn (NASA-TLX 1-5)")
    ax.set_ylabel("YIELD — belief→correct within 3 turns (%)")
    ax.set_title("Deliberation action effects: cost × yield (point size = usage)\n"
                 "pooled deepseek-v3.2 / veteran_attending / 300 episodes")
    ax.grid(alpha=0.3, zorder=0)
    out = f"{ROOT}/plot/result/action_space_cluster.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nstats -> {ROOT_OUT}\nscatter -> {out}")


if __name__ == "__main__":
    main()
