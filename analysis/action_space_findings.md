# Action-space empirical audit — is the 8-fold McBurney space the right one?

**Date:** 2026-07-01  ·  **Data:** existing rollouts only (zero new API calls)
**Reproduce:** `python3 analysis/action_space_effect.py` → `analysis/action_space_stats.txt` + `plot/result/action_space_cluster.png`

---

## 0. Why we asked

The policy's action space is `configs/action_space.yaml`: the 8-fold McBurney–Hitchcock–Parsons
deliberation stages (INFORM / PROPOSE / CONSIDER / REVISE / RECOMMEND / CONFIRM, after CLOSE was
removed on 0701) × locutions. It is theoretically well-grounded (guideline.txt) but two doubts:

1. **Realization doubt.** For a *trained discrete* policy, the chosen `STAGE.locution` is turned
   into behavior only through the short `description` string in `action_space.yaml`
   (`plugins/policy/qwen_policy.py::_build_policy_output`). So the PoC numbers (0.63–0.68) were
   produced by `deliberation_llm`'s *tailored free-text* guidance, NOT by discrete labels — the
   discrete RL policy may not reproduce them.
2. **Axis doubt.** The thesis objective is "belief movement per unit of clinician cognitive
   burden," but the McBurney axis only classifies the *epistemic kind* of an utterance. Neither
   the **pressure/direction** of persuasion nor the **cost** of a move is in the label; both live
   in the free text. Also, removing CLOSE left the policy with **no stop/pacing lever**.

## 1. How we measured (method)

For every medical turn tagged with action `A = STAGE.locution` across 3 pooled runs
(`deliberation_llm`, `_estimate`, `_meta_oracle`; deepseek-v3.2, veteran_attending, 100 cases
each = 300 episodes), we read the move's downstream effect off the clinician turns that follow it
in the same `dialogue_history`:

- **cost** = `cognitive_burden` (per-turn NASA-TLX overall, 1–5) on the *next* clinician turn.
- **yield** = did the clinician's stated MCQ `belief` move **to the correct option**, measured in
  a window W: **W1** = next clinician turn only; **W3** = within the next three clinician turns.
  (The W1↔W3 gap isolates *immediate* vs *delayed* effect — the key to the INFORM question.)
- **regression risk** = clinician was correct before the move, wrong on the next turn (W1).
- ground truth = `case_info.correct_option` from the rollout itself.

## 2. Results

```
STAGE           n  burden  ->corr W1  ->corr W3   delay  regress  cluster
INFORM       1470    1.58       2.2%      15.7%  +13.4%     1.0%  PROBE
PROPOSE       116    1.75      12.5%      16.7%   +4.2%     4.9%  PROBE
CONSIDER      415    2.80      17.8%      28.4%  +10.6%     9.5%  CHALLENGE
REVISE         67    2.66      13.6%      25.0%  +11.4%    21.1%  CHALLENGE
RECOMMEND     392    1.75       6.0%      10.2%   +4.2%     5.1%  CONVERGE
CONFIRM      1140    1.04       2.9%       4.0%   +1.1%     1.5%  CONVERGE

COLLAPSED to 3 clusters (W3 yield):
cluster           n  burden  ->corr W3  regress
PROBE          1586    1.60      15.7%     1.3%
CHALLENGE       482    2.78      27.9%    10.9%
CONVERGE       1532    1.25       6.4%     2.4%
```

## 3. Findings

**F1 — The 8-fold labels are NOT all behaviorally distinct; they collapse to 3 clusters on a
cost × yield × risk manifold.** The scatter (`action_space_cluster.png`) shows three separated
regions, not eight points and not two.

**F2 — INFORM was under-valued by next-turn attribution; it is a cheap, *delayed*-yield probe.**
Its yield jumps 2.2% → 15.7% from W1 to W3 (+13.4pp, the largest delay of any move). INFORM is
not low-yield; it is low-*cost*, effect-shows-up-downstream. (Resolves the caveat flagged earlier
— and reverses it.)

**F3 — The thesis "tension" is now a single measured fact: CHALLENGE is the only high-yield move,
and it is simultaneously the most expensive (burden 2.78 vs ~1.3) AND the most dangerous
(regression 10.9% vs ~2%).** Corrective power, cost, and regression risk are *coupled inside the
same act*. This is exactly the accuracy–burden trade-off, localized to one action cluster — and
it is what a learned policy must arbitrate: when is CHALLENGE worth its cost and risk vs a cheaper
PROBE vs CONVERGE/stop.

**F4 — CONVERGE is the low-cost terminal lever** (burden 1.04–1.75, near-zero yield: it closes,
it doesn't correct). Its existence is what lets a policy *stop paying burden* once belief has
moved — the pacing/stop capability the thesis needs.

## 3b. Burden is driven by *demanded re-reasoning*, not conflict — and reconciles the CHI axis

A follow-up broke the next-turn burden into its NASA-TLX dimensions per cluster:

```
cluster     overall   mental   effort   frustr     perf
EXPAND         1.60     1.93     1.71     1.38     1.39
CHALLENGE      2.78     3.28     3.03     2.43     2.40
CONVERGE       1.25     1.43     1.24     1.20     1.17
Δ(CHALLENGE − EXPAND): mental +1.35, effort +1.32, frustration +1.05, performance +1.01
```

CHALLENGE's excess burden is led by **mental_demand and effort**, not frustration → it is genuine
cognitive load (the clinician is made to re-reason), not conflict-annoyance, and not a per-turn
attribution artifact. This reconciles the apparent contradiction with the CHI paper
("I, Help Me Think"): that paper predicts *think-support* costs more than *recommendation-support*.
Our data agree — but reveal that **think-support itself splits into a cheap and an expensive mode**:

- support-type axis (CHI, citable): recommendation (**CONVERGE**, cheapest) vs cognitive support.
- within cognitive support, a *demanded-re-reasoning* gradient: light ELICIT (**EXPAND**) → heavy
  CONFRONT (**CHALLENGE**). Burden tracks this second axis.

So EXPAND and CHALLENGE are the light/heavy ends of think-support, not "expand vs recommend."
The naming in this file uses the empirical (cost-cluster) labels; the CHI mapping is the theory
layer. Final action space keeps the 3-way (option 가); see `docs/problem_formulation.md §3`.

## 4. Verdict

The action-space doubt was justified, but the fix is **not** "switch to recommend-vs-expand" (that
over-collapses to 2 and drops the stop lever). The data support **rank ≈ 3** along
(cost, yield, risk): **PROBE / CHALLENGE / CONVERGE** — which coincides exactly with the three
levers the thesis requires (cheap exploration / costly correction / cheap closure). Design
decision: keep McBurney as the *realization* layer, expose these 3 as the *control* layer.
See `docs/problem_formulation.md`.
