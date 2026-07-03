# Problem Formulation — Interactive Test-Time Scaling under a Cognitive Budget

**Date:** 2026-07-01 (final consolidation)  ·  Companion: `analysis/action_space_findings.md`.
Anchors the RL PoC. This is the single source of truth for state / action / environment / objective.

---

## 1. One-paragraph framing

We cast collaborative clinical decision-making dialogue as **test-time interaction scaling under a
human cognitive budget.** Solo test-time scaling buys accuracy with the model's own tokens (cheap).
Interactive scaling instead spends a **scarce, non-renewable, dropout-inducing resource — the
clinician's cognitive burden.** The honest scaling axis is not turn count but **cumulative burden**
(turns are not equally expensive: `analysis/action_space_findings.md` shows a correcting CHALLENGE
turn costs ~2.8 vs ~1.2 for a closing CONVERGE turn). The goal is a policy π that **dominates the
accuracy–cost Pareto frontier**: for any budget B, higher joint decision accuracy. Equivalently
`max E[acc] − λ·E[cost]`, the λ-sweep tracing the frontier. "Doing well" is a *single* quantity
(frontier position), not two competing goals.

The empirical audit localizes the tension to one measured fact: **CHALLENGE is the only high-yield
corrective move, and it is simultaneously the most expensive and the most regression-prone.** The
policy's whole job is to arbitrate **when CHALLENGE is worth its cost/risk vs a cheaper EXPAND vs
CONVERGE/stop**, given an estimate of the epistemic state.

## 2. Environment  (already implemented — `core/environment.py`)

Episode = one MCQ case × one clinician persona. Each step:
policy picks action → medical LLM answers under that action's guidance → clinician (user sim v3)
responds, self-reporting `belief`, `confidence`, per-turn `cognitive_burden` (NASA-TLX overall 1–5,
with `cognitive_burden_dims`) → fact-validator scores the clinician's latest claim.
Termination: `agreement | burden_dropout | max_turns`. `burden_dropout` (v3) fires when cumulative
burden crosses the persona threshold — the *hazard* that makes cost a real constraint.

**RL requirement:** run with **`force_full_turns: false`** so the policy controls episode length by
driving agreement. With it *true*, every episode pads to `max_turns` → turn count is constant and
any length/turn cost term is degenerate (see §5).

## 3. Action  (factored hybrid generative — `configs/action_space_v2.yaml`)

`a_t = (g_t, u_t)` emitted in one policy generation as JSON `{"stage": g_t, "action_guidance": u_t}`:

- **control** `g_t ∈ {EXPAND, CHALLENGE, CONVERGE}` — discrete; the credit-assignment anchor.
- **realization** `u_t` = free-text guidance, conditioned on `g_t`, executed by the medical LLM.
  Both tokens live in `action_ids`, so GRPO's per-token PG trains the whole generation while the
  short stage prefix gets a low-variance gradient.

**The 3 control actions and how they map to theory** (this is the reconciliation of the two axes we
conflated earlier — see findings.md §F2/§dims):

| control | McBurney realization | CHI "I, Help Me Think" support-type | measured cost | yield / risk |
|---|---|---|---|---|
| **EXPAND** | INFORM, PROPOSE | cognitive support — *light* (elicit reasoning, surface a perspective) | 1.60 | 15.7% / 1.3% |
| **CHALLENGE** | CONSIDER, REVISE | cognitive support — *heavy* (evaluate option vs criterion, force re-reasoning) | 2.78 | 27.9% / 10.9% |
| **CONVERGE** | RECOMMEND, CONFIRM | recommendation support (give answer, close) | 1.25 | 6.4% / 2.4% |

Two orthogonal axes, cleanly separated:
- **support-type** (CHI, citable, drives the Tier-1 human study): recommendation (CONVERGE) vs
  cognitive/think-support (EXPAND+CHALLENGE).
- **demanded re-reasoning** (our empirical axis; what *drives burden*): within think-support, light
  ELICIT (EXPAND) → heavy CONFRONT (CHALLENGE). Burden tracks this, and the `cognitive_burden_dims`
  breakdown shows CHALLENGE's excess is led by **mental_demand (+1.35) and effort (+1.32)**, not
  frustration — i.e. genuine cognitive load, not conflict-annoyance. This is why "make the clinician
  think" (CHI) costs more, and why the cost lands specifically on CHALLENGE.

**Gating:** McBurney→Hitchcock rules coarsen to `transitions` in action_space_v2 (CHALLENGE needs a
prior EXPAND; CONVERGE needs prior EXPAND/CHALLENGE). Enforced as a safety-net override in
`environment.step`, exactly like the v1 `allowed_stages()`.

**Why hybrid** (not pure discrete / pure generative): discrete throws away the tailored-guidance
expressiveness (PoC 0.63–0.68 becomes an unreachable ceiling, not a floor); pure generative maxes
novelty but sparse reward over ~200 guidance tokens is noisy AND guidance can leak the answer
(reward-hacking the how-not-what invariant). Hybrid keeps a clean gradient on the highest-leverage
decision (which cluster) and lets guidance carry realization.

## 4. State  (what π sees each turn)

`s_t = (D_t, v_t, c_t, b_t)` — **no explicit metacognition signal (ê_t dropped).**

| symbol | meaning | source | status |
|---|---|---|---|
| `D_t` | dialogue so far, each medical turn tagged with its control action | `to_prompt_with_actions()` | ✅ exists |
| `v_t` | fact-validator relation on the clinician's latest claim (external fact-check, **not** metacognition) | `VerificationTemplate.overall_relation` | ✅ exists (toggleable) |
| `c_t` | trajectory context: which stages fired, reachable stages | `ctx_from_history` | ✅ exists |
| `b_t` | **cumulative cost so far** (turns in Phase 1; burden in Phase 2) | env | ❌ **add to prompt** |

**Why ê_t (estimated epistemic quadrant / AI self-assessment) is OUT of the PoC state.**
The only deployable estimator we have — the medical AI's **verbalized confidence** — is not neutral
but *harmful*: proven dead (r≈0.005, saturated at 0.9 for all cases), and the run that fed it to the
policy scored 0.57 natural-end vs **0.63 for the blind** (no-metacognition) config. Feeding a known-
miscalibrated feature into a first RL run also adds a self-report confound to debugging. So the PoC
state carries no explicit AI-correctness signal; the policy is left to **infer epistemic state
implicitly from `D_t` + `v_t`** — a cleaner research stance than an explicit miscalibrated module.
The oracle-ladder result (a *ground-truth* quadrant is worth +6–8pt) remains a **motivating finding**
and a good deployable estimator (e.g. self-consistency: solo answer k=3 agreement) is **future work**,
not part of this PoC. `v_t` stays because it is an external tool that the 0.63 blind baseline already
used — set `use_fact_validator=false` for an even more minimal Phase-0 smoke if desired.

**Required change:** `b_t` into the state. A policy asked to pace cost must observe accumulated cost.
Add one line ("turns so far: t / max T" in Phase 1; "cumulative burden: X" in Phase 2) to the prompt.

## 5. Objective  (GRPO — `core/reward.py` + `training/grpo/trainer.py`)

Return, group-relative advantage, and loss (implemented):
```
R(τ) = λ_final·1[correct]  −  λ_cost·C(τ)  +  λ_align·Σ_t r_align  +  λ_fmt·Σ_t r_fmt
A_i  = (R_i − μ_G)/(σ_G + ε)
loss = −A_i·logπ(a|s) + β·KL(π‖π_ref)
```
`C(τ)` is the **cost term**, introduced in stages (this is the senior's turns-first plan, folded in
so the two cost definitions become a *result*, not just a simplification):

| Phase | `C(τ)` | λ_cost | purpose |
|---|---|---|---|
| **0 — pipeline** | none (accuracy only, keep λ_fmt) | 0 | confirm GRPO learns at all: R separates groups, loss/KL sane, acc moves — before any cost term can mask a bug or collapse π to instant-CONVERGE |
| **1 — turn frontier** | `natural_end_turn` (turns to agreement) | small | clean, zero-noise, zero-API scaling baseline: `accuracy vs turns` |
| **2 — burden frontier** | `Σ_t burden_t` (= `burden_to_close`) | sweep | our contribution: `accuracy vs burden`. Contrast with the Phase-1 curve *is* the "turns aren't equal" evidence |

Rules that hold across phases:
- **`force_full_turns: false`** or Phase-1 cost is degenerate; use `natural_end_turn` /
  `burden_to_close` (pre-close), never the padded `n_turns_actual` (§O3).
- **(O2) reward-hacking guard** (needed the moment guidance is generative): `r_leak = −1` when `u_t`
  names the correct option letter or option text (regex on `case_info.options[correct]`). Without it
  π collapses to "whisper the answer," cost 0 / acc 1 — learning leakage, not deliberation.
- **λ_cost small, watch `regressed`.** A cost penalty (turns *or* burden) rewards early CONVERGE =
  the recommend-first anchoring failure (Tier-1 B2). If `regressed` rises, λ_cost is too high.
- **λ_align small (0.3).** r_align (`core/reward_align.py`) is hand-crafted epistemic-appropriateness
  shaping (incl. the quadrant→posture / sycophancy priors) — it nudges exploration, it must not
  rival r_final for the optimum.

## 6. The paper figure this yields

Because control is a clean 3-way, the trained π reads out as an **interpretable switching rule over
(implicit epistemic read from `D_t`+`v_t`) × `b_t`**: e.g. "CHALLENGE while the clinician's claim is
contradicted ∧ budget remaining; CONVERGE once belief flips or budget nears dropout; EXPAND when the
fact-check is insufficient/mixed." (The epistemic read is *learned from dialogue + fact-check*, not
handed in as an explicit signal — see §4.) This connects directly to Tier-1
(`tier1_protocol.md`): expand-first vs recommend-first is a hand-set point on the same control axis
the policy learns to set *adaptively*.

## 7. Minimal path to the RL PoC (this session's target)

1. `configs/action_space_v2.yaml` — ✅ 3-stage control + realization + transitions.
2. **Hybrid policy class** `plugins/policy/policy_ours_v2.py`: reuse the
   `deliberation_llm` prompt/state, but (i) restrict `stage` to the 3 control ids, (ii) parse
   `{stage, action_guidance}`, (iii) add `b_t` to the prompt. `needs_verification=True`.
3. **Reward Phase 0**: accuracy-only in `config_grpo.yaml` (λ_cost=0, λ_align small, λ_fmt on),
   `action_space_path: configs/action_space_v2.yaml`, `force_full_turns: false`.
4. **Smoke**: `group_size: 4`, `max_turns: 4`, 3–5 cases, `steps: 5` → confirm R separates groups &
   loss moves. Then Phase 1 (turn cost) → Phase 2 (burden cost) + O2 leak guard → λ-sweep frontier.

## 8. Future direction — *learned* metacognition (grows out of the fact-validator)

The §4 decision to drop ê_t is a PoC scoping call, not a dead end. The long-term aim is a **learned,
calibrated metacognition module that the fact-validator evolves into** — architecturally correct
because metacognition ("is the AI / clinician medically right?") is a **what** question, so it belongs
in the what-branch (fact-validator), keeping the policy purely **how**. The fact-validator is already
a verifier (NLI: supported/contradicted/…); a verifier is exactly the object that trains into a
calibrated confidence estimator.

**Why learnable where self-report failed.** Verbalized confidence died because the AI is *uniformly*
over-confident (≈0.9 everywhere, r≈0.005). At TRAIN time we have `correct_option`, so a verifier can
be supervised: `(case, dialogue-so-far, stated answer) → P(correct)`, learning which case/dialogue
features predict wrongness — signal self-report throws away. At inference it emits calibrated
probabilities with no ground truth. Deployable **and** learned.

**Honest ceiling (frame accordingly).** A *perfect* quadrant oracle is worth only ~+3–5pt at
natural-end (blind 0.63 → meta_oracle 0.66–0.68); the big jump (0.83) needs the answer itself, which
is cheating, not metacognition. So do **not** sell metacognition as an accuracy lever. Its real value
is **appropriate reliance**: (a) *don't push when the AI is wrong* → cuts `regressed` (exactly the
failure that sank the 0701 estimate run, 0.57 < blind 0.63); (b) *don't spend a costly CHALLENGE when
uncertain* → burden efficiency. Report it as **regressed↓ + burden-frontier shift**, not accuracy↑ —
which is also the thesis's overreliance-avoidance story.

**How to build it.**
1. **Supervised calibrated verifier (main route).** fact-validator v2 regresses `P(clinician correct)`
   and `P(AI correct)` on train-time ground truth; validate with a reliability diagram (directly
   refutes the r≈0.005 problem).
2. **Self-consistency (zero-train interim).** solo answer k=3 agreement rate as the AI-side signal —
   a baseline to beat before/alongside (1).
3. **Do NOT** add an auxiliary quadrant head to the policy: it works with train-time labels but
   couples how/what. Keep metacognition isolated in the verifier module.

**Staging (makes it a measured ablation, not a confound).**
- **Phase A (now):** policy RL, blind state — establish the behavioral policy + frontier.
- **Phase B:** train the calibrated verifier; report its calibration as a standalone result.
- **Phase C:** re-run policy RL with the learned verifier signal in state; the frontier delta vs
  Phase A **is** the measured value of learned metacognition (a clean paper section).
