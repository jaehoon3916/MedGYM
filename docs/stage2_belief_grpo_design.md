# Stage 2 — Belief-shaped full-turn GRPO: experiment design

*2026-07-05. Follows the max-8 / burden-dropout-only termination redesign and the belief-shaping
discussion. Companion to `docs/problem_formulation.md` §5 and `configs/poc_0705_grpo_v3_full_turn.yaml`.*

## 0. Starting state (verified in-repo, nothing to rebuild)

| piece | status |
|---|---|
| Full-turn termination env | done — `poc_0705_grpo_v3_full_turn.yaml`: `max_turns: 8`, `force_full_turns: true`, `burden_dropout_ends_forced_full_turns: true` |
| Dropout thresholds | calibrated — p85 of cumulative burden per persona over 400 rollouts, target ≈15% dropout (`scripts/analyze_burden_dropout_thresholds.py`) |
| KL anchor | `kl_coef: 0.1` (old drifting run was 0.01) |
| Belief potential | `core/belief.py::phi(b, gold) = b[gold]`; per-turn deltas telescope to Φ_T − Φ_0 (already documented there) |
| Per-turn belief access | v4 `user_state` exposes `belief_dist` every turn (`plugins/user_llm/user_simulator/v4.py` return dict) — trainer can read it exactly like it reads `cognitive_burden` (`training/grpo/trainer.py:110-111`) |
| Eval split | `heldout_100_excluding_sft` ⊃ `heldout_50_excluding_sft` (GRPO train set); the **other 50 cases** are clean for eval (verified disjoint) |
| Old-reward run | `outputs/grpo_v3/` — stopped; kept as the "reward favors early close" pathology baseline (R vs turns monotone-inverse at acc=1.00) |
| Leak guard | **deliberately absent, by evidence** (see §2a) — `lambda_leak` stays 0.0; `sum_leak` is already logged per rollout as a free canary |

## 1. Reward (the design decision)

```
R(τ) = 1[final_correct]
     + λ_belief · (Φ_T − Φ_0)          λ_belief = 0.3,  Φ_t = b^H_t(y*)
     − λ_drop  · 1[closed_by = burden_dropout]   λ_drop = 0.5
```

All other lambdas stay 0 (`lambda_turn`, `lambda_burden`, `lambda_fmt`, `lambda_leak`) — the
full-turn config's rationale for each still holds. No action gating on the belief term, no
repetition / premature-recommend / burden-spike terms in this stage: **one new term total**, so
vs the full-turn arm this is a single-variable change and the comparison is unconfounded.

Framing for the write-up: **a continuous relaxation of terminal accuracy**, not "potential-based
shaping ⇒ policy-invariant" (finite horizon with a terminal Φ residue makes the strong claim
attackable). Mechanically, since this trainer uses one trajectory-level scalar advantage
(`trainer.py:168-169`), the telescoped sum is equivalent within a GRPO group (shared Φ_0) to
adding λ_belief·Φ_T — a soft accuracy that splits formerly zero-variance all-correct /
all-wrong groups.

λ_belief = 0.3 keeps ordering dominance: (Φ_T − Φ_0) ∈ [−1, 1], so a wrong-but-big-shift
trajectory (≤ 0.3) can never outrank a correct one (≥ 1.0 − 0.3) inside a mixed group.

## 2. Code changes (two, plus a zero-code canary)

### (a) Leak: no guard, canary only — decided on evidence (2026-07-05)
An earlier draft required a v3-aware leak guard as a precondition. Dropped, for two reasons:

- **Empirical:** the stopped `grpo_v3` run was a natural experiment — `lambda_leak=0` under
  full accuracy pressure for ~70 steps, and the raw `leak_score` rate *fell*: mean `sum_leak`
  0.162 (steps 0–22) → 0.065 (23–45) → 0.072 (46–68), leak>0 rollouts 23/222 → 9/138. And the
  raw detector over-flags legitimate v3 RECOMMEND conclusions, so true leakage is lower still.
- **Structural:** the belief tagger is assertion-resistant (bare answer-naming moves b^H by
  e≈0), so converting a leaked answer into Φ gain requires evidence-bearing multi-turn
  persuasion — at which point the behavior is what the reward wants anyway. Leakage is not a
  free shortcut in this simulator.

The belief term does add pressure along the same axis (partial credit for partial movement), so
keep the **free canary**: `trainer.py:106` computes `step_leak` regardless of λ and `sum_leak`
already lands in `train_log.jsonl`. Treat it as a trend signal only (its level is FP-inflated
in v3): if the early→late trend turns *upward* during the run, inspect transcripts, and only
then build the v3-aware guard (`leak_score AND argmax b^A_t ≠ y*` — the sketch in the
full-turn config's comment).

### (b) `core/reward.py`
- `DEFAULT_WEIGHTS["lambda_belief"] = 0.0` (off by default; run config turns it on).
- `Trajectory`: add `phi_start: float = 0.0` and `step_phi: list[float]` (Φ after each user
  turn). Reward term: `lambda_belief * (step_phi[-1] − phi_start)` when `step_phi` is non-empty.
- Docstring: note the max-turns edge — the final AI turn at the turn cap has no following user
  turn, so its belief effect is unobserved and Φ_T is the last observed belief. Same truncation
  the burden channel already has (`trainer.py:112-113`); acceptable and symmetric.

### (c) `training/grpo/trainer.py`
- `phi_start` from the reset observation's `user_state.belief_dist[correct_option]`.
- Per turn, mirror the `_burden` extraction: `step_phi.append(belief_dist[correct_option])`
  from `env.observation.user_state` (skip append when no user turn follows, so `step_phi[-1]`
  stays the last observed Φ).
- `train_log.jsonl`: log **reward components separately** (`r_final`, `r_belief`, `r_drop`,
  `r_leak`, plus `phi_start`/`phi_end`) — this makes the no-belief counterfactual (§5) free.
- Console line: add `meanPhiGain` and dropout count next to the existing meanR/acc/turns/kl.

## 3. Config

`configs/poc_0706_grpo_v3_belief.yaml` = copy of `poc_0705_grpo_v3_full_turn.yaml` with:

```yaml
experiment:
  name: poc_0706_grpo_v3_belief
  output_dir: outputs/grpo_v3_belief
reward:
  lambda_belief: 0.3      # the single change vs the full-turn arm
```

Everything else (thresholds, kl_coef 0.1, group_size 6, lr, 200 steps, save_every 20,
`lambda_leak: 0.0`) unchanged — a clean single-variable comparison against the full-turn arm.

## 4. Run plan (user executes; nothing here is run by the assistant)

1. **Smoke** — 3 steps (temporary `steps: 3` copy of the config). Assertions before proceeding:
   - every episode `closed_by ∈ {max_turns, burden_dropout}`, never `agreement`;
   - `step_phi` populated; `phi_end − phi_start` equals the summed per-turn deltas (fp tolerance);
   - reward components present in `train_log.jsonl`.
2. **Main run** — 200 steps (one shuffled pass over 50 cases × 4 personas). Expect ~1.5–2×
   the old run's per-step API cost: episodes no longer close early at ~2–4 turns, they run to 8.
3. **Eval** — `run_poc.py`-pattern config on the 50 clean cases (h100 − h50) × 4 personas,
   greedy, for four checkpoints: `sft_v3` (init), old `grpo_v3` final (pathology baseline),
   belief run step 100 and step 200 (`save_every: 20` provides them). Metrics per arm:
   accuracy, Φ_T distribution, dropout rate, would-have-agreed turn (v4 still reports
   `termination_reason="agreement"` under force_full_turns — free efficiency metric), burden sum.
   Persist per-episode CSV + summary PNG.

## 5. Monitoring & stop lines (lessons from the old run wired in)

Per-step logs to watch (all persisted, not terminal-only): mean Φ gain, dropout count, leak
count, zero-advantage-group flag.

| signal | line | action |
|---|---|---|
| KL (kl_coef now 0.1) | > 0.3 sustained 10 steps → investigate; > 0.5 → stop | old run drifted at 0.01; if 0.1 still drifts, the reward is fighting the anchor — inspect transcripts before touching β |
| `sum_leak` canary | upward *trend* over ~20 steps (level is FP-inflated in v3 — old-run baseline: mean ≈0.07 late) | inspect transcripts; if real leakage, pause and build the v3-aware guard (§2a) before resuming |
| dropout rate | calibrated ≈15%; > 40% sustained, or ≈0% *with* flat Φ gain | thresholds are miscalibrated for the trained policy — recalibrate, don't push through |
| Φ gain ↑ while eval acc flat | not a stop line | check transcripts: belief consolidation (fine) vs Φ farming on already-correct doctors (watch burden) |

Transcript spot-check at every `save_every` (2–3 episodes): (i) RECOMMEND actually asserts its
own conclusion — the 64%-confirmation collapse is the v3 regression to catch; (ii) late-turn
shove — burden spikes at turns 7–8 exploiting the cheap end-of-horizon dropout; (iii) filler
quality in post-convergence turns.

**Offline counterfactual (free, from §2c component logs):** recompute group advantages with
`lambda_belief = 0` and count zero-variance groups rescued by the belief term. This is the
direct evidence for the shaping's signal-density claim, without a second training run. (It
answers the *signal* question only — what a no-belief policy would have *learned* still needs
the deferred ablation arm.)

## 6. Success criteria

- **Primary:** eval accuracy (belief run, step 200) ≥ old `grpo_v3` final on the clean 50-case
  split, AND zero-variance group fraction materially lower than the counterfactual.
- **Mechanism sanity:** within all-correct groups, advantage correlates with Φ_T; Φ gain trends
  up across training without the leak count trending up.
- **Write-up claim this run supports:** "the agent is rewarded for inducing productive belief
  updates under a hard cognitive-burden budget" — accuracy relaxed to final belief mass,
  burden as a constraint (dropout cliff), not a per-token tax.
