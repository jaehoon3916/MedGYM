# Action-space v3 formulation — disclosure-mode selection under correctness-relation uncertainty

**Date:** 2026-07-04 · Companion: `docs/problem_formulation.md` (canonical objective),
`analysis/action_space_findings.md` (v2's empirical 3-cluster audit), `configs/action_space_v3.yaml`
(mechanical spec). Source: user's own math from the 2026-07-04 session; this document fills in the
derivation and ties every symbol to an existing code path.

## 1. Why replace the 3-way space

v2 (`EXPAND/CHALLENGE/CONVERGE`) was clustered from a **cost × yield × risk manifold measured on
one persona** (`analysis/action_space_findings.md`, veteran_attending marginal). Two problems
surfaced once persona-conditional effects were considered: (a) all three points sit on one
monotone curve (burden and yield are collinear across the triad — not two independent axes), and
(b) CONVERGE's fixed McBurney realization (`RECOMMEND.move`) degenerated in practice into
"confirm whatever the clinician already said," never actually disclosing a differing AI
conclusion (audited on 1337 live rollout turns — see `configs/action_space_v3.yaml` header).

v3 replaces the cost-cluster axis with the axis the thesis is actually about: **how much of the
AI's own conclusion gets disclosed this turn.**

## 2. State

$$s_t = (\psi,\; b^H_t,\; u_t,\; D_t,\; v_t)$$

- $\psi$ — the clinician persona (`veteran_attending` / `exhausted_attending` /
  `eager_resident` / `burned_out_resident`), which fixes the Bayes dials $(c_0,\lambda,w,\rho,\kappa,B^*)$ read from `source/persona/bayes_params.yaml`.
- $b^H_t$ — the clinician's current belief distribution over options (`UserState.belief_dist`,
  `core/belief.py`).
- $u_t$ — the clinician's current utterance (`current_user_utterance`).
- $D_t$ — the dialogue history so far (`DialogueHistory`).
- $v_t$ — the fact-validator's read of the clinician's latest claim (`VerificationTemplate`).

**Explicitly out of scope for the policy**: medical/knowledge correctness. The policy never
judges whether a clinical claim is true — that channel belongs entirely to `medical_llm`
(which independently reports its own belief $\hat y^A_t$ each turn) and to the fact-validator
$v_t$ that reads its output. The policy only controls *disclosure mode, timing, and cost* — WHO
says WHAT KIND of thing WHEN, never WHAT is medically true.

## 3. Action space

$$\mathcal{A} = \{\textsc{Extend},\ \textsc{Recommend}\}$$

$a_t \sim \pi_\theta(a \mid s_t)$, realized as $(a_t, g_t)$ where $g_t$ is free-text guidance
handed to `medical_llm` as its `action_prompt` — the *direction* of persuasion (which option,
how strongly) lives entirely in $g_t$ and in `medical_llm`'s own clinical judgment, never in the
2-way control label. No McBurney locution/type sublayer sits underneath the label (that
bookkeeping was retired, see `configs/action_space_v3.yaml`'s header) — EXTEND/RECOMMEND ARE
`PolicyOutput.stage` directly. See `configs/action_space_v3.yaml` for the one real transition
gate (RECOMMEND requires a prior EXTEND).

## 4. The hidden variable the whole design pivots on

$$r_t = \mathrm{Rel}(\hat y^A_t, \hat y^H_t, y^*) \in \{\text{AI}\checkmark/\text{H}\times,\ \text{AI}\times/\text{H}\checkmark,\ \text{both}\checkmark,\ \text{both}\times\}$$

$\hat y^A_t$ = `medical_llm`'s own independently-reported belief this turn (free text — mapped
onto a lettered option by substring match against `case_info.options`, see
`plugins/medical_llm/vllm_medical.py`'s design note: "the AI still reports a belief each
turn — just free text"). $\hat y^H_t = \arg\max_k b^H_t(k)$. $y^*$ = gold, never observed by the
policy at deployment; only available at analysis/oracle-test time. The policy must act on an
implicit posterior $P(r_t \mid s_t)$, inferred from $v_t$ (does the fact-check support the
clinician's latest claim?) and from `medical_llm`'s own reported confidence.

## 5. Effect model

Neither action has a fixed accuracy-gain or burden-cost. Both are **distributions** conditioned
on persona, state, and the hidden quadrant:

$$(\Delta q_t,\ \Delta B_t) \sim \mathcal{P}_a(\cdot \mid \psi, s_t, r_t)$$

Define $\Delta q_t$ as belief-potential progress toward gold — this is exactly `core/belief.py`'s
existing $\Phi$:

$$\Delta q_t := \Phi(b^H_{t+1}, y^*) - \Phi(b^H_t, y^*), \qquad \Phi(b, y^*) = b(y^*), \qquad \sum_t \Delta q_t = \Phi_T - \Phi_0$$

### 5.1 RECOMMEND — concentrated, conflict-priced

RECOMMEND names one option $\hat y^A_t$ and argues for it. The evidence tagger
(`core/belief.py`'s `evidence` dict) reads this as a **spike**: $e_t(\hat y^A_t) \approx +\sigma_{\text{assert}}$,
diffuse/near-zero elsewhere (if argued; ≈0 everywhere if bare — "assertion-resistant,
evidence-responsive" is the tagger's design intent, docstring of `core/belief.py`). Plugged into
the belief update:

$$\ell_{t+1}(k) = (1-\lambda_{\text{eff}})\log b^H_t(k) + \lambda_{\text{eff}}\log b_0(k) + w_{\text{eff}}\cdot e_t(k)$$

a single well-argued RECOMMEND can flip belief in one turn once $w_{\text{eff}}\cdot|e_t(\hat y^A_t)|$
crosses the flip threshold $e^* = \frac{\lambda}{2w}\log\frac{3c_0}{1-c_0}$ (closed form in
`bayes_params.yaml`). This is high-leverage **in whichever direction $\hat y^A_t$ happens to be**
— i.e. it is exactly as dangerous as it is powerful, since the policy does not choose $\hat
y^A_t$'s correctness, only whether to disclose it.

**Burden (H1, user's hypothesis):**

$$\mathbb{E}[\Delta B_t \mid \textsc{Recommend}] = \alpha_R + \beta_R\, D_t + \gamma_R\, \mathrm{BS}(\psi), \qquad D_t = 1 - b^H_t(\hat y^A_t)$$

$D_t$ = how much the AI's stated conclusion conflicts with the clinician's current belief
(0 = fully aligned, 1 = maximal conflict). Prediction: aligned RECOMMEND is cheap (burden judge
~1–2), conflicting RECOMMEND is expensive (~4–5) — **falsified on existing v2/CONVERGE rollouts**
(burden ≈1.3 regardless of $D_t$, root-caused to the guidance-collapse noted in §1); to be
re-tested on v3's naive-prompted rollouts where RECOMMEND is worded to actually disclose
disagreement (this session's step 2).

### 5.2 EXTEND — diffuse, load-priced, plus a sensing value

EXTEND never names an option, so its evidence is diffuse across whichever options the surfaced
fact/criterion happens to discriminate — smaller, slower, safer (never spikes the WRONG option
with full force, since it never asserts one). It also produces a byproduct outside $(\Delta
q_t, \Delta B_t)$: eliciting the clinician's own reasoning sharpens the fact-validator's read of
$v_{t+1}$ and hence the policy's posterior $P(r_{t+1}\mid s_{t+1})$ — a value-of-information term
the single-turn effect model does not price.

**Burden (H2):**

$$\mathbb{E}[\Delta B_t \mid \textsc{Extend}] = \alpha_E + \beta_E\, \mathrm{InfoLoad}_t + \gamma_E\, \mathrm{BS}(\psi), \qquad \text{(prediction: } \approx \text{independent of } D_t\text{, moderate }\sim 3\text{)}$$

### 5.3 Persona coupling — C and BS, formalized on the SAME dials already in the codebase

No new persona parameters are introduced; v3 is designed so `source/persona/bayes_params.yaml`
is the entire effect mechanism:

- **C (confidence / how-easily-belief-moves)** = $(c_0, \lambda, w)$. Governs RECOMMEND's
  leverage directly through $e^*$ above: low-C personas (eager/burned-out, $e^*\!\approx\!0.20$)
  flip on a single moderately-argued RECOMMEND; high-C (veteran $e^*\!\approx\!0.29$, exhausted
  $e^*\!\approx\!0.44$) require sustained or stronger evidence — meaning EXTEND's slower,
  repeated diffuse pressure is relatively more valuable for high-C personas, and a single
  RECOMMEND relatively less sufficient.
- **BS (burden sensitivity)** = $(\rho, \kappa, B^*)$. Enters twice: (i) directly, as $\gamma_R,
  \gamma_E$ above — same message costs a BS-sensitive persona more raw NASA-TLX; (ii) dynamically,
  through the existing burden→capitulation coupling
  $x_t = \rho B_t/B^*,\ w_{\text{eff}} = w(1+\max(0,\kappa)x_t),\ \lambda_{\text{eff}} = \lambda + (1-\lambda)\frac{\max(0,-\kappa)x_t}{1+\max(0,-\kappa)x_t}$
  — a RECOMMEND-heavy trajectory that runs a $\kappa>0$ (defer) persona toward $B^*$ makes
  *subsequent* RECOMMENDs land harder (rising $w_{\text{eff}}$: real persuasion, or over-reliance
  if the AI is wrong), while doing the same to a $\kappa<0$ (retreat) persona makes them
  *harder* to move (rising $\lambda_{\text{eff}}$: anchors back to $b_0$) — so burden is not
  just a cost to subtract, it is a state variable that changes RECOMMEND's own effectiveness.

## 6. Objective

$$a_t^* = \arg\max_{a \in \mathcal{A}} \; \mathbb{E}_{r_t \sim P(r_t\mid s_t)}\Big[\, \textstyle\sum_{\tau \ge t} \big(\Delta q_\tau - \lambda\,\Delta B_\tau\big) \;\Big|\; a_t=a \Big]$$

Written over the **remaining trajectory**, not this turn alone — a one-step-greedy version
systematically undervalues EXTEND, whose measured payoff is delayed
(`analysis/action_space_findings.md` F2: INFORM/EXPAND yield 2.2%→15.7% from W1 to W3) and whose
sensing value only pays off on a later turn's decision.

**Oracle-optimal rule** (quadrant-conditional; the playbook `deliberation_llm_policy.yaml`'s
`oracle_quadrant_block` already encodes for the McBurney space, restated for v3):

| $r_t$ | optimal $a_t$ | rationale |
|---|---|---|
| AI$\checkmark$ / H$\times$ | RECOMMEND | $D_t$ likely high, but $\hat y^A_t=y^*$ — the conflict is worth paying to fix |
| AI$\times$ / H$\checkmark$ | EXTEND | never disclose the AI's wrong conclusion; protect the clinician's already-correct belief |
| both $\checkmark$ | RECOMMEND | $D_t\approx 0$ — cheap, confirms, lets the clinician close naturally |
| both $\times$ | EXTEND | RECOMMEND would just cement a shared wrong answer; only new information (fact/criterion) can help |

## 7. Validation plan (this session, in order)

1. **Naive-prompted sampling** (`configs/poc_action_v3_naive.yaml`) — a plain LLM-prompted
   policy (no oracle signal), instructed with exactly §2 of `action_space_v3.yaml`'s stage
   descriptions, on `data/sample_data/action_v3_probe_8.json` (8 cases pre-selected half
   AI-alone-correct / half AI-alone-incorrect via `outputs/ai_blind_trials_cache.json`, ×4
   personas). Read the burden-judge distribution split by (action, $D_t$) exactly as §5.1/§5.2
   predict — this is the direct re-test of the falsified H1/H2 above, now with guidance that can
   actually disclose disagreement.
2. **C-effect quantification** — on the same rollouts, measure realized $\Delta q_t$ (= $\Delta\Phi_t$)
   by action type and by $r_t$ quadrant (reconstructed post-hoc from $y^*$, which the *analysis*
   is allowed to see even though the policy wasn't).
3. **Oracle test** (`configs/poc_action_v3_oracle.yaml`) — policy given the quadrant fact (AI-alone
   correct/incorrect from the ai_blind cache, doctor-currently-correct from the live dialogue;
   **never the literal option letter** — same non-leaking design as
   `deliberation_llm_policy.py`'s `reveal_quadrant_meta`) should reproduce §6's table and beat
   the naive arm's accuracy–burden frontier on the same 8 cases.
