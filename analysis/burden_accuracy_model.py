"""Core (y, r, c) markov model behind docs/burden_accuracy_tradeoff_poc.md.

Calibration source: analysis/action_space_findings.md (deepseek-v3.2, veteran_attending,
300 episodes, deliberation_llm policy, W3 yield / W1 regression / next-turn burden).

Reproduces the numbers in Prop 1-3: fixed points p*_a, ceiling p_bar, burden price pi_a(p),
efficiency crossover p_cross, and the discrete-turn frontier simulation. Pure stdlib --
no matplotlib/numpy needed (the charts themselves were rendered separately via an HTML/SVG
page + headless Chrome, since this environment currently has no matplotlib install; see the
POC doc's header for why).
"""
from __future__ import annotations

ACTIONS = {
    "PROBE":     dict(y=0.157, r=0.013, c=1.60),
    "CHALLENGE": dict(y=0.279, r=0.109, c=2.78),
    "CONVERGE":  dict(y=0.064, r=0.024, c=1.25),
}
for _a, _d in ACTIONS.items():
    _d["pstar"] = _d["y"] / (_d["y"] + _d["r"])
P_BAR = max(d["pstar"] for d in ACTIONS.values())


def delta_p(action: str, p: float) -> float:
    d = ACTIONS[action]
    return (1 - p) * d["y"] - p * d["r"]


def burden_price(action: str, p: float) -> float | None:
    dp = delta_p(action, p)
    return ACTIONS[action]["c"] / dp if dp > 1e-9 else None


def efficiency(action: str, p: float) -> float:
    return delta_p(action, p) / ACTIONS[action]["c"]


def efficiency_crossover(a: str, b: str, lo: float = 0.0, hi: float = 1.0, iters: int = 200) -> float:
    """Bisection root of efficiency(a,p) - efficiency(b,p) == 0."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if efficiency(a, mid) > efficiency(b, mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate(policy_fn, p0: float, max_turns: int = 400, stop_eps: float = 1e-4):
    """Discrete per-turn rollout of the belief-flip markov chain under `policy_fn(p) -> action`.
    Returns [(cumulative_burden, p_t), ...]. Stops early once a non-CHALLENGE action's Delta p
    is negligible (converged to its fixed point)."""
    p = p0
    traj = [(0.0, p)]
    burden = 0.0
    for _ in range(max_turns):
        a = policy_fn(p)
        dp = delta_p(a, p)
        if abs(dp) < stop_eps and a != "CHALLENGE":
            break
        p = min(max(p + dp, 0.0), 1.0)
        burden += ACTIONS[a]["c"]
        traj.append((burden, p))
    return traj


def rollout_plan(p0: float, plan: str) -> tuple[float, float]:
    """Apply a fixed action sequence (e.g. 'HHHPPPPP', H=CHALLENGE P=PROBE) from p0.
    Returns (final_p, total_burden). Used for the Prop-5 turn-budget enumeration."""
    names = {"P": "PROBE", "H": "CHALLENGE", "V": "CONVERGE"}
    p, burden = p0, 0.0
    for ch in plan:
        a = names[ch]
        p = min(max(p + delta_p(a, p), 0.0), 1.0)
        burden += ACTIONS[a]["c"]
    return p, burden


def turn_budget_table(p0: float = 0.05, T: int = 8):
    """Prop 5: with a finite turn budget T, enumerate m CHALLENGE turns (used FIRST -- order
    matters, see below) followed by T-m PROBE turns. Returns [(m, accuracy, burden), ...]."""
    return [(m, *rollout_plan(p0, "H" * m + "P" * (T - m))) for m in range(T + 1)]


PERSONA_CEILINGS = {
    # re-estimated via analysis/persona_yrc.py on GRPO phase1+phase2 rollouts (ours_v2, evolving
    # policy mid-training -- absolute yields differ from the controlled audit above; only the
    # relative persona ordering is asserted as robust).
    "burned_out_resident": 0.954,
    "eager_resident": 0.843,
    "veteran_attending": 0.743,
    "exhausted_attending": 1.000,  # small-n artifact (PROBE r=0/n, CH/CV n=12/14) -- not trusted
}


if __name__ == "__main__":
    pstars = ", ".join(f"{a}={d['pstar']:.3f}" for a, d in ACTIONS.items())
    print(f"p*_a: {{ {pstars} }}")
    print(f"p_bar (accuracy ceiling) = {P_BAR:.4f}")
    p_cross = efficiency_crossover("CHALLENGE", "PROBE")
    print(f"efficiency crossover rho_CHALLENGE(p)=rho_PROBE(p) at p = {p_cross:.4f}")
    for p in (0.0, 0.5, 0.70, 0.715):
        print(f"  burden_price(CHALLENGE, p={p:.3f}) = {burden_price('CHALLENGE', p):.2f}"
              if burden_price("CHALLENGE", p) else f"  p={p} beyond CHALLENGE's fixed point")

    p0 = 0.05
    always_challenge = simulate(lambda p: "CHALLENGE", p0, 60)
    always_probe = simulate(lambda p: "PROBE", p0, 400)
    switched = simulate(lambda p: "CHALLENGE" if p < p_cross else "PROBE", p0, 400)
    print(f"\np0={p0}: always_challenge final = {always_challenge[-1]}")
    print(f"p0={p0}: always_probe     final = {always_probe[-1]}")
    print(f"p0={p0}: switched         final = {switched[-1]}")
    print(f"switch saves {always_probe[-1][0] - switched[-1][0]:.2f} burden units vs pure PROBE "
          f"(never worse, but small: CHALLENGE's efficiency edge window is narrow and one "
          f"discrete CHALLENGE turn overshoots most of it)")

    # Prop 5: finite turn budget restores planning (interior optimum + order dependence).
    T = 8
    print(f"\n[Prop 5] turn budget T={T}, p0={p0}: m CHALLENGE turns first, then PROBE")
    table = turn_budget_table(p0, T)
    for m, acc, b in table:
        print(f"  H*{m} + P*{T - m}:  acc={acc:.4f}  burden={b:.2f}")
    m_opt, acc_opt, b_opt = max(table, key=lambda t: t[1])
    acc_probe, b_probe = table[0][1], table[0][2]
    acc_rev, _ = rollout_plan(p0, "P" * (T - m_opt) + "H" * m_opt)
    print(f"  optimum m={m_opt}: {acc_opt:.4f} ({(acc_opt - acc_probe) * 100:+.1f}pp vs PROBE-only, "
          f"burden {b_opt - b_probe:+.2f}) -- interior, so the CHALLENGE how-much/when decision is real")
    print(f"  order dependence: same multiset H-first {acc_opt:.4f} vs H-last {acc_rev:.4f} "
          f"(consistent with p_cross={p_cross:.3f}: CHALLENGE's value is escaping LOW p fast)")
