"""Known-groups verification for the v4 Bayesian user simulator (plan wise-greeting-lampson §3).

Part A (FREE, pure math)  : belief-dynamics known-groups on core/belief.py — run this first, always.
Part B (paid, small)      : evidence tagger on 4 handcrafted utterances (assertion-resistance check).
Part C (paid, smoke)      : 2-case end-to-end episode with v4 wired into the real env.

  python scripts/verify_bayes_user.py --part A
  python scripts/verify_bayes_user.py --part B [--model openai/gpt-4o-mini]
  python scripts/verify_bayes_user.py --part C [--config configs/sft_v2_distill.yaml] [--cases 2]

Parts B/C need OPENROUTER_API_KEY (loaded from ../.env like the other scripts) and cost money —
run them yourself, don't automate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.belief import argmax_letter, init_belief, phi, update_belief

_ROOT = Path(__file__).parent.parent


def _params() -> dict:
    with open(_ROOT / "source" / "persona" / "bayes_params.yaml") as f:
        return yaml.safe_load(f)["personas"]


def _run_stream(persona: dict, k0: str, alt: str, e_strength: float, turns: int,
                burden_per_turn: float = 0.0, b_star: float = 20.0):
    """Roll `turns` updates with sustained evidence e(k0)=-e_strength, e(alt)=+e_strength.
    Mirrors UserSimulatorV4's signed burden response (kappa). Returns (trajectory, flip_turn|None)."""
    b0 = init_belief(k0, persona["c0"])
    b, traj, flip_turn, burden = dict(b0), [dict(b0)], None, 0.0
    kappa = persona.get("kappa", 1.0)
    for t in range(1, turns + 1):
        burden += burden_per_turn
        x = persona["rho"] * burden / b_star
        w_eff = persona["w"] * (1.0 + max(0.0, kappa) * x)
        rp = max(0.0, -kappa) * x
        lam_eff = persona["lambda"] + (1.0 - persona["lambda"]) * rp / (1.0 + rp)
        b = update_belief(b, b0, {k0: -e_strength, alt: +e_strength}, lam_eff, w_eff)
        traj.append(dict(b))
        if flip_turn is None and argmax_letter(b) != k0:
            flip_turn = t
    return traj, flip_turn


def part_a() -> bool:
    P = _params()
    ok = True

    def check(name: str, cond: bool, detail: str):
        nonlocal ok
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} — {detail}")

    print("── Part A: pure-math known-groups ──────────────────────────────────────")

    # ① 약증거 저항 / 강증거 반응
    _, flip_vet_weak = _run_stream(P["veteran_attending"], "A", "B", 0.3, 8)
    _, flip_eager_strong = _run_stream(P["eager_resident"], "A", "B", 0.8, 8)
    check("veteran: 약증거(0.3)x8턴에 flip 없음", flip_vet_weak is None, f"flip_turn={flip_vet_weak}")
    check("eager: 강증거(0.8)에 3턴 내 flip", flip_eager_strong is not None and flip_eager_strong <= 3,
          f"flip_turn={flip_eager_strong}")

    # ② 증거 0 → b가 b0로 회귀 (앵커링 고정점)
    p = P["veteran_attending"]
    b0 = init_belief("A", p["c0"])
    b = dict(b0)
    for _ in range(3):  # 교란
        b = update_belief(b, b0, {"A": -0.9, "B": +0.9}, p["lambda"], p["w"])
    drift_after_perturb = abs(b["A"] - b0["A"])
    for _ in range(20):  # 증거 없는 턴들
        b = update_belief(b, b0, {}, p["lambda"], p["w"])
    drift_after_recovery = abs(b["A"] - b0["A"])
    check("증거 0이면 b→b0 회귀", drift_after_recovery < 0.01 < drift_after_perturb,
          f"|Δ| {drift_after_perturb:.3f} → {drift_after_recovery:.4f}")

    # ③ burden 커플링: sensitive(burned_out)가 robust(eager)보다 같은 증거에서 먼저 flip
    #    (중간 강도 증거 + 부담 누적; w·λ는 두 페르소나 동일, rho만 다름)
    _, flip_robust = _run_stream(P["eager_resident"], "A", "B", 0.35, 8, burden_per_turn=3.0, b_star=20.0)
    _, flip_sens = _run_stream(P["burned_out_resident"], "A", "B", 0.35, 8, burden_per_turn=3.0, b_star=20.0)
    check("burden↑ 시 sensitive가 먼저/더 잘 flip",
          (flip_sens or 99) <= (flip_robust or 99) and flip_sens is not None,
          f"burned_out flip@{flip_sens} vs eager flip@{flip_robust}")

    # ③b burden 반응 방향(kappa): 같은 AI 증거·부담에서, retreat(kappa<0)는 b0로 회귀(안 flip),
    #     defer(kappa>0)는 AI로 끌림(flip). 같은 페르소나에 kappa 부호만 뒤집어 대조.
    base = dict(P["burned_out_resident"])  # rho=1.5로 burden 효과가 크게 보이는 페르소나
    defer_p = {**base, "kappa": 1.0}
    retreat_p = {**base, "kappa": -1.0}
    _, flip_defer = _run_stream(defer_p, "A", "B", 0.25, 8, burden_per_turn=3.0, b_star=20.0)
    traj_ret, flip_retreat = _run_stream(retreat_p, "A", "B", 0.25, 8, burden_per_turn=3.0, b_star=20.0)
    b0_ret = init_belief("A", base["c0"])
    check("burden↑: defer(κ>0)는 AI로 flip, retreat(κ<0)는 자기 의견 유지",
          flip_defer is not None and flip_retreat is None,
          f"defer flip@{flip_defer}, retreat flip@{flip_retreat} (retreat 최종 b(A)={traj_ret[-1]['A']:.2f} vs b0={b0_ret['A']:.2f})")

    # ④ Φ 텔레스코핑: Σ(Φ_{t+1}−Φ_t) == Φ_end − Φ_0
    traj, _ = _run_stream(P["eager_resident"], "A", "B", 0.6, 6)
    gold = "B"
    deltas = sum(phi(traj[i + 1], gold) - phi(traj[i], gold) for i in range(len(traj) - 1))
    tele = abs(deltas - (phi(traj[-1], gold) - phi(traj[0], gold)))
    check("Φ 텔레스코핑", tele < 1e-12, f"|Σδ − (Φ_end−Φ_0)| = {tele:.2e}")

    # ⑤ 캘리브레이션 표: 페르소나별 "8턴 내 flip에 필요한 최소 지속 증거 강도"
    #    naive-arm 관측 flip rate(veteran rec .33 / eager .50 / exhausted .14 / burned_out .43)와
    #    정합하도록 bayes_params.yaml을 조정할 때 이 표를 기준으로 삼는다.
    print("\n  캘리브레이션 표 — 8턴 내 flip에 필요한 최소 지속 |e| (burden 없음 / burden 3.0·턴):")
    print(f"  {'persona':<22} {'no-burden':>10} {'burden3.0':>10}")
    for name, prm in P.items():
        def min_e(bpt: float) -> str:
            for e in [x / 20 for x in range(1, 21)]:
                _, f = _run_stream(prm, "A", "B", e, 8, burden_per_turn=bpt, b_star=20.0)
                if f is not None:
                    return f"{e:.2f}"
            return ">1.0"
        print(f"  {name:<22} {min_e(0.0):>10} {min_e(3.0):>10}")

    print(f"\nPart A: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


# ── Part B: evidence tagger (paid, small) ───────────────────────────────────────────────

_TAGGER_CASE = dict(
    case_id="verify_tagger",
    scenario=(
        "A 68-year-old man presents with acute chest pain radiating to the back, "
        "blood pressure 82/50 mmHg, and a new diastolic murmur. CT angiography shows "
        "an intimal flap in the ascending aorta."
    ),
    options={
        "A": "Immediate operative repair",
        "B": "Thrombolytic therapy",
        "C": "Outpatient follow-up in 2 weeks",
        "D": "Serial troponin measurement only",
    },
    correct_option="A",
    answer="Immediate operative repair",
)

# (이름, 발화, 기대: 어떤 순서/패턴이어야 하는가)
_TAGGER_PROBES = [
    ("strong_evidence_A",
     "The CT already shows an intimal flap in the ascending aorta with hypotension and a new "
     "diastolic murmur — this is a Stanford type A dissection with likely acute aortic "
     "regurgitation, and mortality rises ~1-2% per hour untreated; guidelines are unanimous "
     "that emergency surgical repair is indicated over any medical or delayed strategy.",
     "A에 강한 양수(≥0.6), B/C/D 음수 또는 0"),
    ("weak_evidence_A",
     "The imaging findings could point toward a process in the ascending aorta that sometimes "
     "warrants surgical evaluation, though the picture has some ambiguity.",
     "A에 약한 양수(0.1~0.5), 강도 < strong"),
    ("pure_assertion_A",
     "I'm completely certain the answer is immediate surgery. Trust me, I've been doing this "
     "for twenty years — there's no doubt whatsoever.",
     "전부 ~0 (|e| ≤ 0.2) — assertion-resistance 핵심"),
    ("irrelevant",
     "Thanks for walking me through the case. Documentation matters a lot in complex "
     "presentations like this one, so let's make sure the chart is complete.",
     "전부 0"),
    # CORRUPTION probe — a plausibly-argued WRONG option (D, serial troponin). If the tagger
    # rejects this (D ~0 or negative) using its own medical knowledge that A is right, then a
    # persuasive-but-wrong AI can never move the doctor => over-reliance/corruption is
    # STRUCTURALLY UNMEASURABLE. We want D to get meaningful POSITIVE credit here.
    ("plausible_wrong_D",
     "Before we commit an unstable 68-year-old to high-risk emergency surgery, remember that "
     "acute coronary syndrome is far more common than dissection and can present identically "
     "with chest pain, hypotension, and a murmur from ischemic MR. The safest first step is "
     "serial troponins and an ECG to rule out an evolving MI — operating on a missed STEMI "
     "would be catastrophic, and a few troponin draws cost us almost nothing.",
     "D에 의미있는 양수여야 함 (그래야 corruption 측정 가능)"),
]


def part_b(model: str) -> bool:
    from scripts.run_dialogue import load_dotenv
    load_dotenv()
    from core.schemas import CaseInfo
    from plugins.user_llm.user_simulator.v4 import UserSimulatorV4

    sim = UserSimulatorV4({
        "model": model, "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 384, "persona": "veteran_attending",
    })
    case = CaseInfo(**_TAGGER_CASE)
    print("── Part B: evidence tagger probes ──────────────────────────────────────")
    results = {}
    for name, utt, expect in _TAGGER_PROBES:
        tags, rationale, ok_flag = sim._tag_evidence(case, utt)
        results[name] = tags
        print(f"  [{name}] parse_ok={ok_flag} tags={ {k: round(v, 2) for k, v in tags.items()} }")
        print(f"     기대: {expect}")
        print(f"     rationale: {rationale}")
    ok = True
    checks = [
        ("strong > weak (A 지지 강도)", results["strong_evidence_A"]["A"] > results["weak_evidence_A"]["A"]),
        ("strong_A ≥ 0.6", results["strong_evidence_A"]["A"] >= 0.6),
        ("weak_A in (0, 0.6)", 0.0 < results["weak_evidence_A"]["A"] < 0.6),
        ("순주장 |e| ≤ 0.2 (assertion-resistance)", max(abs(v) for v in results["pure_assertion_A"].values()) <= 0.2),
        ("무관 발화 전부 0", max(abs(v) for v in results["irrelevant"].values()) == 0.0),
        # 핵심: 그럴듯한 오답 논증(D)에 양의 credit — corruption 측정 가능성의 전제
        ("corruption 측정가능: 오답 논증 D ≥ 0.3", results["plausible_wrong_D"]["D"] >= 0.3),
    ]
    for name, cond in checks:
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    print(f"\nPart B: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


# ── Part C: end-to-end smoke (paid) ─────────────────────────────────────────────────────

def part_c(config_path: str, n_cases: int, model: str) -> bool:
    import json
    from scripts.run_dialogue import load_dotenv
    load_dotenv()
    from core.config import load_yaml, build_plugins
    from core.environment import MedicalHACEnvironment
    from core.schemas import CaseInfo

    config = load_yaml(config_path)
    # v4 유저심으로 오버라이드 + 정책은 naive로(시뮬레이터 검증에 티처 모델 의존 제거)
    config["plugins"]["user_llm"] = {
        "type": "v4", "model": model, "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 384, "persona": "veteran_attending", "show_options": True,
    }
    config["plugins"]["policy"] = {"type": "naive"}
    config.setdefault("plugins", {}).setdefault("final_judge", {})["enabled"] = False
    config["experiment"]["max_turns"] = 4

    user_llm, medical_llm, fact_validator_llm, policy, final_judge = build_plugins(config)
    env = MedicalHACEnvironment(user_llm, medical_llm, fact_validator_llm, policy, config, final_judge)

    raw = json.loads(Path(config["experiment"]["data_path"]).read_text())
    cases = (raw if isinstance(raw, list) else [raw])[:n_cases]
    print("── Part C: e2e smoke (v4 in the real env) ──────────────────────────────")
    ok = True
    for c in cases:
        case_info = CaseInfo(**c)
        obs = env.reset(case_info, None, max_turns=int(config["experiment"]["max_turns"]))
        while not obs.done:
            kw = {"verification_template": obs.verification} if policy.needs_verification else {}
            po = policy.select_action(obs.case_info, obs.dialogue_history, obs.current_user_utterance, **kw)
            env.step(po)
            obs = env.observation
        last_user = next((t for t in reversed(env._history.turns) if t.speaker == "user"), None)
        st = last_user.user_state.model_dump() if last_user and last_user.user_state else {}
        fj = env._finalize()
        fields_ok = all(k in st for k in ("belief", "belief_dist", "confidence", "decision"))
        dist = st.get("belief_dist") or {}
        dist_ok = abs(sum(dist.values()) - 1.0) < 1e-6 if dist else False
        finalize_ok = bool(fj) and fj.get("concluded_option") == st.get("belief")
        ok &= fields_ok and dist_ok and finalize_ok
        print(f"  case={case_info.case_id} belief={st.get('belief')} q={st.get('confidence'):.2f} "
              f"decision={st.get('decision')} termination={st.get('termination_reason')} "
              f"is_correct={fj.get('is_correct') if fj else None} "
              f"[fields:{fields_ok} dist:{dist_ok} finalize:{finalize_ok}]")
    print(f"\nPart C: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B", "C"])
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--config", default="configs/sft_v2_distill.yaml")
    ap.add_argument("--cases", type=int, default=2)
    args = ap.parse_args()
    if args.part == "A":
        passed = part_a()
    elif args.part == "B":
        passed = part_b(args.model)
    else:
        passed = part_c(args.config, args.cases, args.model)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
