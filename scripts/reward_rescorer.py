"""Tier-0 오프라인 reward 재채점기 (재-rollout 없이 reward 설계를 거른다).

핵심 원리
---------
GRPO 한 스텝의 비용은 rollout(LLM 콜)에 있고, reward 계산 자체는 궤적 feature에
가중치를 곱하는 순수 함수라 공짜다. 그래서 "이미 굴러간 고정 궤적"에 여러 reward
가중치를 오프라인으로 씌워, GPU를 한 번도 안 쓰고 다음을 본다:

  1) 랭킹 보존   : R(oracle) > R(deliberation_llm) > R(naive) 인가? (좋은 정책이 높은 R)
  2) 턴 수렴점   : 이 reward가 원하는 종료 지점 t* = argmax_t meanR(stop@t) 이 어디인가?
                   λ_turn/λ_burden 을 올리면 t* 가 효율점으로 당겨지는가? (재-rollout 없이 실증)
  3) net-recovery: R 이 우리가 실제로 원하는 것(recovery − corruption)과 정합하는가?

입력 = poc_0630_* / scaling_poc_* 계열의 results.json (단일 궤적/케이스 + 턴별 checkpoint).
  checkpoints[t] = {is_correct, cumulative_burden, [cumulative_r_align], self_reported_belief, ...}
  t=0 은 의사 단독(AI 개입 전). record.alone_correct == checkpoints['0'].is_correct.
  * leak/fmt 신호는 이 로그에 없어 0 취급(주석 참고). 실제 GRPO train_log 재채점은 --train-log 모드.

reward 식 (trajectory 가 t 턴에서 종료했다고 가정):
  R(stop@t) = w_final · is_correct(t)
            − w_turn   · t
            − w_burden · cumulative_burden(t)
            + w_align  · cumulative_r_align(t)          # 로그에 있을 때만
            + w_belief · (Phi(t) − Phi(0))              # belief-potential shaping (cell 가중과 동일)
  Phi(t) = 1[self_reported_belief(t)==정답] 이 있으면 그것, 없으면 is_correct(t) (이진 근사)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass


# ── reward 가중치 후보 (전 옵션 주석 포함) ────────────────────────────────────
# 각 필드는 R 식의 계수. 없는 신호(leak/fmt)는 poc_0630 로그에 없어 여기서 다루지 않음.
DEFAULT_WEIGHTS = {
    "w_final": 1.0,    # 최종 정답(=인간 결정) 보상. 목표항. 보통 1.0 고정.
    "w_turn": 0.0,     # 턴당 비용. plateau 무차별을 깨서 턴 길이를 효율점으로 당김. 상승구간 이득(~0.03/턴)보다 작게.
    "w_burden": 0.0,   # 누적 NASA-TLX burden당 비용. 턴 차등비용(CHALLENGE>CONVERGE). 0.02면 nudge 수준.
    "w_align": 0.0,    # r_align 프로세스 shaping. 궤적철학상 RL에선 0 권장(SFT teacher로 강등). 로그에 있을 때만 적용.
    "w_belief": 0.0,   # belief-potential (Phi_end−Phi_0). recovered +, regressed −. medcobe식 무분별 설득을 거름.
}

# 이름붙인 후보 조합 (reward 후보 정리와 1:1 대응)
NAMED_CANDIDATES = {
    "C0_acc_only":      {"w_final": 1.0},
    "C1_acc_turn":      {"w_final": 1.0, "w_turn": 0.02},
    "C2_acc_burden":    {"w_final": 1.0, "w_burden": 0.02},
    "C4_acc_belief":    {"w_final": 1.0, "w_belief": 0.5},
    "C1p4_turn_belief": {"w_final": 1.0, "w_turn": 0.02, "w_belief": 0.5},
}


@dataclass
class Traj:
    """고정 궤적 하나(한 케이스, 한 페르소나). 턴별 배열은 t=0..T (0=단독)."""
    policy: str
    case_id: str
    alone_correct: bool
    is_correct: list[bool]         # [t]
    cum_burden: list[float]        # [t]
    cum_align: list[float] | None  # [t] or None (로그에 없으면)
    phi: list[float]               # [t]  Phi(t)
    natural_end_turn: int | None
    max_t: int


def _phi_series(ck: dict, correct_option: str | None, is_correct: list[bool]) -> list[float]:
    """Phi(t): self_reported_belief 가 정답과 일치하면 1. 정답 라벨/belief 없으면 is_correct 로 이진 근사."""
    ts = sorted(ck, key=int)
    if correct_option:
        out = []
        for t in ts:
            b = ck[t].get("self_reported_belief")
            out.append(1.0 if (b is not None and str(b).strip().upper() == correct_option.strip().upper()) else 0.0)
        return out
    return [1.0 if c else 0.0 for c in is_correct]


def load_trajs(policy_dirs: list[str]) -> list[Traj]:
    trajs: list[Traj] = []
    for pdir in policy_dirs:
        policy = os.path.basename(pdir.rstrip("/"))
        for fp in sorted(glob.glob(os.path.join(pdir, "*", "results.json"))):
            data = json.load(open(fp))
            # 정답 라벨은 results.json 에 없을 수 있음(있으면 belief-정합 Phi, 없으면 is_correct 근사)
            for rec in data.get("records", []):
                ck = rec["checkpoints"]
                ts = sorted(ck, key=int)
                is_corr = [bool(ck[t]["is_correct"]) for t in ts]
                cum_b = [float(ck[t].get("cumulative_burden", 0.0)) for t in ts]
                has_align = all("cumulative_r_align" in ck[t] for t in ts)
                cum_a = [float(ck[t]["cumulative_r_align"]) for t in ts] if has_align else None
                phi = _phi_series(ck, rec.get("correct_option"), is_corr)
                trajs.append(Traj(
                    policy=policy, case_id=str(rec.get("case_id")),
                    alone_correct=bool(rec.get("alone_correct", is_corr[0])),
                    is_correct=is_corr, cum_burden=cum_b, cum_align=cum_a, phi=phi,
                    natural_end_turn=rec.get("natural_end_turn"),
                    max_t=int(ts[-1]),
                ))
    return trajs


def R_at(tr: Traj, t: int, w: dict) -> float:
    """궤적 tr 이 t 턴에서 종료했을 때의 R."""
    r = w.get("w_final", 0.0) * (1.0 if tr.is_correct[t] else 0.0)
    r -= w.get("w_turn", 0.0) * t
    r -= w.get("w_burden", 0.0) * tr.cum_burden[t]
    if tr.cum_align is not None:
        r += w.get("w_align", 0.0) * tr.cum_align[t]
    r += w.get("w_belief", 0.0) * (tr.phi[t] - tr.phi[0])
    return r


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def analyze(trajs: list[Traj], w: dict, quality_order: list[str]) -> dict:
    """한 reward 가중치 w 에 대한 Tier-0 리포트."""
    by_pol: dict[str, list[Traj]] = {}
    for tr in trajs:
        by_pol.setdefault(tr.policy, []).append(tr)

    out: dict = {"per_policy": {}, "weights": w}
    for pol, trs in by_pol.items():
        maxT = min(tr.max_t for tr in trs)
        # (1) 종단 R (t=max, 즉 강제 full-turn 종료 지점)
        term_R = _mean(R_at(tr, tr.max_t, w) for tr in trs)
        # (2) 턴 수렴점 t*: 모든 궤적을 같은 t 에서 종료시켰을 때 평균 R 최대인 t
        curve = {t: _mean(R_at(tr, t, w) for tr in trs) for t in range(maxT + 1)}
        t_star = max(curve, key=lambda t: curve[t])
        acc_at_tstar = _mean(1.0 if tr.is_correct[t_star] else 0.0 for tr in trs)
        burden_at_tstar = _mean(tr.cum_burden[t_star] for tr in trs)
        # (3) net-recovery @ t* : recovery − corruption
        wrong = [tr for tr in trs if not tr.alone_correct]
        right = [tr for tr in trs if tr.alone_correct]
        rec = _mean(1.0 if tr.is_correct[t_star] else 0.0 for tr in wrong) if wrong else float("nan")
        cor = _mean(0.0 if tr.is_correct[t_star] else 1.0 for tr in right) if right else float("nan")
        # 자연 종료점(에이전트가 실제로 멈췄을 지점)에서의 R/acc
        nat = [tr for tr in trs if tr.natural_end_turn is not None]
        nat_R = _mean(R_at(tr, min(int(tr.natural_end_turn), tr.max_t), w) for tr in nat) if nat else float("nan")
        nat_t = _mean(tr.natural_end_turn for tr in nat) if nat else float("nan")
        out["per_policy"][pol] = {
            "n": len(trs), "term_R": term_R, "t_star": t_star, "R_at_tstar": curve[t_star],
            "acc_at_tstar": acc_at_tstar, "burden_at_tstar": burden_at_tstar,
            "recovery": rec, "corruption": cor, "net": rec - cor if wrong and right else float("nan"),
            "nat_end_turn": nat_t, "nat_end_R": nat_R,
        }
    # 랭킹 보존 체크 (quality_order = 낮은 품질 → 높은 품질)
    present = [p for p in quality_order if p in out["per_policy"]]
    term = [out["per_policy"][p]["term_R"] for p in present]
    out["ranking_preserved"] = all(term[i] <= term[i + 1] for i in range(len(term) - 1)) if len(term) > 1 else None
    out["ranking_order"] = present
    return out


def print_report(name: str, rep: dict):
    w = {k: v for k, v in rep["weights"].items()}
    print(f"\n{'='*100}\n[{name}]  weights={w}")
    hdr = f"{'policy':<38} {'n':>4} {'termR':>7} {'t*':>3} {'R@t*':>7} {'acc@t*':>7} {'burd@t*':>8} {'recov':>6} {'corr':>6} {'net':>6} {'natT':>5}"
    print(hdr)
    for pol, m in sorted(rep["per_policy"].items(), key=lambda kv: kv[1]["term_R"]):
        print(f"{pol:<38} {m['n']:>4} {m['term_R']:>7.3f} {m['t_star']:>3} {m['R_at_tstar']:>7.3f} "
              f"{m['acc_at_tstar']:>7.2f} {m['burden_at_tstar']:>8.2f} {m['recovery']:>6.2f} {m['corruption']:>6.2f} "
              f"{m['net']:>6.2f} {m['nat_end_turn']:>5.1f}")
    rp = rep["ranking_preserved"]
    print(f"랭킹 보존({' < '.join(rep['ranking_order'])}): {rp}")


def main():
    ap = argparse.ArgumentParser(description="Tier-0 오프라인 reward 재채점기")
    ap.add_argument("--outputs", default="outputs", help="results.json 들이 있는 outputs 루트")
    ap.add_argument("--policies", nargs="+", required=True,
                    help="정책 run 디렉터리명 (품질 오름차순으로: 낮은 품질 먼저). 예: poc_0630_naive poc_0630_deliberation_llm poc_0630_deliberation_llm_oracle")
    ap.add_argument("--candidates", nargs="+", default=list(NAMED_CANDIDATES),
                    help=f"평가할 후보. 기본 전부: {list(NAMED_CANDIDATES)}")
    ap.add_argument("--turn-sweep", nargs="*", type=float, default=None,
                    help="주면 w_turn 스윕(예: 0 0.01 0.02 0.05 0.1)해서 t* 이동을 표로")
    args = ap.parse_args()

    pdirs = [os.path.join(args.outputs, p) for p in args.policies]
    trajs = load_trajs(pdirs)
    print(f"로드된 궤적: {len(trajs)}개  (정책 {len(args.policies)}개)")
    quality_order = args.policies  # 품질 오름차순 가정

    for name in args.candidates:
        rep = analyze(trajs, {**DEFAULT_WEIGHTS, **NAMED_CANDIDATES[name]}, quality_order)
        print_report(name, rep)

    if args.turn_sweep is not None:
        sweep = args.turn_sweep or [0.0, 0.01, 0.02, 0.05, 0.1]
        print(f"\n{'='*100}\n[w_turn 스윕]  각 정책의 t*(수렴 종료점) 이동 — plateau를 깨서 효율점으로 당겨지는지")
        by_pol_present = sorted({tr.policy for tr in trajs})
        print(f"{'w_turn':>8} " + " ".join(f"{p[:24]:>26}" for p in by_pol_present))
        for wt in sweep:
            rep = analyze(trajs, {**DEFAULT_WEIGHTS, "w_final": 1.0, "w_turn": wt}, quality_order)
            cells = []
            for p in by_pol_present:
                m = rep["per_policy"].get(p)
                cells.append(f"t*={m['t_star']}(acc{m['acc_at_tstar']:.2f})" if m else "-")
            print(f"{wt:>8.3f} " + " ".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    main()
