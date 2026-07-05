# Burden–Accuracy Tension: a POC-level derivation

**Date:** 2026-07-03 · **Companion:** `analysis/action_space_findings.md`, `docs/problem_formulation.md`
**Reproduce (numbers):** `python3 analysis/burden_accuracy_model.py` (fixed points, ceiling, price,
efficiency crossover, frontier sim) · `python3 analysis/persona_yrc.py` (persona re-estimation).
**Reproduce (figures):** rendered from an HTML/SVG page via headless Chrome, not matplotlib — this
env has no matplotlib install after the server migration (2026-07-03). See `analysis/burden_accuracy_model.py`
for the plain-Python numbers if re-plotting with a different toolchain.

목표: "Align when they want"류 논문이 alignment↔complementarity tradeoff를 수식으로
정당화하듯, **우리 실측 (yield, risk, cost) 삼중값만으로 cognitive burden↔accuracy(complementarity)
사이의 긴장이 자유점심이 아니라 구조적으로 강제된다**는 것을 POC 수준에서 증명한다.

핵심 주장은 두 개다: **(C1)** accuracy에는 1보다 작은 상한 $\bar p$가 있고 그 값은 측정된 yield/risk
비로 결정된다; **(C2)** 그 상한에 다가갈수록 정확도 1점을 사는 burden 가격이 발산한다. 둘 다
아래 마르코프 모델에서 정리로 나오고, 실측값으로 캘리브레이션된다.

**한눈에 (C2):** PROBE만 계속 써서 $p_0=0.05$부터 올라갈 때, 정확도 1점(pp)을 사는 데 드는 burden이
구간마다 이렇다:

![burden price escalation](../plot/result/burden_price_escalation.png)

같은 "정확도 1점"인데 90→92% 구간이 5→30% 구간보다 **63배 더 비싸다** — 이게 이 문서 전체가
하는 얘기의 요지다. 아래는 이 숫자가 어디서 나오는지의 유도.

---

## 1. 모델 M(y, r, c) — 턴당 belief-flip 마르코프

임상의의 MCQ belief를 이진 상태 $b_t\in\{\text{correct},\text{wrong}\}$로 두고
$p_t=\Pr(b_t=\text{correct})$라 하자. accuracy $\equiv p_{\text{term}}$, complementarity $\equiv$
임상의 단독 수준 $p_0$ 위로 $p$를 끌어올린 양.

정책은 매 턴 행동 $a\in\{P,H,V\}$(PROBE / CHALLENGE / CONVERGE)를 고르고, 실측된 세 값이 곧
전이확률·비용이다:

- **yield** $y_a=\Pr(\text{wrong}\!\to\!\text{correct}\mid a)$
- **risk** $r_a=\Pr(\text{correct}\!\to\!\text{wrong}\mid a)$ (regression)
- **cost** $c_a=$ 다음 임상의 턴의 NASA-TLX burden 기대증가

전이:
$$p_{t+1}=p_t(1-r_a)+(1-p_t)\,y_a=p_t+\underbrace{(1-p_t)y_a-p_t r_a}_{\Delta p_a(p_t)}.$$

**측정 캘리브레이션 (controlled audit — `action_space_findings.md`, deepseek-v3.2 /
veteran_attending / 300 ep, deliberation_llm 정책, W3 yield):**

| a | $y_a$ | $r_a$ | $c_a$ |
|---|---|---|---|
| PROBE | 0.157 | 0.013 | 1.60 |
| CHALLENGE | 0.279 | 0.109 | 2.78 |
| CONVERGE | 0.064 | 0.024 | 1.25 |

### 가정 (POC라서 명시)
1. belief 이진화 (실제 belief/confidence는 연속).
2. $(y_a,r_a)$가 대화 history·turn index에 무관(homogeneous). 실제론 history 의존.
3. W3 yield를 "1회 적용 효과"로 사용 — delay(F2)를 순간효과로 근사.
4. 인구 동질(single case-population). §4에서 persona로 relax.

이 가정들은 절대수치를 낙관/왜곡할 수 있으나, 아래 정리의 **정성적 구조(상한의 존재, 가격의 발산,
내부 최적)** 는 $y_a,r_a\in(0,1),\ y_a+r_a<1$ 이기만 하면 성립한다.

---

## 2. 정리

### Prop 1 (행동별 fixed point → accuracy 상한 $\bar p<1$)
행동 $a$를 반복 적용하면 $p_t$는 유일 고정점
$$p^*_a=\frac{y_a}{y_a+r_a}$$
로 **기하적으로 수렴**한다. 왜냐하면
$p_{t+1}-p^*_a=(1-y_a-r_a)(p_t-p^*_a)$ 이고 수축계수 $1-y_a-r_a\in(0,1)$.
어떤 행동도 $p$를 자기 고정점 위로 못 올리므로, 도달 가능한 accuracy의 상한은
$$\boxed{\;\bar p=\max_a p^*_a\;}$$
측정값: $p^*_P=0.924,\ p^*_H=0.719,\ p^*_V=0.727 \Rightarrow \bar p=0.924<1.$

> **함의:** burden을 무한히 써도 accuracy는 1에 못 간다. 상한 0.924는 저위험 PROBE가 세운다
> (CHALLENGE가 아님 — 고yield지만 risk가 커서 고정점이 오히려 낮다). 이것이 **(C1)**.

![Δp(p) by action](../plot/result/delta_p_vs_p.png)

### Prop 2 (상한 근처에서 burden 가격 발산)
$\Delta p_a(p)$는 $p$에 단조감소하고 $p^*_a$에서 0이 된다. 정확도 1점을 사는 한계 burden 가격
$$\pi_a(p)=\frac{c_a}{\Delta p_a(p)}=\frac{c_a}{(1-p)y_a-p\,r_a}\;\xrightarrow[p\to p^*_a]{}\;\infty.$$
CHALLENGE 기준: $\pi_H(0)=9.96,\ \pi_H(0.5)=32.7,\ \pi_H(0.70)=376,\ \pi_H(0.715)=1759.$
등가로, $p^*_a-\varepsilon$까지 좁히는 데 드는 총 burden은
$\sim c_a\log(1/\varepsilon)/(y_a+r_a)\to\infty$.

> **함의:** complementarity의 마지막 몇 점은 사실상 무한 burden을 요구한다. 이것이 **(C2)**.
> Pareto frontier $F(B)=\max\{p_{\text{term}}:\text{burden}\le B\}$는 증가·오목·$\bar p$에서 포화,
> $dF/dB\to0$. 이 곡선이 우리 버전의 tradeoff 곡선이다.

![burden price by action](../plot/result/burden_price_vs_p.png)

### Prop 3 (지배 행동 없음 → 내부 최적, 긴장이 진짜다)
burden당 정확도 효율 $\rho_a(p)=\Delta p_a(p)/c_a$. $p=0$에서
$\rho_P=0.0981,\ \rho_H=0.1004,\ \rho_V=0.0512$ — **PROBE와 CHALLENGE가 거의 동률**인데 risk는
8배 차이($r_H/r_P=8.4$). $p$가 오르면 $r_H$가 커서 $\Delta p_H$가 더 빨리 꺼져 PROBE가 역전한다.
정확한 교차점(이분법으로 계산): $\rho_H(p)=\rho_P(p)$ at $\mathbf{p_{\text{cross}}=0.0671}$.
따라서 최적 행동이 $p$에 따라 바뀌는 **임계규칙(threshold rule)** 이고, 코너해가 아니다 — 다만
교차점이 $p^*_H=0.719$가 아니라 **0.067**로 아주 낮다는 점이 처음 스케치(채팅)에서 정정된 부분.

> **정정 (채팅 스케치 대비):** 초안에서는 "CHALLENGE로 빠르게 올리다 $p^*_H(0.72)$ 근처에서
> PROBE로 전환"이라고 썼는데, 이는 틀렸다. $\rho$ 교차점은 $p{=}0.067$이라 **CHALLENGE가 효율상
> 우위인 구간은 극히 좁다.** 게다가 이산 턴 시뮬레이션(`analysis`/그림 참고, $p_0{=}0.05$)에서 실제로
> 확인해보면, CHALLENGE 한 턴이 그 좁은 구간을 통째로 오버슈팅한다(한 턴 만에 $p{:}0.05\to0.31$).
> 그 결과 "최적 스위치" 전략은 순수 PROBE 대비 burden을 겨우 **0.42 단위**(총 64 중) 아끼는 데
> 그친다 — 방향은 맞지만(스위치가 PROBE보다 나쁜 적은 없음) 크기는 미미하다. 반면 CHALLENGE만
> 계속 쓰면 $p^*_H=0.719$에서 캡되어 PROBE-only(0.924 근처)에 크게 못 미친다.
>
> **함의(운영, 수정):** burden-efficiency 관점에서는 거의 항상 PROBE가 낫다 — CHALLENGE의 존재
> 이유는 "burden당 효율"이 아니라 다른 곳에 있어야 한다: (a) **턴 수**(wall-clock/횟수) 제약 하에서
> 빠르게 큰 점프가 필요할 때, (b) 처음부터 매우 안 움직이는($p_0$ 극히 낮은) 경우의 좁은 초기 창.
> `problem_formulation.md §6`의 "학습된 switching rule"은 여전히 성립하지만, 그 근거는
> burden-효율의 내부 최적이 아니라 **턴-예산 하의 최적**으로 다시 세워야 한다 — Prop 5에서 세운다.

### Prop 4 (complementarity ≡ risk 결합)
complementarity(=wrong→correct 이동)의 유일한 고yield 수단은 CHALLENGE인데, regression risk도
거기 집중돼 있다($r_H=0.109$, 타행동의 ~8배). yield/risk 비: PROBE 12.1, CHALLENGE 2.6,
CONVERGE 2.7 — CHALLENGE는 **최고 yield이면서 yield/risk는 최악**. 즉 "빠른 보완을 사는 행동이
곧 최대 퇴행을 주입하는 행동"이고, tradeoff가 하나의 action cluster에 localize된다
(`findings.md` F3의 정식화).

### Prop 5 (턴 예산 하에서 planning 복원)
Prop 3의 정정("burden-효율만 보면 거의 항상 PROBE")은 **턴 무제한 + burden만 과금하는 극한**의
결과다 — 그 극한에서는 시간이 공짜라 싼 행동(PROBE)을 오래 반복하는 것으로 burden을 대체할 수
있고, planning이 퇴화한다. 그러나 실제 환경은 턴이 유한하다(max_turns=8, agreement 종료, 임상의의
인내). 턴 예산 $T$를 고정하고 CHALLENGE 투입량 $m$과 위치를 전수 계산하면
($T{=}8,\ p_0{=}0.05$; `burden_accuracy_model.py`):

| plan | accuracy | burden |
|---|---|---|
| PROBE×8 | 0.727 | 12.8 |
| **CHALLENGE×3 → PROBE×5 (최적)** | **0.783** | 16.3 |
| CHALLENGE×8 | 0.706 | 22.2 |

턴 예산 하에서 세 가지가 복원된다:
1. **내부 최적 존재** — 순수 전략 둘 다 지고 $m{=}3$이 최적(+5.6pp vs PROBE-only). "언제,
   몇 번 CHALLENGE"가 퇴화하지 않는 진짜 결정이 된다.
2. **순서 의존성** — 같은 $\{H{\times}3, P{\times}5\}$라도 H를 앞에 쓰면 0.783, 뒤에 쓰면 0.687
   (**순서 하나로 ~10pp**). $p_{\text{cross}}{=}0.067$과 일관: CHALLENGE의 가치는 낮은 $p$를
   빠르게 탈출시키는 데 있고, 늦게 쓰면 자기 고정점($p^*_H{=}0.719$)이 오히려 발목을 잡는다.
3. **tradeoff 복원** — 최적 mix는 PROBE-only보다 burden을 +3.5 더 쓴다. 즉 **턴이 부족할 때
   burden을 지불해서 accuracy를 사는** 구조 — "interactive test-time scaling under a cognitive
   budget"의 정확한 수학적 형태.

> **함의:** CHALLENGE의 존재 이유는 burden-효율이 아니라 **턴-효율**이다. 정책이 학습해야 하는
> 것은 (i) 남은 턴·burden 예산과 (ii) 추정된 $p$(직접 관측 불가 — $D_t{+}v_t$에서 추론해야 하는
> POMDP)를 놓고 CHALLENGE의 투입량·타이밍을 정하는 일이며, max_turns가 짧을수록·persona의 $y$가
> 낮을수록 최적 CHALLENGE 비중이 커진다. **planning은 죽지 않았고, burden-only 극한에서만
> 퇴화한다.** `problem_formulation.md §6`의 switching rule은 이 턴-예산 근거 위에서 성립한다.

### Prop 6 (유효 horizon은 내생적 — `max_turns`가 아니라 임상의의 인내가 희소자원)
Prop 5는 "턴 예산이 planning을 되살린다"고 했는데, 그 예산이 `max_turns`(config 값)이 아니라
**임상의가 스스로 대화를 접는 시점**이라는 게 rollout 468개에서 그대로 나온다.

**종료 원인 분포:** agreement 326 (70%) · max_turns 139 (30%) · burden_dropout 3.
**persona별 실제 길이** (medical 턴 수, max_turns=8):

| persona | 평균 턴 | ≤2턴 | =8턴(cap) |
|---|---|---|---|
| exhausted_attending | 1.5 | 91% | 4% |
| veteran_attending | 2.1 | 82% | 12% |
| burned_out_resident | 4.7 | 24% | 30% |
| eager_resident | 6.2 | 7% | 58% |

**함의:** 70%가 `max_turns`에 닿기도 전에 임상의 쪽에서 끝낸다. `max_turns`를 늘려도 이 70%는
안 바뀐다 — 이미 그 전에 끝났으니까. 게다가 horizon이 정책의 burden 소비에 **내생적으로 묶여**
있다(CHALLENGE를 많이 쓰면 burden이 쌓여 dropout/조기 이탈이 당겨짐 — 정책이 자기 자신의
horizon을 소비한다). `max_turns`라는 config 노브로는 못 푸는 survival/pacing 문제이므로,
Prop 5의 "턴 예산"은 설정값이 아니라 **환경이 부과하는 구조적 제약**으로 다시 읽어야 한다.

**단, 이 종료 메커니즘 자체가 지금 하나의 계산상 우연에 크게 좌우된다 — 아래 §4b 참고.**

---

## 3. 요약 그림 — 시뮬레이션된 Pareto frontier

이산 턴 시뮬레이션($p_0=0.05$, 캘리브레이션된 $(y,r,c)$ 그대로 사용)으로 $F(B)$를 직접 그린 것:

![accuracy vs cumulative burden by strategy](../plot/result/burden_accuracy_frontier.png)

CHALLENGE-only는 $p^*_H=0.719$에서 캡되고, PROBE-only와 (거의 동일한) switch 전략은 $\bar p=0.924$
근처까지 올라간다 — Prop 1(상한)과 Prop 2(포화 접근시 가격 발산)가 시뮬레이션에서 그대로 재현됨.
switch가 PROBE-only를 능가하는 폭이 작다는 것(Prop 3 정정 참고)도 그림에서 두 곡선이 거의 겹치는
것으로 보임 — 이건 버그가 아니라 정직한 결과.

---

## 4. 예측 검증 — ceiling $\bar p$는 persona마다 이동한다

Prop 1은 $(y_a,r_a)$가 바뀌면 $\bar p$가 바뀐다고 예측한다. persona는 정확히 $(y_a,r_a)$를
바꾸는 축(confidence=belief 이동성, burden_sensitivity)이므로, **persona별 $\bar p$가 갈려야
하고, 잘 안 움직이는 persona일수록 낮아야 한다.**

GRPO rollout(phase1+phase2, 4-persona, ours_v2)에서 재추정 (`analysis/persona_yrc.py`):

| persona (2×2) | $\bar p$ | 세우는 lever | 비고 |
|---|---|---|---|
| burned_out_resident (low-conf) | **0.954** | PROBE (y=48.6%) | 잘 움직임 → 높은 상한 |
| eager_resident (low-conf) | **0.843** | PROBE (y=12.6%) | |
| veteran_attending (high-conf, robust) | **0.743** | PROBE (y=6.4%) | **최저 — 오늘 GRPO 0/6 collapse와 일치** |
| exhausted_attending (high-conf) | (1.000) | — | small-n 아티팩트: PROBE r=0/n, CH·CV n=12·14 → 무시 |

![accuracy ceiling by persona](../plot/result/persona_ceiling_bar.png)

**결과:** confidence 축을 따라 상한이 정렬된다 — low-confidence(움직이는) persona가 높은 $\bar p$,
high-confidence veteran이 가장 낮은 $\bar p=0.743$. 이는 오늘 관측된 "veteran_attending에서 정책이
1턴에 찍고 0/6로 무너짐"을 **상한 자체가 낮다**로 설명하고, 정책 실패가 아니라 환경의 구조적
한계일 수 있음을 시사한다.

**caveat:** 이 재추정은 학습 중인 ours_v2 정책의 rollout이라 절대 yield가 audit(§1)과 다르고,
일부 셀은 small-n(veteran CONVERGE n=16, exhausted 전반)이라 노이즈가 크다. 여기서 신뢰할 만한 건
**persona 간 상대 순서**와 **veteran이 최저**라는 정성적 방향이지 소수점이 아니다.

## 4b. 이 caveat이 왜 심각한가 — c0 vs q_close, 그리고 재보정 논의

Prop 6의 짧은 horizon(특히 veteran/exhausted 1.5~2.1턴)이 "설득이 빨리 끝나서"가 아니라
**초기화값이 종료선에 우연히 가까워서**라는 게 코드에서 그대로 확인된다.

**메커니즘 (`core/belief.py`, `source/persona/bayes_params.yaml`):**
- 종료 조건: $q_t=\max_k b_t(k)\ge q_{\text{close}}=0.90$ (전 persona 공통, config 단일값)
- 초기화: $b_0(k_0)=c_0$ — turn-0 확신도가 곧 시작 belief 질량
- $c_0$: veteran/exhausted **0.85**, eager/burned_out **0.55**

| persona | $c_0$ | $q_{\text{close}}$까지 거리 | 실측 턴당 $\Delta q$ | 예측 턴수 | 실측 평균 턴 |
|---|---|---|---|---|---|
| exhausted_attending | 0.85 | 0.05 | +0.053 | ~1.0 | 1.5 |
| veteran_attending | 0.85 | 0.05 | +0.025 | ~2.0 | 2.1 |
| burned_out_resident | 0.55 | 0.35 | +0.066 | ~5.3 | 4.7 |
| eager_resident | 0.55 | 0.35 | +0.036 | ~9.7 (cap에 걸림) | 6.2 |

(턴당 $\Delta q$는 실제 rollout의 `user_state.confidence` 연속값에서 직접 측정 — turn-1 $q$는
정확히 $c_0$와 일치해 첫 갱신이 turn 2부터 시작함을 확인.) "거리 ÷ 턴당 증가율"만으로 관측된
평균 턴수가 거의 그대로 재현된다 — 즉 **persona 순서를 만드는 주된 변수는 $\lambda,w$(설득
저항성, "confidence 축"이 원래 인코딩하려던 것)가 아니라 $c_0$의 시작 위치**다.

> **재보정하지 않기로 결정 (2026-07-03).** `bayes_params.yaml`은 v4를 쓰는 모든 실험이 공유하는
> 파일이라, 바꾸면 오늘 이전의 모든 persona 관련 결과와 비교가 깨진다 — 지금 이 재보정을 정당화할
> 만큼 급하지 않다는 판단. 대신 **해석(narrative)을 정정한다:**
>
> "confidence 축"은 원래 "얼마나 설득에 저항하는가"($\lambda,w$)를 의도했지만, 실측상 그 축을
> **지배하는 건 $\lambda,w$가 아니라 $c_0$의 시작 위치**다(위 표: 거리÷턴당증가율만으로 관측
> 평균턴수가 재현됨). 즉 이 4-persona 축이 실제로 인코딩하는 것은 **"얼마나 완고한가"가 아니라
> "대화가 얼마나 오래 가는가(=observation horizon)"** 이고, veteran/exhausted 같은 이름이 주는
> "완고한 시니어" 인상은 표면적 롤플레이 텍스트(persona_*.yaml)가 주는 것이지 실제 belief
> dynamics가 주는 게 아니다. 페르소나 이름은 그 자체로 실재하는 사실이 아니라 4개 $(c_0,\lambda,
> w,\rho,\kappa)$ 조합에 붙인 라벨일 뿐이므로, 라벨과 다이나믹스가 어긋나도 시뮬레이터가 틀린
> 것은 아니다 — 다만 **논문/문서에서 "veteran이 완고해서 안 움직인다"라고 쓰면 안 되고, "veteran
> 조합이 관측 horizon을 짧게 만든다"라고 써야 한다.**
>
> 남는 실질적 영향은 하나뿐: §4의 veteran $\bar p=0.743$과 exhausted의 small-n(CH n=12, CV n=14)은
> "genuine하게 안 움직이는 persona"가 아니라 **"관측 기회가 애초에 없어서"** 나온 결과일 수 있다는
> 것 — 이 caveat은 재보정 여부와 무관하게 §4 숫자를 인용할 때마다 계속 따라다녀야 한다.
> 재보정 옵션 자체($c_0$↓ 또는 persona별 $q_{\text{close}}$)는 필요해지면 다시 꺼낼 수 있도록
> 위 표에 남겨둔다.

---

## 5. 한계와 강화 경로 (다음 단계)

1. **고정 정책으로 재측정.** §4는 evolving 정책이라 오염. SFT-고정(ours_v2 final) 또는
   deliberation_llm로 4-persona × N케이스 eval 돌려 $(y_a,r_a,c_a)$를 persona별로 깨끗이 추정하면
   $\bar p$ 순서가 안정적인지 확인 가능. (비용 있는 실행 — 지시 시 config 준비)
2. **연속 belief로 확장.** 이진 대신 confidence를 상태로 두면 $\Delta p$가 $2\times2$ 전이핵이
   되고 fixed point는 정상분포의 correct-mass가 된다. 상한·발산 구조는 보존될 것으로 예상.
3. **history 의존성.** $(y_a,r_a)$를 turn index/누적 burden의 함수로 추정(예: burden_dropout 이후
   yield 급감). dropout hazard를 넣으면 $F(B)$가 포화가 아니라 **감소**로 꺾일 수 있음(과다 대화가
   오히려 해로움) — 더 강한 tradeoff 주장.
4. **GRPO와의 연결.** Prop 3의 임계규칙이 학습된 정책의 실제 행동선택과 일치하는지
   (rollout에서 $p$ 추정치 대비 control 선택 분포) 검증하면 이론↔학습을 닫는다.
