# Tier 1 Protocol — Order Reality Check (Contrast B first)

> **Go/no-go for the whole thesis.** 시뮬레이터·RL 짓기 전에 이것부터.
> "같은 act 다중집합, 순서만 다르면 실인간의 appropriate reliance가 유의하게 갈리는가?"

## 0. 결정된 사항
- **과제 도메인:** 의료 vignette (Tier 4 임상 헤드라인으로 직접 전이).
- **설계:** mixed, order는 item-내 within (Latin-square 카운터밸런싱). [대안: between-subjects, N 2배]
- **풀:** 사용자가 직접 모집·운영.
- **먼저 돌릴 것:** Contrast B (recommend 타이밍) 단독. A는 B 통과 후.

## 1. 가설
- **H_B (precedence):** expand-first(B1)는 AI가 맞을 때 적절히 따라가고 AI가 틀릴 때 self-reliance를 보존한다. recommend-first(B2)는 anchoring → AI가 틀려도 수용(overreliance).
- **통계적 표현:** `order × ai_correct` 교호작용 유의.

## 2. 과제 (hard, 초기정확도 55–65% 목표)
- 2-가설 진단. 기준 vignette: 62세 남성, 3일 점진적 호흡곤란 + 양측 하지부종, HTN 기왕력, 경미한 수포음, BNP 경계역↑, 흉부X선 모호한 울혈, D-dimer 경도↑. **H1 CHF vs H2 PE.**
- **AI correctness 조작:** ground truth 고정(예: CTPA가 PE 확정 → AI가 PE 권고 = correct, CHF 권고 = incorrect).
- vignette 변형 K=8 (같은 임상 패턴, 표면만 변경) → anchoring 전이 차단 + 반복측정 파워.

## 3. 절차 (2-stage judge-advisor)
1. **초기 판단:** 선택(H1/H2) + confidence(0–100) + 근거 자유서술
2. **AI 대화 노출:** 배정된 순서 스크립트 (B1 또는 B2)
3. **최종 판단:** 선택 + confidence + NASA-TLX
- switch는 반드시 **AI correctness로 조건화** → RAIR/RSR 산출.

## 4. 스크립트 (Contrast B) — 발화 고정, 순서만 교체
구성 발화 (다중집합 고정):
- `COMPARE_HYPOTHESES` ①: "CHF와 PE 둘 다 설명 가능합니다. 부종·BNP는 CHF를, D-dimer·급성 경과는 PE를 시사합니다."
- `PROVIDE_REFUTING_EVIDENCE` ②: "다만 명확한 심비대나 뚜렷한 폐부종 소견은 약합니다."
- `SURFACE_ALTERNATIVE` ③: "양측성이라 PE를 배제하기 쉽지만, bilateral/saddle PE도 이 양상으로 옵니다."
- `MOVE_RECOMMEND + ASSERT_RECOMMEND_REASON` ④: "종합하면 CTPA로 PE를 먼저 배제할 것을 권합니다 — 급성 경과와 D-dimer 때문입니다."

| 조건 | 순서 |
|---|---|
| **B1 expand-first** | ① → ② → ③ → ④ |
| **B2 recommend-first** | ④ → ① → ② → ③ |

내용·문장·act 집합 동일, 순서만 변동.

### (나중) Contrast A 슬롯
- `ELICIT_RATIONALE` ("현재 어느 쪽이고 근거는?") vs `PROMPT_REVISION/PROVIDE_REFUTING_EVIDENCE` (challenge)
- A1 = elicit→challenge, A2 = challenge→elicit. ELICIT 응답 = reasoning engagement DV.

## 5. 측정 (사전등록; 주 DV는 accuracy 아님)
| 구성 | 프록시 |
|---|---|
| **appropriate reliance (주)** | RAIR/RSR (Schemmer) + switch×correctness |
| accuracy (부) | 최종 정답률 |
| cognitive load | NASA-TLX + 체류시간·클릭 |
| reasoning engagement | 산출 고려사항 수 |

## 6. 검정
- **Primary:** `glmer(switch ~ order * ai_correct + (1|subject) + (1|item))` — 교호작용이 precedence 검정.
- **Secondary:** 최종 accuracy / NASA-TLX / RAIR·RSR 각 혼합모델.
- **Go 게이트:** ≥1 DV 순서 유의 → 통과. 교호작용 유의 → 강한 통과(precedence 입증).
- **No-go:** 순서 효과 없음 → trajectory 주장 사망 또는 "adaptive support selection"으로 강등.

## 7. N
1. **파일럿 n≈20** → 분산성분·효과크기 추정.
2. `simr`로 시뮬레이션 파워 → 교호작용 80% 파워 N 확정.
- ballpark(d≈0.35, K=8 within): 주효과 ~50–70/arm, 교호작용 총 ~100–140. between이면 2배.

## 8. 다음 작업 큐
- [ ] vignette 8변형 작성 + 각 ground truth 고정
- [ ] B1/B2 스크립트 8세트 인스턴스화
- [ ] 사전등록 문서 (OSF) — 주 DV·검정·중지규칙
- [ ] Qualtrics/oTree 플로우 (2-stage + 랜덤배정)
- [ ] 파일럿 → simr 파워 → 본 N 확정
