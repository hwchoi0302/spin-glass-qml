# 4×4 Spin Glass — 시각화 결과

> [!CAUTION]
> **2026-08-28 정정.** 5번(바닥상태)과 4번(b) 패널은 게이트 순서 버그가 있는
> 실행에서 나온 것이라 무효입니다. 1·2·3·6·7 번은 유효합니다.
> 자세한 내용은 [extended_visualization.md](extended_visualization.md) 를 보세요.

## 1. 격자 구조 시각화 (J 커플링 + h 자기장)

![4×4 EA Bimodal Spin Glass lattice with J couplings and h field](../results/4x4/plots/01_lattice_J_h.png)

- **파란색 ($J=+1$)**: 강자성 결합 — 스핀이 같은 방향으로 정렬하려 함
- **빨간색 ($J=-1$)**: 반강자성 결합 — 스핀이 반대 방향으로 정렬하려 함
- **주황색 화살표**: 횡자기장 $h=1.0$ (모든 큐비트에 동일)
- $J$의 무작위 부호로 인해 **frustration** 발생 → 바닥상태가 복잡한 얽힘 구조를 가짐

---

## 2. 시간진화 Fidelity 비교 (ED vs Trotter vs HVA)

![Fidelity comparison](../results/4x4/plots/02_fidelity_comparison.png)

> [!IMPORTANT]
> **Fidelity 검증 방식**: HVA 회로에 훈련된 파라미터를 넣고, **Qiskit Statevector 시뮬레이터**로 16큐비트 상태벡터 $|\psi_{\rm HVA}\rangle$를 정확히 계산한 뒤, ED의 정확한 시간진화 $|\psi_{\rm exact}\rangle$와 비교했습니다. 이것은 **16큐비트($2^{16} = 65,536$ 차원)라서 고전적으로 가능**합니다. 100큐비트에서는 불가능합니다.

- HVA (depth 16, 2Q 72): Fidelity **0.9975** at $t = 0.5$
- Trotter $dt = 0.1$ (depth 71, 2Q 240): Fidelity **0.9998** at $t = 0.5$
- Trotter $dt = 0.2$ (depth 29, 2Q 96): Fidelity **0.8503** at $t = 0.5$

---

## 3. 관측량 비교 ($t=0.5$)

![Observable comparison](../results/4x4/plots/03_observable_comparison.png)

- 상단: $\langle X_i \rangle$, $\langle Z_i \rangle$ 값 비교 — ED, Trotter dt=0.1/0.2, HVA
- 하단: 각 방법의 ED 대비 오차 $|\Delta|$
- HVA와 Trotter dt=0.1 모두 ED와 거의 일치
- Trotter dt=0.2는 눈에 띄게 편차가 큼

---

## 4. 훈련 곡선

![Training curves](../results/4x4/plots/04_training_curves.png)

- **(a) 시간진화 압축**: Loss 35.1 → 0.049 (99.9% 감소, 200 epochs)
- **(b) 바닥상태**: 무효 — 게이트 순서 버그로 켤레 상태를 최적화한 결과

---

## 5. 바닥상태 비교

![Ground state comparison](../results/4x4/plots/05_ground_state.png)

> [!CAUTION]
> 이 그림은 무효입니다. 원인은 truncation bias 가 아니라 게이트 순서 버그였고,
> 최적화가 `U(θ)†|0⟩` 에 대해 이뤄졌습니다. 수정 후 Trotter warm start +
> Adam → L-BFGS-B 로 재실행이 필요합니다.

---

## 6. 회로 깊이 비교

![Circuit depth comparison](../results/4x4/plots/06_depth_comparison.png)

- $t = 0.5$ 기준: Trotter $dt = 0.1$ depth 71 / 2Q 240 vs HVA depth 16 / 2Q 72
  → 깊이 4.4배, 2큐빗 게이트 3.3배 압축
- $t = 1.0$ 기준: depth 141 vs 31 → 4.5배 (블록 반복)

---

## 7. 큐비트별 Loss 히트맵

![Loss heatmap](../results/4x4/plots/07_loss_heatmap.png)

- 격자 중앙부(q5, q6, q9, q10)의 loss가 가장 큼 — 더 많은 이웃과 상호작용하여 SPO가 복잡
- 코너(q0, q3, q12, q15)의 loss가 가장 작음 — 이웃이 적어 SPO가 단순

---

## 8. 종합 요약 (BP-PPS 논문 스타일)

![Summary](../results/4x4/plots/08_summary.png)
