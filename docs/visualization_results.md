# 4×4 Spin Glass — 시각화 결과

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

- HVA (depth=15): Fidelity **0.9975** at $t=0.5$
- Trotter dt=0.1 (depth=70): Fidelity **0.9998** at $t=0.5$
- Trotter dt=0.2 (depth=28): Fidelity **0.8503** at $t=0.5$

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
- **(b) 바닥상태**: $E = -3.93 \to -20.75$ (ED 바닥 -22.47 대비 gap = 1.72)

---

## 5. 바닥상태 비교

![Ground state comparison](../results/4x4/plots/05_ground_state.png)

> [!WARNING]
> 바닥상태 결과는 아직 수렴하지 못했습니다 (cutoff=1e-3으로 인한 truncation bias). 개선 방향:
> - cutoff ↓ (1e-4 이하) + layer 수 ↑ (3→5~8)
> - 시간진화 파라미터로 warm-start
> - 허수 시간진화 (Imaginary Time Evolution) 방법 적용

---

## 6. 회로 깊이 비교

![Circuit depth comparison](../results/4x4/plots/06_depth_comparison.png)

- $t=0.5$ 기준: Trotter dt=0.1 depth=70 vs HVA depth=15 → **4.7배 압축**
- $t=1.0$ 기준: Trotter dt=0.1 depth=140 vs HVA depth=30 → **4.7배 압축** (구성 반복)

---

## 7. 큐비트별 Loss 히트맵

![Loss heatmap](../results/4x4/plots/07_loss_heatmap.png)

- 격자 중앙부(q5, q6, q9, q10)의 loss가 가장 큼 — 더 많은 이웃과 상호작용하여 SPO가 복잡
- 코너(q0, q3, q12, q15)의 loss가 가장 작음 — 이웃이 적어 SPO가 단순

---

## 8. 종합 요약 (BP-PPS 논문 스타일)

![Summary](../results/4x4/plots/08_summary.png)
