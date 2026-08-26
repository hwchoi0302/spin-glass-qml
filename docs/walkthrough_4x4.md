# 4×4 Spin Glass QML — 시뮬레이션 결과 워크스루

## 개요

4×4 (16큐비트) 2D 스핀 유리 모델의 양자 시뮬레이션 파이프라인을 Python BP-PPS 엔진으로 전체 실행했습니다. 총 ~10.5시간 소요.

---

## 코드 변경 사항

### 4차 Suzuki-Trotter 구현

#### `src/bppps/propagation.py`
- `_append_s2_step()` 헬퍼 함수 추가 — $S_2(\tau)$ 한 스텝을 sequence에 append
- `build_trotter_gate_sequence()`에 **order=4** 지원 추가
- $S_4$ 공식: $S_4(dt) = S_2(p\cdot dt)^2 \cdot S_2((1-4p)\cdot dt) \cdot S_2(p\cdot dt)^2$, $p = 1/(4 - 4^{1/3})$

### 시뮬레이션 스크립트

#### `scripts/run_4x4_simulation.py`
- Stage 1-4 전체 파이프라인 (타겟 생성 → ED 비교 → HVA 훈련 → 검증)

#### `scripts/run_4x4_completion.py`
- 바닥상태 훈련(cutoff=1e-3) + 최종 검증 마무리 스크립트

#### `scripts/plot_4x4_results.py` & `scripts/plot_4x4_extended.py`
- 논문급 시각화 생성 스크립트 (총 11개 그림)

---

## 실행 결과 요약

### Stage 1: Target SPO 생성 ✅

| 항목 | 값 |
|:---|:---|
| Trotter 차수 | 4차 Suzuki-Trotter ($S_4$) |
| dt | 0.001 |
| $\Delta t$ | 0.5 |
| Trotter steps | 500 |
| Observable 수 | 32 ($X_0 \sim X_{15} + Z_0 \sim Z_{15}$) |
| 총 Pauli terms | **1,974,348** |
| 소요 시간 | 24,322초 (~6.8시간) |

> [!NOTE]
> $X$ observable은 8K~293K terms, $Z$ observable은 2.6K~65K terms로 큐비트 위치에 따라 큰 차이. 격자 중앙부($X_5, X_6, X_9, X_{10}$)가 ~290K terms로 가장 크고, 코너($X_0, X_3, X_{12}, X_{15}$)가 ~8K terms로 작음.

### Stage 2: 고전 비교 데이터 ✅

#### Exact Diagonalization
- **바닥 에너지**: $E_0 = -22.4722$

| t | E(t) | $\langle X_0 \rangle$ | $\langle Z_0 \rangle$ |
|:---|:---|:---|:---|
| 0.1 | -4.000 | -0.000001 | 0.980 |
| 0.5 | -4.000 | -0.004863 | 0.556 |
| 1.0 | -4.000 | 0.054485 | -0.051 |

#### Trotter 시뮬레이션 (Statevector)

| t | dt=0.1 fidelity | dt=0.1 depth | dt=0.2 fidelity | dt=0.2 depth |
|:---|:---|:---|:---|:---|
| 0.1 | 0.99998 | 14 | 0.85271 | 0 |
| 0.5 | 0.99984 | 70 | 0.85029 | 28 |
| 1.0 | 0.99978 | 140 | 0.99628 | 70 |

> [!IMPORTANT]
> dt=0.2에서 t=0.1 fidelity가 0.853인 이유: `round(0.1/0.2) = 0`으로 Trotter step이 0이 되어 identity 회로가 됩니다. dt=0.2는 $t \ge 0.2$인 시점에서만 유효합니다.

### Stage 3: HVA 훈련 ✅

#### 시간진화 압축 (핵심 결과)

| 항목 | 값 |
|:---|:---|
| HVA layers | 3 |
| 파라미터 수 | 120 |
| Epochs | 200 |
| LR | 0.01 (Adam) |
| Cutoff | 1e-4 |
| **초기 loss** | **35.10** |
| **최종 loss** | **0.049** |
| **감소율** | **99.9%** |
| 소요 시간 | 3,976초 (~66분) |

```
Epoch   1: loss=35.104  |grad|=10.24
Epoch  40: loss= 0.942  |grad|= 3.11   ← 급격한 수렴
Epoch 100: loss= 0.078  |grad|= 0.21
Epoch 200: loss= 0.049  |grad|= 0.09   ← 안정적 수렴
```

#### 바닥상태 준비

| 항목 | 값 |
|:---|:---|
| Cutoff | 1e-3 (속도를 위해 높임) |
| Epochs | 100 |
| LR | 0.05 |
| 초기 에너지 | -3.93 |
| 최종 에너지 (BP-PPS) | -20.75 |
| ED 바닥 에너지 | -22.47 |
| 에너지 갭 | 1.72 |
| 소요 시간 | 1,067초 (~18분) |

### Stage 4: 검증 결과 ✅

#### 시간진화 ($t=0.5$) — 우수한 결과

| 지표 | 값 |
|:---|:---|
| **State fidelity** | **0.9975** |
| Total $\mathcal{L}_{XZ}$ loss | 0.0491 |
| Mean $\|\langle X_i \rangle_{\rm HVA} - \langle X_i \rangle_{\rm ED}\|$ | 0.0104 |
| Mean $\|\langle Z_i \rangle_{\rm HVA} - \langle Z_i \rangle_{\rm ED}\|$ | 0.0112 |
| $\|\Delta E\|$ | 0.083 |

> [!TIP]
> **Fidelity 0.9975**는 3-layer HVA (depth=15)로 500-step $S_4$ Trotter (depth~28,000)를 압축한 것으로, **회로 깊이를 ~1,867배 줄이면서 99.75% fidelity**를 달성한 것입니다.

#### 바닥상태 — 개선 필요

| 지표 | 값 |
|:---|:---|
| GS fidelity | 0.00006 |
| BP-PPS energy | -20.75 |
| Statevector energy | -10.71 |
| ED ground energy | -22.47 |

> [!WARNING]
> 바닥상태 결과는 cutoff=1e-3의 truncation으로 인한 systematic bias가 있습니다. BP-PPS 에너지 추정(-20.75)과 실제 statevector 에너지(-10.71)의 큰 차이는 truncation이 에너지 landscape를 왜곡하기 때문입니다. 개선 방향:
> - cutoff를 1e-4 이하로 낮추고 더 많은 epoch 수행
> - layer 수 증가 (3 → 5~8)
> - 시간진화 파라미터를 초기값으로 warm-start

---

## 생성된 파일

| 파일 | 크기 | 설명 |
|:---|:---|:---|
| `results/4x4/model_config.json` | 1.2 KB | 모델 설정 (J, bonds, etc.) |
| `results/4x4/ed_results.json` | 10 KB | ED 정답 데이터 |
| `results/4x4/trotter_results.json` | 17 KB | Trotter 시뮬레이션 비교 데이터 |
| `results/4x4/trained_params.json` | 8 KB | 시간진화 훈련 파라미터 + loss |
| `results/4x4/gs_trained_params.json` | 5.5 KB | 바닥상태 훈련 파라미터 + loss |
| `results/4x4/validation_results.json` | 1.8 KB | 검증 결과 |
| `results/4x4/composition_fidelity.json` | 1.2 KB | 다중 시간스텝 composition fidelity |
| `results/4x4/gs_multi_layer.json` | 12 KB | Layer별 (1~5) 바닥상태 훈련 결과 |

---

## 핵심 결론

1. **시간진화 압축 성공**: 3-layer HVA (depth 15)로 $S_4$ Trotter (depth ~28K)를 fidelity 99.75%로 압축
2. **국소 관측량 정확도**: $\langle X_i \rangle$, $\langle Z_i \rangle$ 평균 오차 ~0.01 수준
3. **Trotter 비교**: dt=0.1 (depth 70)에 비해 HVA (depth 15)가 ~4.7배 더 얕은 회로
4. **바닥상태**: cutoff 제약으로 완전 수렴하지 못함 — 추가 최적화 필요
