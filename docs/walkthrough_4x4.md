# 4×4 Spin Glass QML — 시뮬레이션 결과 워크스루

> [!CAUTION]
> **2026-08-28 정정.** 이 문서가 기록한 실행은 파울리 전파 엔진에 게이트 순서
> 관례 버그가 있는 상태에서 이뤄졌습니다. `propagate_forward` 가 게이트 목록을
> 회로 순서대로 소비해, 실제로는 `U(θ)ᵀ` 를 전파하고 있었습니다.
>
> - **시간진화 결과는 유효합니다.** `H` 가 실대칭이라 `V = exp(-iHΔt)` 가
>   complex-symmetric 이고, 타겟 Trotter 시퀀스도 palindromic 이어서
>   `U_c^T ≈ V` 와 `U_c ≈ V` 가 같은 조건이 됩니다. 측정된 fidelity 0.9975 는
>   실제 값입니다.
> - **바닥상태 결과는 무효입니다.** 최적화가 배포될 상태 `U(θ)|0⟩` 가 아니라
>   `U(θ)†|0⟩` 에 대해 이뤄졌습니다. 아래 "바닥상태" 절과
>   `extended_visualization.md` 의 truncation-bias 설명은 원인 진단이 틀렸습니다.
>
> 수정 후 회귀 테스트(TEST 10·11·13)가 추가되었고, 바닥상태는 재실행이 필요합니다.

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

#### `scripts/run_pipeline.py` (당시 이름 `run_4x4_simulation.py`)
- Stage 1-4 전체 파이프라인 (타겟 생성 → ED 비교 → HVA 훈련 → 검증)

#### `scripts/plot_results.py` & `scripts/plot_extended.py`
- 논문급 시각화 생성 스크립트 (총 11개 그림)

> [!NOTE]
> 스크립트 이름과 인터페이스는 이후 설정 기반으로 재구성되었습니다.
> 현재 사용법은 [manual.md](manual.md) 를 보세요.

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
> **Fidelity 0.9975** 는 3-layer HVA (depth 16, 2Q 게이트 72개) 가 달성한 값입니다.
>
> 압축률은 **하드웨어에서 실행 가능한 회로** 와 비교해야 합니다. 같은 $t = 0.5$ 에서
> 2차 Trotter($dt = 0.1$, depth 71, 2Q 240개) 대비 **깊이 4.4배 / 2큐빗 게이트 3.3배**
> 감소입니다. 타겟 생성에 쓴 500-step $S_4$ 회로($dt = 0.001$)와 비교해
> "1,867배" 라고 말하면 안 됩니다 — 그것은 하드웨어 회로가 아니라 수치적으로
> 정확한 기준값이고, BP-PPS 논문도 같은 이유로 2차 Trotter 를 비교군으로 씁니다.
> 참고로 논문이 보고한 압축률은 40% 이므로, 4.4배는 그보다 좋은 결과입니다.

#### 바닥상태 — 개선 필요

| 지표 | 값 |
|:---|:---|
| GS fidelity | 0.00006 |
| BP-PPS energy | -20.75 |
| Statevector energy | -10.71 |
| ED ground energy | -22.47 |

> [!CAUTION]
> **이 수치들은 무효입니다.** BP-PPS 에너지(−20.75)와 statevector 에너지(−10.71)의
> 차이는 truncation 이 아니라 게이트 순서 버그 때문입니다. 실제로 −20.75 는
> `U(θ)†|0⟩` 의 에너지(−20.7527)와 일치하고, 같은 cutoff 에서 순수 절단 오차는
> 0.0015 수준에 불과합니다.
>
> 수정된 코드에서는 검증 단계가 `|BP-PPS − statevector|` 를 추적된 절단 오차
> 추정값과 직접 비교하므로 같은 상황이 다시 발생하면 즉시 드러납니다.
> 바닥상태는 Trotter warm start + Adam → L-BFGS-B 로 재실행이 필요합니다.

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
3. **Trotter 비교**: dt=0.1 (depth 71, 2Q 240) 대비 HVA (depth 16, 2Q 72) — 깊이 4.4배, 2큐빗 게이트 3.3배 압축
4. **바닥상태**: 무효 (게이트 순서 버그). 재실행 필요
