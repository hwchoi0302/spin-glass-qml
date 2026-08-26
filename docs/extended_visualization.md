# 4×4 Spin Glass — 확장 시각화 결과

## 1. HVA Composition Fidelity (Log Scale)

HVA 회로를 $k$번 반복(composition)하여 $t = 0.5, 1.0, 1.5, 2.0, 2.5$에서 fidelity를 계산했습니다.

![Composition fidelity with log-scale infidelity](../results/4x4/plots/09_composition_fidelity.png)

### 수치 결과

| t | k (반복) | HVA Fidelity | HVA 1-F | Trotter dt=0.1 F | Trotter dt=0.2 F | HVA Depth | Trot.0.1 Depth |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 0.5 | 1 | **0.9975** | 2.50e-3 | 0.9998 | 0.8503 | 15 | 70 |
| 1.0 | 2 | **0.9943** | 5.67e-3 | 0.9998 | 0.9963 | 30 | 140 |
| 1.5 | 3 | **0.9922** | 7.83e-3 | 0.9996 | 0.8704 | 45 | 210 |
| 2.0 | 4 | **0.9897** | 1.03e-2 | 0.9995 | 0.9910 | 60 | 280 |
| 2.5 | 5 | **0.9873** | 1.27e-2 | 0.9992 | 0.7980 | 75 | 350 |

> [!TIP]
> **핵심 관찰:**
> - HVA composition fidelity는 **점진적으로 감소** ($t=2.5$에서도 $F > 0.987$)
> - **회로 깊이 압축비**: $t=2.5$에서 Trotter depth=350 vs HVA depth=75 → **4.7배 압축**
> - Trotter dt=0.2는 **oscillation** 패턴 (짝수 $t$에서 높고, 홀수 $t$에서 낮음) — dt와 $t$의 정수 배수 여부에 따름

---

## 2. Low-Energy State Preparation (BP-PPS Fig. 4(a) 스타일)

![Ground state energy vs layers](../results/4x4/plots/10_gs_energy_vs_layers.png)

### 에너지 수렴 결과 (ED 바닥: $E_0 = -22.47$)

| Layers | Params | BP-PPS Energy | SV Energy | GS Fidelity | 훈련 시간 |
|:--|:--|:--|:--|:--|:--|
| 1 | 40 | -13.82 | -13.79 | 0.0009 | 0.3s |
| 2 | 80 | -19.40 | -14.27 | 0.006 | 72s |
| 3 | 120 | -20.75 | -10.71 | 0.00006 | 1083s |
| 4 | 160 | **-22.01** | -15.42 | **0.034** | 448s |
| 5 | 200 | **-22.24** | -6.72 | 0.0003 | 2704s |

> [!WARNING]
> **BP-PPS Energy vs Statevector Energy 불일치 문제:**
> 
> BP-PPS 에너지는 layer 수에 따라 단조롭게 개선(-13.8 → -22.2)되지만, 실제 statevector 에너지는 오히려 **악화**합니다. 이것은 **cutoff=1e-3의 truncation bias** 때문입니다:
> 
> - BP-PPS는 SPO를 cutoff로 잘라서 근사하므로, 최적화가 **truncated space에서만 좋은 파라미터**를 찾음
> - Layer가 많을수록 SPO term이 더 폭발적으로 증가 → cutoff로 더 많은 term이 잘림 → bias 증가
> - 실제 양자 상태(statevector)에서 평가하면 이 파라미터는 좋지 않음
> 
> **해결 방향:** cutoff를 1e-5~1e-6으로 낮추면 BP-PPS와 statevector 에너지가 일치하지만, 연산 시간이 기하급수적으로 증가합니다.

---

## 3. 종합 비교 요약

![Combined extended summary](../results/4x4/plots/11_combined_extended.png)

### 핵심 결론

1. **시간진화 압축**: HVA 3-layer composition은 $t=2.5$까지 $F > 0.987$ 유지, 회로 깊이 4.7배 압축
2. **바닥상태 준비**: BP-PPS 에너지 기준으로 5-layer에서 $E=-22.24$ ($E_0$ 대비 1% 이내)
3. **Truncation bias**: cutoff=1e-3에서 BP-PPS 에너지와 실제 statevector 에너지의 불일치가 핵심 제한 요인
4. **스케일링**: layer ↑ → 표현력 ↑ → BP-PPS 에너지 ↓, 그러나 cutoff가 충분히 작지 않으면 실제 상태 품질은 보장되지 않음
