# 4×4 Spin Glass — 확장 시각화 결과

> [!CAUTION]
> **2026-08-28 정정.** 아래 2번 절의 "BP-PPS Energy vs Statevector Energy 불일치"
> 진단은 틀렸습니다. 원인은 truncation bias 가 아니라 파울리 전파 엔진의 게이트
> 순서 관례 버그였고, 그 결과 바닥상태 데이터 전체가 무효입니다. 1번 절의
> composition fidelity 는 유효합니다. 자세한 내용은 각 절의 정정 상자를 보세요.

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
> - HVA composition fidelity 는 **점진적으로 감소** ( $t = 2.5$ 에서도 $F > 0.987$ )
> - **회로 깊이 압축비**: $t = 2.5$ 에서 2차 Trotter( $dt = 0.1$ ) depth 351 / 2Q 1200 대비
>   HVA depth 76 / 2Q 360 → 깊이 4.6배, 2큐빗 게이트 3.3배
> - Trotter $dt = 0.2$ 는 **oscillation** 패턴 (짝수 $t$ 에서 높고 홀수 $t$ 에서 낮음) —
>   $dt$ 와 $t$ 의 정수 배수 여부에 따름

---

## 2. Low-Energy State Preparation (BP-PPS Fig. 4(a) 스타일)

![Ground state energy vs layers](../results/4x4/plots/10_gs_energy_vs_layers.png)

### 에너지 수렴 결과 (무효)

> [!CAUTION]
> **이 표 전체가 무효입니다.** 아래 수치는 배포될 상태 `U(θ)|0⟩` 가 아니라
> 켤레 상태 `U(θ)†|0⟩` 를 최적화한 결과입니다. 재실행 전까지 인용하지 마세요.

| Layers | Params | BP-PPS Energy | SV Energy | GS Fidelity | 훈련 시간 |
|:--|:--|:--|:--|:--|:--|
| 1 | 40 | -13.82 | -13.79 | 0.0009 | 0.3s |
| 2 | 80 | -19.40 | -14.27 | 0.006 | 72s |
| 3 | 120 | -20.75 | -10.71 | 0.00006 | 1083s |
| 4 | 160 | -22.01 | -15.42 | 0.034 | 448s |
| 5 | 200 | -22.24 | -6.72 | 0.0003 | 2704s |

#### 정정: 불일치의 진짜 원인

원래 이 문서는 BP-PPS 에너지가 layer 수에 따라 단조 개선(−13.8 → −22.2)되는데
statevector 에너지는 오히려 악화하는 현상을 **cutoff = 1e-3 의 truncation bias**
로 설명했습니다. 이 설명은 틀렸습니다.

실제 원인은 `propagate_forward` 가 게이트 시퀀스를 회로 순서대로 소비한 것입니다.
게이트 목록 `[g₁, …, g_T]` 에 대해 이 루프는

$$g_T^\dagger \cdots g_1^\dagger \, O \, g_1 \cdots g_T$$

를 계산하는데, 이는 `U = g₁g₂⋯g_T` 즉 게이트가 **역순으로 적용되는 회로**에
해당합니다. Qiskit 회로의 유니터리는 `U_c = g_T⋯g₁` 이고, RX·RZZ 는 모두 실대칭
생성자를 가지므로 `g^T = g`, 따라서 엔진이 전파한 것은 `U_c^T` 였습니다.
그 결과 바닥상태 모드는 `⟨0|U_c H U_c†|0⟩`, 즉 `U(θ)†|0⟩` 의 에너지를
최소화했습니다.

저장된 3-layer 파라미터로 직접 확인한 값:

```
BP-PPS reported final energy : -20.7512
E[ U†|0> ]  (SPD가 실제로 최적화한 것) : -20.7527   ← 일치, 차이 1.5e-3 = 진짜 절단 오차
E[ U|0>  ]  (하드웨어에 배포될 상태)   : -10.7092   ← 문서가 "SV Energy" 로 보고한 값
ED ground energy                       : -22.4722
```

즉 truncation 이 설명하는 부분은 10.04 중 0.0015 뿐입니다. "cutoff 를 1e-5~1e-6 으로
낮추면 일치한다" 는 제안도 성립하지 않습니다 — 수정 전에는 아무리 낮춰도 일치하지
않았을 것입니다.

시간진화 모드가 살아남은 이유는 `H` 가 실대칭이라 `V = exp(-iHΔt)` 가
complex-symmetric( $V^T = V$ )이고, 타겟 $S_4$ Trotter 시퀀스도 palindromic 이라
역순 불변이기 때문입니다. `U_c^T ≈ V` 와 `U_c ≈ V^T = V` 가 같은 조건이 되어
우연히 상쇄됩니다.

#### 수정 내용

게이트 시퀀스는 회로 순서로 유지하고, `propagate_forward` 가 역순으로,
`propagate_backward` 가 정순으로 순회하도록 바꿨습니다 (관측량은 회로 출력에
있으므로 입력 쪽으로 밀려나야 합니다). 아울러:

- `scripts/00_validate_small.py` 에 회귀 테스트 3개를 추가했습니다.
  TEST 10 은 SPD 전파를 실제 Qiskit 회로의 $U^\dagger O U$ 와 대조하고,
  TEST 11 은 gradient 를 중앙차분과 대조하며, TEST 13 은 BP-PPS 에너지가
  `U|0⟩` 의 에너지와 일치하는지 확인합니다.
- BP-PPS Appendix B 의 절단 오차 추정(Eq. B16)을 구현해, 검증 단계가
  `|BP-PPS − statevector|` 를 추적된 오차와 직접 비교합니다. 이 진단이 있었다면
  위 오진은 바로 걸러졌을 것입니다.

## 3. 종합 비교 요약

![Combined extended summary](../results/4x4/plots/11_combined_extended.png)

### 핵심 결론

1. **시간진화 압축**: HVA 3-layer composition은 $t=2.5$까지 $F > 0.987$ 유지, 회로 깊이 4.7배 압축
2. **바닥상태 준비**: BP-PPS 에너지 기준으로 5-layer에서 $E=-22.24$ ($E_0$ 대비 1% 이내)
3. **Truncation bias**: cutoff=1e-3에서 BP-PPS 에너지와 실제 statevector 에너지의 불일치가 핵심 제한 요인
4. **스케일링**: layer ↑ → 표현력 ↑ → BP-PPS 에너지 ↓, 그러나 cutoff가 충분히 작지 않으면 실제 상태 품질은 보장되지 않음
