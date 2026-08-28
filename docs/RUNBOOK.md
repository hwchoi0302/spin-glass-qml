# RUNBOOK — T1 / T1-T 실행 지침

이 문서는 **데스크탑에서 시뮬레이션을 실행하는 사람 또는 AI 에이전트**를 위한 것입니다.
저장소를 처음 보는 상태에서 이 문서만 따라가면 T1 과 T1-T 를 끝내고 결과를 정리할 수 있습니다.

- 배경과 연구 논지: [benchmark_plan.md](benchmark_plan.md)
- 파이프라인 상세: [manual.md](manual.md)

---

## 0. 이 저장소가 무엇을 하는가 (30초 요약)

2D 스핀 유리 `H = -Σ J_ij Z_i Z_j - h Σ X_i` 의 시간진화 유니터리를,
파울리 전파(SPD)와 BP-PPS 역전파로 **고전 컴퓨터에서 훈련해** 얕은 양자 회로로 압축합니다.

세 가지 목표가 있고 **주장의 종류가 각각 다릅니다.**

| # | 목표 | 주장 |
|:--|:--|:--|
| ① | 임의 초기상태 동역학 | 검증 (장치가 옳게 돈다) |
| ② | 얕은 회로 샘플링 | **양자 우위** |
| ③ | 바닥상태 준비 | 회로 깊이 우위 (고전 우위 아님 — QMC 가 가능한 문제) |

**T1-T 가 답해야 하는 질문**: 아래 세 조건이 동시에 성립하는 `T` 구간이 존재하는가.

- (a) 고전 SPD 가 실패한다
- (b) 압축 회로가 하드웨어 오차 예산 안에 있다
- (c) 압축 회로가 여전히 정확하다

---

## 1. 실행 전 확인

```bash
cd /home/hyunwoo/workspace/spin-glass-qml
source .venv/bin/activate
python scripts/00_validate_small.py
```

**반드시 `ALL 14 TESTS PASSED` 가 나와야 합니다.** 수 분이면 끝납니다.

| 테스트 | 무엇을 지키나 |
|:--|:--|
| 10 | SPD 전파가 실제 Qiskit 회로의 `U†OU` 와 일치 (게이트 순서 규약) |
| 11 | BP-PPS gradient 가 중앙차분과 일치, BP-PPS 에너지 = `U|0⟩` 의 에너지 |
| 12 | 절단 오차 추정 `eps_emp` 가 실제 오차를 상한 |
| 13 | Trotter warm start → Adam → L-BFGS-B 전 구간 |
| 14 | 타겟 Trotter 시퀀스가 역순 불변 (기존 캐시 재사용 가능) |

하나라도 실패하면 **여기서 멈추고 보고하세요.** 시뮬레이션을 돌리면 안 됩니다.

---

## 2. T1 — 4×4 기준 실행

### 2-1. 모델 확인

```bash
python scripts/00_build_model.py
```

기대 출력:

```
model_config.json already matches this config: .../results/4x4/model_config.json
```

다른 메시지가 나오면 커플링이 바뀐 것이므로 **기존 결과 전체가 무효**입니다. 보고하세요.

새로 생성되는 경우의 기대값:

| 속성 | 값 |
|:--|:--|
| 큐비트 / 본드 | 16 / 24 |
| ΣJ | +4.0 |
| Frustration | 0.667 |
| substeps | 1:8, 2:4, 3:8, 4:4 |

### 2-2. 타겟 캐시 확인

`results/4x4/targets_dt0.5.json` (약 100 MB) 이 있으면 **재생성하지 마세요.**

```bash
python scripts/00_verify_targets.py --observables X_0 Z_0
```

- 노름 결손이 `1e-5` 이하 → 정상 (cutoff = 1e-8 기준)
- 코너 관측량 재생성 대조에서 `OK` → 캐시 유효, 10~20분 소요
- `MISMATCH` → 캐시를 지우고 재생성 (약 7시간)

파일이 없으면:

```bash
python scripts/run_pipeline.py --stages 1     # 약 7시간
```

### 2-3. 저비용 확인 실행 (필수, 15~30분)

**본 실행 전에 반드시 먼저 하세요.**

```bash
python scripts/run_pipeline.py --stages 2 3 4 \
    --set truncation.adaptive=false \
    --set truncation.initial_delta=1e-3 \
    --set optimizer.stage1.epochs=50 \
    --set optimizer.ground_state.epochs=50 \
    --set optimizer.stage2.enabled=false
```

**확인할 한 줄**:

```
|BP-PPS - statevector| = 0.00xxxx   (should be <= the truncation estimate ...)
```

- 이 값이 추적된 절단 오차 이내 → 통과, 다음 단계로
- 크게 벌어짐 → **멈추고 보고.** 과거에 이 값이 10.04 였던 버그가 있었습니다

### 2-4. 본 실행 (3~5시간)

```bash
python scripts/run_pipeline.py --stages 2 3 4
```

기본 설정: Trotter warm start → Adam(200 epochs) → L-BFGS-B(200 it),
절단은 `1e-3` 에서 시작해 오차 추정값이 커지면 `1e-5` 까지 자동 강화.

**기대 결과** (게이트 순서 수정 전 참고값, 재실행 후 갱신 필요):

| 항목 | 참고값 |
|:--|:--|
| 시간진화 `L_XZ` | 35.1 → 0.05 근처 |
| `T = 0.5` fidelity | 0.997 이상 |
| 평균 `|Δ⟨X_i⟩|`, `|Δ⟨Z_i⟩|` | 0.01 근처 |
| ED 바닥 에너지 | −22.4722 |
| 바닥상태 BP-PPS 에너지 | statevector 에너지와 절단 오차 이내로 일치해야 함 |

### 2-5. 그림과 회로

```bash
python scripts/plot_results.py        # 그림 1~8, 빠름
python scripts/plot_extended.py       # 그림 9~11, layer 1~5 재훈련 포함 (3~6시간)
python scripts/01_build_hw_circuits.py --repeats 1 2 3 4 5
python scripts/01_build_hw_circuits.py --repeats 1 2 3 4 5 --basis X
```

`plot_extended.py` 는 오래 걸리므로 시간이 없으면 뒤로 미뤄도 됩니다.

---

## 3. T1-T — 시간 스윕 (핵심 실험)

```bash
python scripts/02_time_sweep.py
```

`T = 0.5, 1.0, 2.0, 4.0, 8.0` 에서 (a)(b)(c) 세 조건을 동시에 측정합니다.
설정은 `configs/time_sweep.yaml` 에 있습니다.

### 단계별

| 단계 | 내용 | 예상 시간 | 재개 가능 |
|:--|:--|:--|:--|
| 1 | 타겟 시계열 생성 | **수 시간 ~ 수 일** | 스냅샷 단위 |
| 2 | 블록 합성 평가 (훈련 없음) | 수 분 | — |
| 3 | 각 `T` 에서 직접 훈련 + 층 수 탐색 | 수 시간 | (T, L) 단위 |
| 4 | 분석 (SPD 수렴, 에너지 보존, 예산) | 수 분 | — |

```bash
python scripts/02_time_sweep.py --stages 1       # 타겟만 (가장 김)
python scripts/02_time_sweep.py --stages 2 4     # 싼 단계만
python scripts/02_time_sweep.py --stages 3       # 훈련만 (재개됨)
```

### 단계 1이 왜 그나마 싼가

`V_k = B^k` 이므로 Heisenberg 그림에서 `V_k† O V_k = (B†)^k O B^k` 입니다.
즉 **한 청크씩 전파하며 스냅샷을 찍으면** `k = 1, 2, 4, 8, 16` 을 전부
`k = 16` 하나의 비용으로 얻습니다 (31배 절약). 등가성은 코드로 검증돼 있습니다.

타겟 `dt` 는 `0.01` 입니다 (T1 의 `0.001` 보다 거침). S4 오차가 `O(T·dt⁴)` 이라
`T = 8` 에서도 `8e-8` 로 절단 하한 아래이고 10배 쌉니다. 단계 1이 끝나면
스크립트가 T1 의 정밀 타겟과 자동 대조하여 이 근사를 검증합니다 —
`OK: the coarser sweep dt reproduces the fine target` 가 나와야 합니다.

### 중단과 재개

단계 1은 스냅샷마다 `results/4x4/time_sweep/targets_k<k>.json` 을 저장합니다.
**단, 스윕은 `k` 에 대해 순차적이므로 중간에 죽으면 처음부터 다시 돕니다.**
`k = 16` 까지 갈 시간이 없다고 판단되면 미리 줄이세요:

```bash
python scripts/02_time_sweep.py --stages 1 --set time_sweep.snapshots=[1,2,4,8]
```

단계 3은 `(T, layers)` 쌍마다 `direct.json` 에 저장하고 재실행 시 건너뜁니다.

### 메모리 감시

단계 1에서 SPO 가 폭발할 수 있습니다. 로그의 `largest` 열을 보세요.

```
    k=  8 (T= 4.00):   4821093 terms total,   812443 largest, eps_trunc=3.1e-06,  8123.4s
```

`largest` 가 1000만을 넘어가면 메모리가 위험합니다. 그 경우 중단하고
`time_sweep.cutoff` 를 `1e-7` 로 올린 뒤 다시 시작하되, **cutoff 를 바꿨다는 사실을
반드시 기록**하세요 (타겟 품질이 달라집니다).

---

## 4. 결과 정리 — 무엇을 보고할 것인가

산출물은 `results/4x4/time_sweep/` 에 있습니다.

| 파일 | 내용 |
|:--|:--|
| `targets_meta.json` | 각 `k` 의 항 수, 절단 오차 추정 |
| `composition.json` | 블록 합성 결과 (fidelity, 2Q, depth) |
| `direct.json` | 직접 훈련 결과 (모든 `(T, L)` 쌍) |
| `summary.json` | 세 조건 종합 |

### 반드시 채워야 할 표 4개

**표 1 — 조건 (c): 정확도 대 시간**

| T | 합성 F | 합성 2Q | 직접 최소 L | 직접 F | 직접 2Q |
|:--|:--|:--|:--|:--|:--|

핵심 질문: **직접 훈련이 합성보다 얕은가?** 합성은 깊이가 `T` 에 선형이라
Trotter 와 같은 스케일링입니다. 직접 훈련의 필요 층 수가 `T` 에 **선형보다 느리게**
늘어나면 그것이 진짜 압축 결과입니다.

**표 2 — 조건 (a): SPD 붕괴**

`summary.json` 의 `spd_convergence` 에서, 각 `T` 마다

| T | δ | f(δ) | Cauchy 차 | eps_emp | 실제 오차 (ED) |
|:--|:--|:--|:--|:--|:--|

핵심 질문 두 개:
1. **`eps_emp` 가 실제 오차를 상한하는가?** 스크립트가 비율을 세어 출력합니다
   (`eps_emp upper-bounded the true error in N/M cases`). 이것이 100큐비트에서
   `eps_emp` 를 오차막대로 쓸 근거입니다.
2. 가장 작은 `δ` 에서도 `f` 가 계속 움직이는 `T` 가 있는가? 그것이 `t*` 후보입니다.

**표 3 — 조건 (b): 하드웨어 예산**

단계 4의 "the advantage window" 표를 그대로 옮기되, `survival` 열은
**전역 fidelity 의 조잡한 하한**임을 반드시 명시하세요. 우리가 재는 것은
국소 관측량과 에너지이고 이들은 훨씬 천천히 감쇠합니다.

**표 4 — 목표 ①: 에너지 보존**

| T | 사용한 L | 평균 \|E(T)−E(0)\| | 최대 |
|:--|:--|:--|:--|

`E(0) = -Σ J_ij s_i s_j` 는 해석적으로 알려져 있으므로 **어떤 크기에서도 고전
기준선이 필요 없습니다.** 초기상태 8개에 대한 드리프트 분산이 크면, 그것 자체가
`L_XZ` (무한온도 평균) 목적함수의 초기상태 의존성을 보여주는 결과입니다.

### 결론 문장 작성 요령

세 조건이 겹치는 `T` 구간을 찾았다면 그렇게 쓰고, **못 찾았다면 못 찾았다고 쓰세요.**
못 찾은 경우 그 자체가 중요한 결과이며, 어느 조건이 먼저 깨졌는지가 다음 방향을
결정합니다.

- (c) 가 먼저 깨짐 → 안자츠 표현력 부족. 층 수 / 게이트 종류를 늘려야 함
- (a) 가 안 깨짐 → SPD 가 너무 잘 버팀. `T` 를 더 늘리거나 모델을 바꿔야 함
- (b) 가 먼저 깨짐 → 압축률이 부족. 직접 훈련 스케일링을 개선해야 함

---

## 5. 문제 발생 시

| 증상 | 대응 |
|:--|:--|
| 테스트 실패 | **멈추고 보고.** 시뮬레이션 금지 |
| `\|BP-PPS - statevector\|` 가 절단 오차보다 큼 | 멈추고 보고. 절단 문제가 아님 |
| `model_config.json already matches` 가 안 나옴 | 멈추고 보고. 기존 결과가 무효화됨 |
| 메모리 초과 | `truncation.initial_delta` 또는 `time_sweep.cutoff` 상향, **기록 필수** |
| 단계 1이 너무 오래 걸림 | `snapshots` 를 `[1,2,4,8]` 로 축소 |
| `MISMATCH` (타겟 검증) | 캐시 삭제 후 `--stages 1` 재생성 |

### 절대 하지 말아야 할 것

- **Julia 쪽에서 커플링을 재생성하지 마세요.** `model_config.json` 이 유일한 소스입니다
- **`00_build_model.py --force` 를 임의로 쓰지 마세요.** 기존 결과가 전부 무효화됩니다
- **`targets_dt0.5.json` 을 지우지 마세요** (검증에서 `MISMATCH` 가 난 경우 제외).
  재생성에 7시간이 듭니다

---

## 6. 참고 — 아직 하지 않기로 한 것

| 항목 | 상태 |
|:--|:--|
| T1.5 (5×5) | **하지 않음** (`MAX_QUBITS_ED = 26` 이라 가능은 하지만 이번엔 제외) |
| T2 (7×7) | 보류. 진입 전 bitpacking 엔진 전환 필요 |
| T3 (10×10) | 보류 |
| TN / QMC 기준선 | 보류 |
| SPD 자기수렴 수학 심화 | 보류 |

`T = 8.0` 은 10×10 하드웨어에서 **불가능**합니다 (블록 합성 시 2Q 게이트 8640개,
누적 오차 ~26, 생존 확률 ~1e-11). 4×4 에서도 2Q 1152개로 매우 어렵습니다.
그래서 T1-T 는 **하드웨어 실행이 아니라 고전 시뮬레이션 실험**이며, 목적은
"어느 `T` 까지가 현실적인가" 의 상한을 정하는 것입니다.
