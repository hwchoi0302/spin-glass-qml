# 03 — 엔진과 성능

담당 범위: Pauli propagation 엔진의 표현·속도·메모리. **7×7 진입 전에 반드시
처리해야 하는 항목들**이 여기 있습니다.

## 왜 지금 중요한가

4×4 T1-T stage 1의 k=1 스냅샷만으로 이미:

```
n_terms 8,304,835   largest_observable 1,239,943   targets_k1.json 364MB
```

문자열 키 dict 기준 항당 대략 190바이트 → 8.3M 항이 ~1.6GB. **49큐비트에서는
현재 표현으로 못 갑니다.**

## 비트패킹 (7×7부터, 4×4는 그대로)

BP-PPS 논문 Appendix C. 파울리 문자열 `P ∈ {I,X,Y,Z}^n` 을 문자열이 아니라
정수 두 개(X 마스크, Z 마스크)로 담습니다. n=49면 각각 `uint64` 하나, 즉 항 하나가
16바이트 + 계수 8바이트입니다. 문자열 dict 대비 열 배 이상 줄고, 더 중요하게는
**게이트 적용이 비트 연산**이 됩니다:

- 교환 여부 판정 = popcount 한 번
- 부호 = 마스크 AND/XOR
- 해시 = 정수 두 개

현재 `src/bppps/propagation.py` 는 `Dict[str, float]` 입니다. 전환 시 결정할 것:

- n > 64 를 처리할 방법 (10×10은 100큐비트 → `uint64` 두 쌍, 또는 numpy `uint64`
  배열 + 구조화 dtype)
- dict 를 유지할지, 정렬 배열 + 병합으로 갈지. 후자가 캐시 지역성이 훨씬 좋고
  벡터화가 되지만 코드가 커집니다
- Julia 쪽(`julia/src/bppps_engine.jl`)과 표현을 맞출지

**등가성 검증은 4×4에서** 해야 합니다 — 두 엔진이 같은 입력에 같은 계수를
내는지 비교할 수 있는 마지막 규모입니다.

## 타겟 캐시 형식

지금은 JSON입니다. 364MB/스냅샷이고 파싱이 느립니다. 비트패킹으로 가면 자연스럽게
`npz` 또는 `hdf5` (마스크 배열 + 계수 배열) 로 갑니다. 이건 비트패킹과 같은
작업으로 묶는 게 맞습니다.

## Julia 엔진

- `PauliPropagation.jl` 은 **선택적 의존성**입니다. `julia/Project.toml` 의
  `[deps]` 에서 뺐고, `HAS_PAULI_PROPAGATION` 가드로 감쌌습니다. 실제 훈련은
  `julia/src/bppps_engine.jl` 의 자체 구현이 합니다.
- `julia/scripts/check_env.jl` 가 엔진, 모델 계약, PauliPropagation API 를
  검증합니다. 데스크탑에서 Julia를 처음 쓸 때 이걸 먼저 돌리세요.
- **Julia는 커플링을 재생성하지 않습니다.** `model_config.json` 을 읽고 본드
  순서를 검증만 합니다. 예전에 Python은 `for x, y`, Julia는 `for y, x` 로 수직
  본드를 열거해 두 파이프라인이 다른 모델을 풀고 있었습니다.

## 열린 질문

- 비트패킹 전환을 Python에서 할지 Julia로 넘어갈지. Julia가 이런 커널에는
  자연스럽지만, 파이프라인 전체가 Python에 있습니다.
- 정렬 배열 병합 방식이 dict 대비 실제로 얼마나 빠른지 — 측정해야 합니다.
- 적응적 절단(adaptive truncation)이 시간 스윕에서 실제로 도움이 되는지.
  T1-T stage 4의 δ 스윕이 근거를 줍니다.
- **병렬화·GPU 는 아직 손도 대지 않았습니다.** Python `propagation.py` 도 Julia
  `bppps_engine.jl` 도 단일 스레드 `Dict{String,Float64}` 입니다. 데스크탑
  Ryzen 5 5600 은 6코어/12스레드인데 항상 1코어만 씁니다. 순서는
  ① 비트패킹(`UInt64` 키) → ② 스레드 병렬 → ③ GPU 입니다. GPU를 먼저 보면 안
  되는 이유는, 지금 커널이 문자열 해시맵 삽입이라 GPU로 옮길 대상 자체가
  아니기 때문입니다. 정렬 기반 병합으로 바꾼 뒤에야 GPU가 의미를 가집니다.
  (01 세션에서 나온 질문. 자세한 근거는 `01-scale-plan.md` 의 "왜 CPU만 쓰나".)
