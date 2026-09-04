# scripts/archive — 끝난 일회성 스크립트

여기 있는 것들은 **한 번 돌려서 결론을 낸 뒤 다시 돌릴 일이 없는** 스크립트
입니다. 지우지 않은 이유는 이들이 아직 인용되는 데이터를 만들었기 때문입니다 —
지우면 그 JSON 의 출처가 사라집니다.

| 파일 | 무엇을 만들었나 | 지금 지위 |
|:--|:--|:--|
| `03b_trotter_baseline.py` | `statevector_pilot.json` 의 `trotter` 절 | qiskit 합성이 스텝당 RZZ 를 48개 뿜는 문제를 `03c` 가 잡음 |
| `03c_trotter_verify.py` | 같은 파일의 `trotter_grouped_s2` / `_s4` | **이 값이 지금 쓰는 Trotter 기준선입니다** |
| `03d_hva_extend.py` | 같은 파일의 깊은 `L` 행 | 안수 상한을 `L=12,16` 까지 연장 |
| `03h_lightcone_production_delta.py` | `lightcone_production_delta.json` | 프로덕션 δ 에서 빛원뿔 포화 재확인. 끝 |
| `plot_pilot_comparison.py` | `goal1_hva_vs_trotter.png` | ⚠️ **철회된 그림.** pilot ceiling 은 시간진화 회로가 아닙니다 (`a45a88a`) |

**`plot_pilot_comparison.py` 는 다시 돌리지 마세요.** 그 그림이 쓰는 pilot 최적점은
`|0…0⟩` 하나에 각도를 맞춘 것이라 곱상태 평균 infidelity 가 0.43 입니다.
지금 목표 ① 그림은 `fig1_goal1_accuracy_per_depth.png` 이고 `plot_report_4x4.py`
가 그립니다.

경로는 한 단계 깊어진 만큼 `PROJECT_ROOT` 를 고쳐 뒀으므로 저장소 어디서든
`.venv/bin/python scripts/archive/<파일>` 로 그대로 돕니다.
