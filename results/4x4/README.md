# 4×4 Spin Glass QML Simulation Results

This directory contains the simulation outputs, exact diagonalization benchmarks, variational training data, and figures for the 4×4 (16-qubit) Edwards-Anderson bimodal spin glass model ($H = -\sum_{\langle i,j \rangle} J_{ij} Z_i Z_j - h \sum_i X_i$).

> [!NOTE]
> **State as of 2026-08-29.** Everything here is from the post-fix code
> (gate ordering, `e9c3b50`; Trotter step rounding, this session).
>
> | Check | Result |
> |:---|:---|
> | Time evolution ($t=0.5$) fidelity vs ED | 0.99918 |
> | Ground state `\|BP-PPS - statevector\|` | 0.0343 (within the truncation estimate, 0.121) |
> | `scripts/00_validate_small.py` | ALL 15 TESTS PASSED |
>
> Two known gaps, both with fixes already in the code and neither re-run yet:
> the ground state was trained from `|0...0>`, whose parity caps its fidelity
> at 0.5 (the config now says `|+...+>`), and it stopped 1.09 above what the
> same 3-layer ansatz can reach, which points at the truncation delta. See
> [RUNBOOK §1.5](../../docs/RUNBOOK.md) for the commands and
> [issues/01-scale-plan.md](../../docs/issues/01-scale-plan.md) for the
> evidence.
>
> `targets_dt0.5.json` was **not regenerated**; `scripts/00_verify_targets.py`
> reports a ~4e-6 MISMATCH against it because the same fix made truncation
> apply uniformly to commuting-branch coefficients. The deviation is far below
> what this run resolves, so it was kept rather than spending the ~7h
> regeneration (owner's decision, 2026-08-29).

## Overview of Figures

| Figure | Description |
|:---|:---|
| ![Lattice](plots/01_lattice_J_h.png) | **Lattice Structure**: $4\times 4$ grid with $J_{ij} = \pm 1$ couplings and transverse field $h=1.0$. |
| ![Fidelity](plots/02_fidelity_comparison.png) | **Fidelity Comparison**: State fidelity of Exact, Trotter ($S_2, \Delta t=0.1, 0.2$), and HVA. |
| ![Observables](plots/03_observable_comparison.png) | **Local Observables**: $\langle X_i \rangle, \langle Z_i \rangle$ at $t=0.5$ and deviation $|\Delta|$. |
| ![Training](plots/04_training_curves.png) | **Training Curves**: Time-evolution loss (99.9% reduction) and ground state energy minimization. |
| ![Ground State](plots/05_ground_state.png) | **Ground State Observables**: Comparison between ED and variational HVA. |
| ![Depth](plots/06_depth_comparison.png) | **Circuit Depth & 2Q Gate Count**: HVA vs Trotter circuits. |
| ![Heatmap](plots/07_loss_heatmap.png) | **Loss Heatmap**: Per-qubit training error distribution across the lattice. |
| ![Summary](plots/08_summary.png) | **Summary Panel**: Multi-panel summary in BP-PPS paper style. |
| ![Composition](plots/09_composition_fidelity.png) | **Composition Fidelity**: Long-time evolution ($t=0.5 \sim 2.5$) via repeated HVA blocks (log scale). |

## Data Files

- `model_config.json`: Lattice parameters, coupling values, and run configuration.
- `ed_results.json`: Exact Diagonalization ground state and time evolution benchmarks.
- `trotter_results.json`: Trotterized quantum circuit simulation benchmarks ($\Delta t=0.1, 0.2$).
- `trained_params.json`: Optimized HVA parameters and loss history for time evolution ($\Delta t=0.5$).
- `gs_trained_params.json`: Ground state preparation parameters (3 layers, still from `|0...0>`).
- `gs_multi_layer.PRE-FIX.json`: layer sweep (1-5) from **before** the gate-ordering fix. Kept for the trend only - do not quote the numbers. `plot_extended.py --part 2` regenerates it.
- `composition_fidelity.json`: Fidelity and depth evaluations for composed HVA blocks at $t \in [0.5, 2.5]$.
- `validation_results.json`: Summary validation metrics comparing HVA against exact diagonalization.

## Documentation

- [Results and figures, annotated](../../docs/results_4x4.md)
- [RUNBOOK — how to run this](../../docs/RUNBOOK.md)
- [Code and config reference](../../docs/manual.md)
