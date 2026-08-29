# 4×4 Spin Glass QML Simulation Results

This directory contains the simulation outputs, exact diagonalization benchmarks, variational training data, and figures for the 4×4 (16-qubit) Edwards-Anderson bimodal spin glass model ($H = -\sum_{\langle i,j \rangle} J_{ij} Z_i Z_j - h \sum_i X_i$).

> [!NOTE]
> **Re-run as of 2026-08-29** with the gate-ordering fix (`e9c3b50`). All files
> below, including `gs_trained_params.json` and plots 02–08, are now from the
> post-fix code and the ground-state optimiser targets the deployed
> `U(theta)|0>` correctly.
>
> | Check | Result |
> |:---|:---|
> | Time evolution ($t=0.5$) fidelity vs ED | 0.99918 |
> | Ground state `\|BP-PPS - statevector\|` | 0.0343 (within the tracked truncation estimate, 0.121) |
>
> `gs_multi_layer.json`, `composition_fidelity.json`, and plots 09–11 (layer
> sweep / composition, from `plot_extended.py`) are **still from before the
> fix** — that script was not re-run yet (it takes 3–6h). Treat those three
> files and plots as stale until it is.
>
> `targets_dt0.5.json` (the ED-precision target cache) was **not regenerated**
> for this run. `scripts/00_verify_targets.py` flags a MISMATCH against it
> (corner-observable deviation ~4e-6) because the same fix also made the
> truncation rule apply uniformly to commuting-branch coefficients, which the
> palindrome argument below does not cover. The deviation is far below what
> this run's losses/fidelities resolve, so it was kept rather than spending
> the ~7h regeneration; see the project owner's decision in session history.
> See `docs/extended_visualization.md` for the original bug diagnosis.

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
| ![GS vs Layers](plots/10_gs_energy_vs_layers.png) | **Low-Energy Preparation vs Layers**: Energy convergence for 1 to 5 HVA layers (BP-PPS Fig. 4a style). |
| ![Extended Summary](plots/11_combined_extended.png) | **Extended Summary**: Combined view of composition and multi-layer state preparation. |

## Data Files

- `model_config.json`: Lattice parameters, coupling values, and run configuration.
- `ed_results.json`: Exact Diagonalization ground state and time evolution benchmarks.
- `trotter_results.json`: Trotterized quantum circuit simulation benchmarks ($\Delta t=0.1, 0.2$).
- `trained_params.json`: Optimized HVA parameters and loss history for time evolution ($\Delta t=0.5$).
- `gs_trained_params.json`: Ground state preparation parameters (3 layers).
- `gs_multi_layer.json`: Low-energy state preparation data across 1, 2, 3, 4, 5 layers.
- `composition_fidelity.json`: Fidelity and depth evaluations for composed HVA blocks at $t \in [0.5, 2.5]$.
- `validation_results.json`: Summary validation metrics comparing HVA against exact diagonalization.

## Documentation

- [Detailed Simulation Walkthrough](../../docs/walkthrough_4x4.md)
- [Visualization Analysis Report](../../docs/visualization_results.md)
- [Extended Visualization & Truncation Bias Report](../../docs/extended_visualization.md)
