#!/usr/bin/env python3
"""4×4 Spin Glass — Extended Visualization.

1. HVA composition fidelity at t = 0.5k (k=1..5) + log-scale infidelity
2. Low-energy state preparation: energy vs layers (BP-PPS Fig. 4(a) style)
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from hamiltonians import SpinGlass2D
from classical_bench import ExactDiag
from ansatz import HVA, TrotterCircuit
from bppps import BPPPSTrainer
from bppps.warm_start import build_initial_params
from bppps.pauli_utils import make_observable_label
from qiskit.quantum_info import Statevector

from config import load_config, output_dir  # noqa: E402

CONFIG = load_config()
RESULTS_DIR = output_dir(CONFIG, create=False)
# Layer sweep for the low-energy state-preparation figure
LAYER_SWEEP = [1, 2, 3, 4, 5]
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Load config and data
with open(os.path.join(RESULTS_DIR, 'model_config.json')) as f:
    config = json.load(f)
with open(os.path.join(RESULTS_DIR, 'trained_params.json')) as f:
    te_data = json.load(f)
with open(os.path.join(RESULTS_DIR, 'ed_results.json')) as f:
    ed_data = json.load(f)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# Setup model
model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
H = model.build_sparse_matrix()
ed = ExactDiag(H, 16)
hamiltonian_op = model.get_pauli_terms()
trotter_builder = TrotterCircuit(hamiltonian_op, 16)
sv_init = Statevector.from_label('0' * 16)
psi0 = np.zeros(2**16); psi0[0] = 1.0

# HVA builder
te_params = np.array(te_data['params'])
hva_builder = HVA(num_qubits=16, bonds=model.bonds, n_layers=3, Lx=4, Ly=4)


# ============================================================================
# Part 1: Composition Fidelity at t = 0.5k
# ============================================================================
def compute_composition_fidelities():
    """Compute fidelity for HVA composed k times vs ED at t = 0.5k."""
    print("=" * 60)
    print("Computing HVA composition fidelities...")
    print("=" * 60)

    # Build single-chunk HVA circuit
    hva_single = hva_builder.build_circuit(te_params)

    # Time points: t = k * 0.5
    k_values = [1, 2, 3, 4, 5]
    time_pts = [k * 0.5 for k in k_values]

    hva_fidelities = []
    hva_energies = []

    for k, t in zip(k_values, time_pts):
        # Compose HVA circuit k times
        from qiskit import QuantumCircuit
        qc_composed = QuantumCircuit(16)
        for _ in range(k):
            qc_composed.compose(hva_single, inplace=True)

        psi_hva = np.array(sv_init.evolve(qc_composed))
        psi_exact = ed.time_evolve(psi0, t)
        fid = ed.state_fidelity(psi_hva, psi_exact)
        E_hva = ed.compute_energy(psi_hva, model.bonds, model.J, model.h)

        hva_fidelities.append(fid)
        hva_energies.append(E_hva)
        print(f"  t={t:.1f} (k={k}): F={fid:.8f}, 1-F={1-fid:.2e}, E={E_hva:.4f}")

    # Trotter fidelities at same time points
    trotter_01_fids = []
    trotter_02_fids = []
    trotter_01_depths = []
    trotter_02_depths = []

    for t in time_pts:
        # Trotter dt=0.1
        qc_t01 = trotter_builder.build_circuit(t, 0.1, order=2)
        psi_t01 = np.array(sv_init.evolve(qc_t01))
        psi_exact = ed.time_evolve(psi0, t)
        fid_01 = ed.state_fidelity(psi_t01, psi_exact)
        trotter_01_fids.append(fid_01)
        trotter_01_depths.append(qc_t01.depth())

        # Trotter dt=0.2
        qc_t02 = trotter_builder.build_circuit(t, 0.2, order=2)
        psi_t02 = np.array(sv_init.evolve(qc_t02))
        fid_02 = ed.state_fidelity(psi_t02, psi_exact)
        trotter_02_fids.append(fid_02)
        trotter_02_depths.append(qc_t02.depth())

        print(f"  t={t:.1f}: Trot01 F={fid_01:.8f}, Trot02 F={fid_02:.8f}")

    return {
        'time_pts': time_pts,
        'k_values': k_values,
        'hva_fid': hva_fidelities,
        'hva_energy': hva_energies,
        'trot01_fid': trotter_01_fids,
        'trot02_fid': trotter_02_fids,
        'trot01_depth': trotter_01_depths,
        'trot02_depth': trotter_02_depths,
        'hva_depth': [15 * k for k in k_values],
    }


# ============================================================================
# Part 2: Low-Energy State Preparation (energy vs layers)
# ============================================================================
def train_ground_state_multi_layers():
    """Train GS with different layer counts: 1, 2, 3, 4, 5."""
    print("\n" + "=" * 60)
    print("Training ground state with different layer counts...")
    print("=" * 60)

    # Build Hamiltonian SPO. This is the operator propagated forward in
    # ground-state mode: in the Heisenberg picture the circuit is moved onto
    # the observable, and the observable we want is H itself.
    N = model.num_qubits
    ham_spo = {}
    for idx, (i, j) in enumerate(model.bonds):
        label = ['I'] * N
        label[i] = 'Z'; label[j] = 'Z'
        key = ''.join(label)
        ham_spo[key] = ham_spo.get(key, 0.0) - model.J[idx]
    for q in range(N):
        label = make_observable_label(N, 'X', q)
        ham_spo[label] = ham_spo.get(label, 0.0) - model.h

    E0 = ed.ground_energy()
    print(f"  ED ground energy: {E0:.4f}")

    opt = CONFIG['optimizer']
    trunc = CONFIG['truncation']
    delta_t = CONFIG['target']['delta_t']
    layer_counts = LAYER_SWEEP
    all_results = {}

    for n_layers in layer_counts:
        n_params = n_layers * (N + model.num_bonds)
        print(f"\n  --- {n_layers} layers ({n_params} params) ---")
        hva_tmp = HVA(num_qubits=N, bonds=model.bonds,
                      n_layers=n_layers, Lx=model.Lx, Ly=model.Ly, J=model.J)

        trainer = BPPPSTrainer(
            num_qubits=N,
            bonds=model.bonds,
            substep_bonds=hva_tmp.substep_bonds,
            n_layers=n_layers,
            delta=trunc['initial_delta'],
            min_delta=trunc['min_delta'],
            adaptive_delta=trunc['adaptive'],
            delta_factor=trunc['factor'],
            error_ratio=trunc['error_ratio'],
            patience=trunc['patience'],
            lambda_ose=opt.get('lambda_ose', 0.0),
            mode='ground_state',
            hamiltonian_spo=ham_spo,
        )

        params_init, init_desc = build_initial_params(
            n_params, opt['init'], N, model.bonds, model.J, model.h,
            delta_t, n_layers)
        print(f"  init: {init_desc}")

        t_start = time.time()
        params, record = trainer.optimize(opt, params_init=params_init,
                                          verbose=True)
        losses = record['losses']
        elapsed = time.time() - t_start

        # Compute actual statevector energy
        hva_qc = hva_tmp.build_circuit(params)
        psi = np.array(sv_init.evolve(hva_qc))
        E_sv = ed.compute_energy(psi, model.bonds, model.J, model.h)
        energies_gs, states_gs = ed.ground_state(k=1)
        fid = ed.state_fidelity(psi, states_gs[:, 0])

        all_results[n_layers] = {
            'losses': losses,
            'final_bppps_energy': losses[-1],
            'statevector_energy': E_sv,
            'fidelity': fid,
            'n_params': len(params),
            'time_s': elapsed,
            'truncation_error_estimate': record['truncation_error_estimate'],
            'final_delta': record['final_delta'],
        }
        print(f"    BP-PPS E = {losses[-1]:.4f}, SV E = {E_sv:.4f}, "
              f"F = {fid:.6f}, time = {elapsed:.1f}s")

    return all_results, E0


# ============================================================================
# Plotting
# ============================================================================
def plot_composition_fidelity(comp_data):
    """Plot composition fidelity with log-scale infidelity."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    t = comp_data['time_pts']
    hva_fid = comp_data['hva_fid']
    t01_fid = comp_data['trot01_fid']
    t02_fid = comp_data['trot02_fid']
    hva_d = comp_data['hva_depth']
    t01_d = comp_data['trot01_depth']
    t02_d = comp_data['trot02_depth']

    # --- (a) Infidelity (log scale) vs time ---
    ax = axes[0]
    ax.semilogy(t, [1-f for f in t01_fid], 's-', color='#2196F3', markersize=8, lw=2,
               label=r'Trotter $S_2$ ($\Delta t$=0.1)')
    ax.semilogy(t, [1-f for f in t02_fid], '^-', color='#FF9800', markersize=8, lw=2,
               label=r'Trotter $S_2$ ($\Delta t$=0.2)')
    ax.semilogy(t, [1-f for f in hva_fid], 'D-', color='#E91E63', markersize=8, lw=2,
               label='HVA 3-layer (composed)')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Infidelity $1 - F$')
    ax.set_title('(a) Infidelity vs Time (log scale)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(t)

    # --- (b) Fidelity vs time ---
    ax = axes[1]
    ax.plot(t, t01_fid, 's-', color='#2196F3', markersize=8, lw=2,
           label=r'Trotter $S_2$ ($\Delta t$=0.1)')
    ax.plot(t, t02_fid, '^-', color='#FF9800', markersize=8, lw=2,
           label=r'Trotter $S_2$ ($\Delta t$=0.2)')
    ax.plot(t, hva_fid, 'D-', color='#E91E63', markersize=8, lw=2,
           label='HVA 3-layer (composed)')
    ax.axhline(y=1.0, color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Fidelity $F$')
    ax.set_title('(b) Fidelity vs Time')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(t)

    # --- (c) Infidelity vs circuit depth ---
    ax = axes[2]
    ax.semilogy(t01_d, [1-f for f in t01_fid], 's-', color='#2196F3', markersize=8, lw=2,
               label=r'Trotter $S_2$ ($\Delta t$=0.1)')
    ax.semilogy(t02_d, [1-f for f in t02_fid], '^-', color='#FF9800', markersize=8, lw=2,
               label=r'Trotter $S_2$ ($\Delta t$=0.2)')
    ax.semilogy(hva_d, [1-f for f in hva_fid], 'D-', color='#E91E63', markersize=8, lw=2,
               label='HVA 3-layer (composed)')
    ax.set_xlabel('Circuit Depth')
    ax.set_ylabel('Infidelity $1 - F$')
    ax.set_title('(c) Infidelity vs Circuit Depth')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Annotate compression ratios
    for i, ti in enumerate(t):
        if t01_d[i] > 0 and hva_d[i] > 0:
            ratio = t01_d[i] / hva_d[i]
            if ratio > 1.5:
                ax.annotate(f'{ratio:.1f}×',
                           xy=(hva_d[i], 1 - hva_fid[i]),
                           xytext=(hva_d[i] + 15, (1-hva_fid[i]) * 3),
                           fontsize=9, color='#E91E63',
                           arrowprops=dict(arrowstyle='->', color='#E91E63', lw=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '09_composition_fidelity.png'))
    plt.close()
    print("✅ 09_composition_fidelity.png")


def plot_gs_energy_vs_layers(gs_results, E0):
    """BP-PPS Fig. 4(a) style: energy vs layers + training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    layers = sorted(gs_results.keys())
    colors = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350', '#AB47BC']

    # --- (a) Energy convergence per layer (Fig 4a style) ---
    ax = axes[0]
    for i, n_l in enumerate(layers):
        losses = gs_results[n_l]['losses']
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, '-', color=colors[i], lw=2,
               label=f'{n_l} layer{"s" if n_l > 1 else ""} ({n_l*40} params)')

    ax.axhline(y=E0, color='#333333', ls='--', lw=2, label=f'ED ground ($E_0={E0:.2f}$)')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Energy $E(\\theta)$')
    ax.set_title('(a) Low-Energy State Preparation\n(BP-PPS Fig. 4a style)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- (b) Final energy vs layers ---
    ax = axes[1]
    bppps_energies = [gs_results[n]['final_bppps_energy'] for n in layers]
    sv_energies = [gs_results[n]['statevector_energy'] for n in layers]

    ax.plot(layers, bppps_energies, 'o-', color='#9C27B0', markersize=10, lw=2,
           label='BP-PPS energy (truncated)')
    ax.plot(layers, sv_energies, 's-', color='#E91E63', markersize=10, lw=2,
           label='Statevector energy (exact)')
    ax.axhline(y=E0, color='#333333', ls='--', lw=2,
              label=f'ED ground ($E_0={E0:.2f}$)')
    ax.set_xlabel('Number of HVA Layers')
    ax.set_ylabel('Energy $E$')
    ax.set_title('(b) Final Energy vs Layers')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)

    # --- (c) Relative energy error ---
    ax = axes[2]
    rel_errors_bppps = [(e - E0) / abs(E0) for e in bppps_energies]
    rel_errors_sv = [(e - E0) / abs(E0) for e in sv_energies]
    fidelities = [gs_results[n]['fidelity'] for n in layers]

    ax2 = ax.twinx()
    bars = ax.bar([l - 0.15 for l in layers], rel_errors_bppps, 0.3,
                  label='BP-PPS rel. error', color='#9C27B0', alpha=0.7)
    bars2 = ax.bar([l + 0.15 for l in layers], rel_errors_sv, 0.3,
                   label='SV rel. error', color='#E91E63', alpha=0.7)
    line = ax2.plot(layers, fidelities, 'ko-', markersize=8, lw=2,
                   label='GS fidelity')

    ax.set_xlabel('Number of HVA Layers')
    ax.set_ylabel('Relative Energy Error $(E - E_0) / |E_0|$')
    ax2.set_ylabel('Ground State Fidelity $F$')
    ax.set_title('(c) Error & Fidelity vs Layers')
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.2, axis='y')

    # Combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '10_gs_energy_vs_layers.png'))
    plt.close()
    print("✅ 10_gs_energy_vs_layers.png")


def plot_combined_summary(comp_data, gs_results, E0):
    """Combined summary of time-evolution + ground state results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    t = comp_data['time_pts']
    colors_layers = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350', '#AB47BC']
    layers = sorted(gs_results.keys())

    # --- (a) Infidelity log scale ---
    ax = axes[0, 0]
    ax.semilogy(t, [1-f for f in comp_data['trot01_fid']], 's-', color='#2196F3',
               markersize=8, lw=2, label=r'Trotter $\Delta t$=0.1')
    ax.semilogy(t, [1-f for f in comp_data['trot02_fid']], '^-', color='#FF9800',
               markersize=8, lw=2, label=r'Trotter $\Delta t$=0.2')
    ax.semilogy(t, [1-f for f in comp_data['hva_fid']], 'D-', color='#E91E63',
               markersize=8, lw=2, label='HVA (composed)')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Infidelity $1 - F$')
    ax.set_title('(a) Time-Evolution Infidelity')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(t)

    # --- (b) Circuit depth comparison ---
    ax = axes[0, 1]
    ax.plot(t, comp_data['trot01_depth'], 's-', color='#2196F3', markersize=8, lw=2,
           label=r'Trotter $\Delta t$=0.1')
    ax.plot(t, comp_data['trot02_depth'], '^-', color='#FF9800', markersize=8, lw=2,
           label=r'Trotter $\Delta t$=0.2')
    ax.plot(t, comp_data['hva_depth'], 'D-', color='#E91E63', markersize=8, lw=2,
           label='HVA (composed)')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Circuit Depth')
    ax.set_title('(b) Circuit Depth vs Time')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(t)

    # --- (c) GS energy convergence (Fig 4a style) ---
    ax = axes[1, 0]
    for i, n_l in enumerate(layers):
        losses = gs_results[n_l]['losses']
        ax.plot(range(1, len(losses)+1), losses, '-', color=colors_layers[i], lw=2,
               label=f'{n_l}L ({n_l*40}p)')
    ax.axhline(y=E0, color='#333333', ls='--', lw=2, label=f'$E_0={E0:.1f}$')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Energy $E(\\theta)$')
    ax.set_title('(c) Low-Energy State Preparation')
    ax.legend(fontsize=9, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- (d) Final energy bar chart ---
    ax = axes[1, 1]
    bppps_e = [gs_results[n]['final_bppps_energy'] for n in layers]
    sv_e = [gs_results[n]['statevector_energy'] for n in layers]

    x = np.arange(len(layers))
    w = 0.3
    ax.barh(x - w/2, bppps_e, w, label='BP-PPS energy', color='#9C27B0', alpha=0.8)
    ax.barh(x + w/2, sv_e, w, label='Statevector energy', color='#E91E63', alpha=0.8)
    ax.axvline(x=E0, color='#333333', ls='--', lw=2, label=f'$E_0={E0:.1f}$')
    ax.set_yticks(x)
    ax.set_yticklabels([f'{n}L' for n in layers])
    ax.set_xlabel('Energy $E$')
    ax.set_title('(d) Final Energy by Layer Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')

    fig.suptitle('4×4 Spin Glass — Extended Results: Composition & Low-Energy Preparation',
                fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '11_combined_extended.png'))
    plt.close()
    print("✅ 11_combined_extended.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['1', '2', 'all'], default='all',
                    help="1 = composition fidelity only (statevector, ~minutes). "
                         "2 = ground-state layer sweep only (BP-PPS training, hours). "
                         "Part 1 depends on trained_params.json, so it must be re-run "
                         "whenever the time-evolution training is re-run.")
    # Part 2's cost is not "hours" at the default truncation schedule. The
    # standalone L=5 ground-state run of 2026-08-31 spent 19.9 h on Adam alone
    # and was still inside L-BFGS-B at 41 h, because delta tightened to 1e-7 and
    # each evaluation became enormous. The sweep runs that same training five
    # times over, so it needs to be bounded rather than launched and hoped for.
    ap.add_argument('--layers', type=int, nargs='+', default=None,
                    help=f'Layer counts to sweep (default {LAYER_SWEEP})')
    ap.add_argument('--set', dest='overrides', action='append', default=[],
                    metavar='section.key=value',
                    help='Config override, e.g. truncation.min_delta=1e-6 to stop '
                         'the adaptive schedule before it reaches the regime that '
                         'made the L=5 run unbounded.')
    args = ap.parse_args()

    if args.overrides:
        from config import apply_overrides
        CONFIG = apply_overrides(CONFIG, args.overrides)
        print(f"  config overrides: {args.overrides}")
    if args.layers:
        LAYER_SWEEP = args.layers
        print(f"  layer sweep: {LAYER_SWEEP}")

    t_start = time.time()

    # Part 1: Composition fidelity
    comp_data = compute_composition_fidelities()

    # Save composition data
    comp_path = os.path.join(RESULTS_DIR, 'composition_fidelity.json')
    with open(comp_path, 'w') as f:
        json.dump(comp_data, f, indent=2)
    print(f"\nSaved: {comp_path}")

    plot_composition_fidelity(comp_data)

    if args.part == '1':
        elapsed = time.time() - t_start
        print(f"\nPart 1 only. Total time: {elapsed:.1f}s")
        raise SystemExit(0)

    # Part 2: GS training with multiple layers
    gs_results, E0 = train_ground_state_multi_layers()

    # Save GS multi-layer results
    gs_ml_path = os.path.join(RESULTS_DIR, 'gs_multi_layer.json')
    gs_save = {str(k): v for k, v in gs_results.items()}
    with open(gs_ml_path, 'w') as f:
        json.dump(gs_save, f, indent=2)
    print(f"Saved: {gs_ml_path}")

    # Plot everything
    print("\nGenerating plots...")
    plot_gs_energy_vs_layers(gs_results, E0)
    plot_combined_summary(comp_data, gs_results, E0)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Plots saved to: {PLOT_DIR}/")
