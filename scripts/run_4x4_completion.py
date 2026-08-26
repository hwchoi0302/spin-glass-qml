#!/usr/bin/env python3
"""4×4 Spin Glass — Ground State Training + Validation (completion script).

Runs only the remaining steps:
  - Ground state training with higher cutoff (1e-3) for speed
  - Full validation (HVA vs ED comparison)

Assumes Stage 1-2 and time-evolution training already completed.
"""

import sys
import os
import json
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from hamiltonians import SpinGlass2D
from classical_bench import ExactDiag
from ansatz import HVA
from bppps import BPPPSTrainer
from bppps.pauli_utils import make_observable_label, is_iz_only
from bppps.propagation import build_hva_gate_sequence, propagate_forward

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', '4x4')

# Config
LX, LY = 4, 4
N_QUBITS = LX * LY
H_FIELD = 1.0
SEED = 42
N_LAYERS = 3
DELTA_T = 0.5
CUTOFF_TRAIN = 1e-4
CUTOFF_GS = 1e-3  # Higher cutoff for GS training speed
N_EPOCHS_GS = 100
LR_GS = 0.05


def print_banner(msg):
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def main():
    print_banner("COMPLETION: Ground State Training + Validation")

    # Load model
    model = SpinGlass2D(Lx=LX, Ly=LY, h=H_FIELD,
                        coupling_type='ea_bimodal', seed=SEED)
    hva = HVA(num_qubits=model.num_qubits, bonds=model.bonds,
              n_layers=N_LAYERS, Lx=LX, Ly=LY, J=model.J)

    # Load existing results
    with open(os.path.join(OUTPUT_DIR, 'ed_results.json')) as f:
        ed_results = json.load(f)
    with open(os.path.join(OUTPUT_DIR, 'trained_params.json')) as f:
        te_data = json.load(f)
    te_params = np.array(te_data['params'])

    # Load targets
    with open(os.path.join(OUTPUT_DIR, f'targets_dt{DELTA_T}.json')) as f:
        targets = json.load(f)

    # ================================================================
    # Ground State Training (higher cutoff for speed)
    # ================================================================
    print_banner("Ground State Training (cutoff=1e-3, 100 epochs)")

    # Build Hamiltonian SPO
    ham_spo = {}
    for idx, (i, j) in enumerate(model.bonds):
        label = ['I'] * model.num_qubits
        label[i] = 'Z'
        label[j] = 'Z'
        key = ''.join(label)
        ham_spo[key] = ham_spo.get(key, 0.0) - model.J[idx]

    for q in range(model.num_qubits):
        label = make_observable_label(model.num_qubits, 'X', q)
        ham_spo[label] = ham_spo.get(label, 0.0) - H_FIELD

    trainer_gs = BPPPSTrainer(
        num_qubits=model.num_qubits,
        bonds=model.bonds,
        substep_bonds=hva.substep_bonds,
        n_layers=N_LAYERS,
        delta=CUTOFF_GS,  # Higher cutoff for speed
        lambda_ose=0.0,
        mode='ground_state',
        hamiltonian_spo=ham_spo,
    )

    t_start = time.time()
    gs_params, gs_losses = trainer_gs.train(
        n_epochs=N_EPOCHS_GS,
        lr=LR_GS,
        verbose=True,
    )
    gs_elapsed = time.time() - t_start
    print(f"  Training time: {gs_elapsed:.1f}s")
    print(f"  Energy: {gs_losses[0]:.6f} → {gs_losses[-1]:.6f}")

    # Save GS params
    gs_data = {
        'params': gs_params.tolist(),
        'losses': gs_losses,
        'n_layers': N_LAYERS,
        'n_params': len(gs_params),
        'n_epochs': N_EPOCHS_GS,
        'lr': LR_GS,
        'cutoff': CUTOFF_GS,
        'training_time_s': gs_elapsed,
    }
    gs_path = os.path.join(OUTPUT_DIR, 'gs_trained_params.json')
    with open(gs_path, 'w') as f:
        json.dump(gs_data, f, indent=2)
    print(f"  Saved: {gs_path}")

    # ================================================================
    # Validation
    # ================================================================
    print_banner("VALIDATION: HVA vs ED Comparison")

    # -- Time-evolution validation --
    print("\n--- HVA vs Target SPO (time-evolution, t=0.5) ---")
    gate_seq = build_hva_gate_sequence(
        model.num_qubits, model.bonds, hva.substep_bonds,
        N_LAYERS, te_params,
    )

    total_loss = 0.0
    obs_errors = {}
    for obs_key, target_spo in targets.items():
        pauli, q_str = obs_key.split('_')
        q = int(q_str)
        init_label = make_observable_label(model.num_qubits, pauli, q)
        init_spo = {init_label: 1.0}

        evolved = propagate_forward(init_spo, gate_seq, CUTOFF_TRAIN)

        all_paulis = set(evolved.keys()) | set(target_spo.keys())
        loss_g = 0.0
        for P in all_paulis:
            a = evolved.get(P, 0.0)
            a_t = target_spo.get(P, 0.0)
            loss_g += (a - a_t) ** 2
        total_loss += loss_g
        obs_errors[obs_key] = loss_g

    print(f"  Total L_XZ loss: {total_loss:.8f}")
    print(f"  Avg per observable: {total_loss / len(targets):.8f}")

    # -- Statevector comparison --
    print("\n--- HVA vs ED Statevector (t=0.5) ---")
    from qiskit.quantum_info import Statevector

    hva_qc = hva.build_circuit(te_params)
    sv_init = Statevector.from_label('0' * model.num_qubits)
    psi_hva = np.array(sv_init.evolve(hva_qc))

    H = model.build_sparse_matrix()
    ed = ExactDiag(H, model.num_qubits)
    psi0 = np.zeros(2**N_QUBITS)
    psi0[0] = 1.0
    psi_exact = ed.time_evolve(psi0, DELTA_T)

    fid = ed.state_fidelity(psi_hva, psi_exact)
    print(f"  Fidelity |⟨ψ_HVA|ψ_exact⟩|² = {fid:.10f}")

    obs_hva = ed.local_observables(psi_hva, model.bonds)
    obs_exact = ed_results['time_points']['0.5']
    x_error = np.mean(np.abs(obs_hva['X'] - np.array(obs_exact['X'])))
    z_error = np.mean(np.abs(obs_hva['Z'] - np.array(obs_exact['Z'])))
    print(f"  Mean |⟨X_i⟩_HVA - ⟨X_i⟩_ED| = {x_error:.8f}")
    print(f"  Mean |⟨Z_i⟩_HVA - ⟨Z_i⟩_ED| = {z_error:.8f}")

    E_hva = ed.compute_energy(psi_hva, model.bonds, model.J, model.h)
    E_exact = obs_exact['energy']
    print(f"  E_HVA = {E_hva:.6f}, E_exact = {E_exact:.6f}, |ΔE| = {abs(E_hva - E_exact):.8f}")

    # -- Ground state validation --
    print("\n--- Ground State Validation ---")
    E0 = ed.ground_energy()
    print(f"  ED ground energy: {E0:.6f}")
    print(f"  BP-PPS GS energy: {gs_losses[-1]:.6f}")
    print(f"  Energy gap: {gs_losses[-1] - E0:.6f}")

    gs_qc = hva.build_circuit(gs_params)
    psi_gs_hva = np.array(sv_init.evolve(gs_qc))
    energies_gs, states_gs = ed.ground_state(k=1)
    psi_gs_exact = states_gs[:, 0]
    fid_gs = ed.state_fidelity(psi_gs_hva, psi_gs_exact)
    E_gs_hva = ed.compute_energy(psi_gs_hva, model.bonds, model.J, model.h)
    print(f"  GS fidelity: {fid_gs:.10f}")
    print(f"  GS energy (statevector): {E_gs_hva:.6f}")

    # Save validation results
    validation = {
        'time_evolution': {
            'total_loss': total_loss,
            'avg_loss_per_obs': total_loss / len(targets),
            'fidelity': fid,
            'mean_x_error': float(x_error),
            'mean_z_error': float(z_error),
            'energy_hva': E_hva,
            'energy_exact': E_exact,
            'per_observable_loss': obs_errors,
        },
        'ground_state': {
            'ed_ground_energy': E0,
            'bppps_final_energy': gs_losses[-1],
            'gs_fidelity': fid_gs,
            'gs_energy_statevector': E_gs_hva,
            'energy_gap': gs_losses[-1] - E0,
        },
    }
    val_path = os.path.join(OUTPUT_DIR, 'validation_results.json')
    with open(val_path, 'w') as f:
        json.dump(validation, f, indent=2)
    print(f"\n  Validation saved: {val_path}")

    # Final summary
    print_banner("ALL STAGES COMPLETE")
    print(f"  Output: {OUTPUT_DIR}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
