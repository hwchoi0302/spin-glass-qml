#!/usr/bin/env python3
"""4×4 Spin Glass Full Simulation Pipeline.

Runs all three stages sequentially:
  1. Target SPO generation (4th-order Suzuki-Trotter, dt=0.001, T=0.5)
  2. Classical comparison data (ED exact + Trotter simulation)
  3. HVA training with BP-PPS (time-evolution + ground-state)

Usage:
    python scripts/run_4x4_simulation.py

Output directory: results/4x4/
"""

import sys
import os
import json
import time
import numpy as np

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from hamiltonians import SpinGlass2D
from classical_bench import ExactDiag
from ansatz import HVA
from bppps import TargetGenerator, BPPPSTrainer
from bppps.pauli_utils import make_observable_label, is_iz_only

# ============================================================================
# Configuration (matching BP-PPS paper parameters)
# ============================================================================
LX, LY = 4, 4
N_QUBITS = LX * LY  # 16
H_FIELD = 1.0
SEED = 42

# Target generation (paper: 4th order, dt=0.001, cutoff=1e-8)
DT_FINE = 0.001
TROTTER_ORDER = 4
CUTOFF_TARGET = 1e-8
DELTA_T = 0.5

# Training
N_LAYERS = 3
CUTOFF_TRAIN = 1e-4
N_EPOCHS_TE = 200    # Time-evolution epochs
N_EPOCHS_GS = 100    # Ground-state epochs
LR_TE = 0.01
LR_GS = 0.05

# Classical Trotter comparison
TROTTER_DT_COMPARE = [0.1, 0.2]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', '4x4')


def print_banner(msg: str):
    """Print a formatted banner."""
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def print_section(msg: str):
    """Print a section header."""
    print(f"\n--- {msg} ---")


# ============================================================================
# Stage 1: Model Creation + Target SPO Generation
# ============================================================================
def stage1_generate_targets():
    """Generate target SPOs using 4th-order Suzuki-Trotter."""
    print_banner("STAGE 1: Model Creation + Target SPO Generation")

    # Build model
    print_section("Building 4×4 spin glass model")
    model = SpinGlass2D(Lx=LX, Ly=LY, h=H_FIELD,
                        coupling_type='ea_bimodal', seed=SEED)
    print(f"  Qubits: {model.num_qubits}, Bonds: {model.num_bonds}")
    print(f"  Coupling: EA bimodal (seed={SEED})")
    print(f"  J values: {model.J}")

    # Build HVA for substep_bonds classification
    hva = HVA(num_qubits=model.num_qubits, bonds=model.bonds,
              n_layers=N_LAYERS, Lx=LX, Ly=LY, J=model.J)

    n_params = hva.count_params()
    print(f"  HVA: {N_LAYERS} layers, {n_params} params, "
          f"depth={hva.circuit_depth()}, 2Q gates={hva.count_2q_gates()}")

    # Save model config
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = {
        'Lx': LX, 'Ly': LY, 'h': H_FIELD, 'seed': SEED,
        'J': model.J.tolist(),
        'bonds': model.bonds,
        'num_qubits': model.num_qubits,
        'num_bonds': model.num_bonds,
        'n_layers': N_LAYERS,
        'delta_t': DELTA_T,
        'dt_fine': DT_FINE,
        'trotter_order': TROTTER_ORDER,
        'cutoff_target': CUTOFF_TARGET,
    }
    config_path = os.path.join(OUTPUT_DIR, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Model config saved: {config_path}")

    # Generate targets
    print_section("Generating target SPOs")
    print(f"  Δt={DELTA_T}, dt={DT_FINE}, order={TROTTER_ORDER}, "
          f"cutoff={CUTOFF_TARGET}")

    n_steps = int(round(DELTA_T / DT_FINE))
    print(f"  Trotter steps: {n_steps}")

    tgen = TargetGenerator(
        num_qubits=model.num_qubits,
        bonds=model.bonds,
        substep_bonds=hva.substep_bonds,
        J=model.J,
        h=H_FIELD,
    )

    t_start = time.time()
    targets = tgen.generate(
        delta_t=DELTA_T,
        dt_trotter=DT_FINE,
        order=TROTTER_ORDER,
        delta=CUTOFF_TARGET,
        observables='XZ',
        verbose=True,
    )
    elapsed = time.time() - t_start
    print(f"  Target generation time: {elapsed:.1f}s")

    # Save targets
    targets_serializable = {}
    for key, spo in targets.items():
        targets_serializable[key] = {p: float(c) for p, c in spo.items()}

    targets_path = os.path.join(OUTPUT_DIR, f'targets_dt{DELTA_T}.json')
    with open(targets_path, 'w') as f:
        json.dump(targets_serializable, f, indent=2)
    print(f"  Targets saved: {targets_path}")

    # Summary
    total_terms = sum(len(spo) for spo in targets.values())
    print(f"\n  Summary: {len(targets)} observables, {total_terms} total Pauli terms")
    for key in sorted(targets.keys()):
        print(f"    {key}: {len(targets[key])} terms")

    return model, hva, targets


# ============================================================================
# Stage 2: Classical Comparison Data
# ============================================================================
def stage2_classical_comparison(model, hva):
    """Generate classical comparison data: ED exact + Trotter simulation."""
    print_banner("STAGE 2: Classical Comparison Data Generation")

    # --- 2a: Exact Diagonalization ---
    print_section("Exact Diagonalization")
    H = model.build_sparse_matrix()
    ed = ExactDiag(H, model.num_qubits)

    # Ground state
    E0 = ed.ground_energy()
    print(f"  Ground energy E0 = {E0:.8f}")

    # Time evolution from |0...0⟩
    psi0 = np.zeros(2**N_QUBITS)
    psi0[0] = 1.0

    time_points = [0.1, 0.2, 0.3, 0.5, 1.0]
    ed_results = {'ground_energy': E0, 'time_points': {}}

    for t in time_points:
        psi_t = ed.time_evolve(psi0, t)
        obs = ed.local_observables(psi_t, model.bonds)
        E = ed.compute_energy(psi_t, model.bonds, model.J, model.h)

        ed_results['time_points'][str(t)] = {
            'X': obs['X'].tolist(),
            'Z': obs['Z'].tolist(),
            'ZZ': obs['ZZ'].tolist(),
            'energy': E,
        }
        print(f"  t={t:.1f}: E={E:.6f}, "
              f"⟨X_0⟩={obs['X'][0]:.6f}, ⟨Z_0⟩={obs['Z'][0]:.6f}")

    # Ground state observables
    energies, states = ed.ground_state(k=1)
    psi_gs = states[:, 0]
    gs_obs = ed.local_observables(psi_gs, model.bonds)
    ed_results['ground_state'] = {
        'X': gs_obs['X'].tolist(),
        'Z': gs_obs['Z'].tolist(),
        'ZZ': gs_obs['ZZ'].tolist(),
        'energy': E0,
    }

    ed_path = os.path.join(OUTPUT_DIR, 'ed_results.json')
    with open(ed_path, 'w') as f:
        json.dump(ed_results, f, indent=2)
    print(f"  ED results saved: {ed_path}")

    # --- 2b: Classical Trotter Simulation (via statevector) ---
    print_section("Classical Trotter Simulation (statevector)")

    from qiskit.quantum_info import Statevector
    from ansatz import TrotterCircuit
    hamiltonian_op = model.get_pauli_terms()
    trotter_builder = TrotterCircuit(hamiltonian_op, model.num_qubits)

    sv_init = Statevector.from_label('0' * model.num_qubits)
    trotter_results = {}

    for dt_comp in TROTTER_DT_COMPARE:
        trotter_results[str(dt_comp)] = {}
        print(f"\n  Trotter dt={dt_comp}:")

        for t in time_points:
            qc = trotter_builder.build_circuit(t, dt_comp, order=2)
            psi_trotter = np.array(sv_init.evolve(qc))

            # ED reference for fidelity
            psi_exact = ed.time_evolve(psi0, t)
            fid = ed.state_fidelity(psi_trotter, psi_exact)

            obs_t = ed.local_observables(psi_trotter, model.bonds)
            E_t = ed.compute_energy(psi_trotter, model.bonds, model.J, model.h)
            depth = trotter_builder.circuit_depth(t, dt_comp, order=2)
            n2q = trotter_builder.count_2q_gates(t, dt_comp, order=2)

            trotter_results[str(dt_comp)][str(t)] = {
                'X': obs_t['X'].tolist(),
                'Z': obs_t['Z'].tolist(),
                'ZZ': obs_t['ZZ'].tolist(),
                'energy': E_t,
                'fidelity': fid,
                'depth': depth,
                'n_2q_gates': n2q,
            }
            print(f"    t={t:.1f}: fid={fid:.8f}, E={E_t:.6f}, "
                  f"depth={depth}, 2Q={n2q}")

    trotter_path = os.path.join(OUTPUT_DIR, 'trotter_results.json')
    with open(trotter_path, 'w') as f:
        json.dump(trotter_results, f, indent=2)
    print(f"\n  Trotter results saved: {trotter_path}")

    return ed_results, trotter_results


# ============================================================================
# Stage 3: HVA Training with BP-PPS
# ============================================================================
def stage3_train_hva(model, hva, targets):
    """Train HVA parameters using BP-PPS."""
    print_banner("STAGE 3: HVA Training with BP-PPS")

    # --- 3a: Time-Evolution Compression ---
    print_section(f"Time-Evolution Training ({N_EPOCHS_TE} epochs)")
    print(f"  Layers: {N_LAYERS}, Params: {hva.count_params()}")
    print(f"  LR: {LR_TE}, cutoff: {CUTOFF_TRAIN}")

    trainer_te = BPPPSTrainer(
        num_qubits=model.num_qubits,
        bonds=model.bonds,
        substep_bonds=hva.substep_bonds,
        n_layers=N_LAYERS,
        delta=CUTOFF_TRAIN,
        lambda_ose=0.0,
        mode='time_evolution',
        target_spos=targets,
    )

    t_start = time.time()
    te_params, te_losses = trainer_te.train(
        n_epochs=N_EPOCHS_TE,
        lr=LR_TE,
        verbose=True,
    )
    te_elapsed = time.time() - t_start
    print(f"  Training time: {te_elapsed:.1f}s")
    print(f"  Loss: {te_losses[0]:.6f} → {te_losses[-1]:.6f} "
          f"({(1 - te_losses[-1]/te_losses[0])*100:.1f}% reduction)")

    # Save trained params
    te_data = {
        'params': te_params.tolist(),
        'losses': te_losses,
        'n_layers': N_LAYERS,
        'n_params': len(te_params),
        'n_epochs': N_EPOCHS_TE,
        'lr': LR_TE,
        'cutoff': CUTOFF_TRAIN,
        'delta_t': DELTA_T,
        'training_time_s': te_elapsed,
    }
    te_path = os.path.join(OUTPUT_DIR, 'trained_params.json')
    with open(te_path, 'w') as f:
        json.dump(te_data, f, indent=2)
    print(f"  Saved: {te_path}")

    # --- 3b: Ground State Training ---
    print_section(f"Ground State Training ({N_EPOCHS_GS} epochs)")

    # Build Hamiltonian SPO for ground state training
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
        delta=CUTOFF_TRAIN,
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
        'cutoff': CUTOFF_TRAIN,
        'training_time_s': gs_elapsed,
    }
    gs_path = os.path.join(OUTPUT_DIR, 'gs_trained_params.json')
    with open(gs_path, 'w') as f:
        json.dump(gs_data, f, indent=2)
    print(f"  Saved: {gs_path}")

    return te_params, te_losses, gs_params, gs_losses


# ============================================================================
# Stage 4: Validation & Summary
# ============================================================================
def stage4_validation(model, hva, targets, te_params, ed_results):
    """Validate trained HVA against ED and Trotter references."""
    print_banner("STAGE 4: Validation & Summary")

    # Propagate each observable through trained HVA and compare to target
    from bppps.propagation import build_hva_gate_sequence, propagate_forward

    gate_seq = build_hva_gate_sequence(
        model.num_qubits, model.bonds, hva.substep_bonds,
        N_LAYERS, te_params,
    )

    print_section("HVA vs Target SPO Comparison (t=0.5)")
    total_loss = 0.0
    obs_errors = {}

    for obs_key, target_spo in targets.items():
        pauli, q_str = obs_key.split('_')
        q = int(q_str)
        init_label = make_observable_label(model.num_qubits, pauli, q)
        init_spo = {init_label: 1.0}

        evolved = propagate_forward(init_spo, gate_seq, CUTOFF_TRAIN)

        # Compute error
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

    # Compare HVA statevector with ED at t=0.5
    print_section("HVA vs ED Statevector Comparison (t=0.5)")

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
    obs_exact_t05 = ed_results['time_points']['0.5']

    x_error = np.mean(np.abs(obs_hva['X'] - np.array(obs_exact_t05['X'])))
    z_error = np.mean(np.abs(obs_hva['Z'] - np.array(obs_exact_t05['Z'])))
    print(f"  Mean |⟨X_i⟩_HVA - ⟨X_i⟩_ED| = {x_error:.8f}")
    print(f"  Mean |⟨Z_i⟩_HVA - ⟨Z_i⟩_ED| = {z_error:.8f}")

    E_hva = ed.compute_energy(psi_hva, model.bonds, model.J, model.h)
    E_exact = obs_exact_t05['energy']
    print(f"  E_HVA = {E_hva:.6f}, E_exact = {E_exact:.6f}, "
          f"|ΔE| = {abs(E_hva - E_exact):.8f}")

    # Save validation results
    validation = {
        'total_loss': total_loss,
        'avg_loss_per_obs': total_loss / len(targets),
        'fidelity': fid,
        'mean_x_error': float(x_error),
        'mean_z_error': float(z_error),
        'energy_hva': E_hva,
        'energy_exact': E_exact,
        'per_observable_loss': obs_errors,
    }
    val_path = os.path.join(OUTPUT_DIR, 'validation_results.json')
    with open(val_path, 'w') as f:
        json.dump(validation, f, indent=2)
    print(f"\n  Validation results saved: {val_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print_banner("4×4 SPIN GLASS QML — FULL SIMULATION PIPELINE")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: {LX}×{LY}, h={H_FIELD}, seed={SEED}")
    print(f"  Target: S{TROTTER_ORDER} Trotter, dt={DT_FINE}, Δt={DELTA_T}")
    print(f"  HVA: {N_LAYERS} layers, cutoff_train={CUTOFF_TRAIN}")
    print(f"  Training: TE={N_EPOCHS_TE}ep, GS={N_EPOCHS_GS}ep")
    print(f"  Output: {OUTPUT_DIR}")

    t_total_start = time.time()

    # Stage 1: Target generation
    model, hva, targets = stage1_generate_targets()

    # Stage 2: Classical comparison
    ed_results, trotter_results = stage2_classical_comparison(model, hva)

    # Stage 3: HVA training
    te_params, te_losses, gs_params, gs_losses = stage3_train_hva(
        model, hva, targets
    )

    # Stage 4: Validation
    stage4_validation(model, hva, targets, te_params, ed_results)

    # Final summary
    t_total = time.time() - t_total_start
    print_banner("PIPELINE COMPLETE")
    print(f"  Total time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
