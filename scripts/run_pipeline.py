#!/usr/bin/env python3
"""Full BP-PPS pipeline: targets -> classical baselines -> training -> validation.

Every setting comes from ``configs/*.yaml`` (see ``src/config.py``); nothing is
hardcoded, so the same script drives 4x4 on a laptop and 10x10 on a server.

Stages:
  1. Target SPO generation   G~ = V^dag G V for the precision Trotter V
  2. Classical baselines     exact diagonalisation + hardware Trotter circuits
  3. Training                time-evolution compression and ground state,
                             Trotter warm start -> Adam -> L-BFGS-B
  4. Validation              SPD loss, statevector fidelity, local observables

Usage:
    python scripts/00_build_model.py
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --stages 3 4          # resume
    python scripts/run_pipeline.py --set ansatz.n_layers=5

Stage 2 and the statevector checks in stage 4 need exact diagonalisation and
are skipped automatically above 22 qubits.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from ansatz import HVA, TrotterCircuit                        # noqa: E402
from bppps import BPPPSTrainer, TargetGenerator               # noqa: E402
from bppps.propagation import (                               # noqa: E402
    TruncationStats, build_hva_gate_sequence, propagate_forward,
)
from bppps.pauli_utils import make_observable_label           # noqa: E402
from bppps.warm_start import build_initial_params             # noqa: E402
from classical_bench import (                                 # noqa: E402
    ExactDiag, MAX_QUBITS_ED, MAX_QUBITS_SPARSE, statevector_gb,
)
from config import (apply_overrides, describe, load_config, output_dir,  # noqa: E402
                    params_path, resolve_params_path)
from hamiltonians import SpinGlass2D                          # noqa: E402


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}")


def section(msg: str) -> None:
    print(f"\n--- {msg} ---")


def load_model(out_dir: str) -> tuple:
    """Load the model written by scripts/00_build_model.py."""
    path = os.path.join(out_dir, 'model_config.json')
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}. Run `python scripts/00_build_model.py` first — "
            f"it is the only place the couplings are generated."
        )
    with open(path) as f:
        model_config = json.load(f)
    return SpinGlass2D.from_config_dict(model_config), model_config


def _hamiltonian_for_ed(model: SpinGlass2D):
    """Pick the cheapest exact representation of H for this lattice size.

    Up to 22 qubits the sparse matrix is fine and slightly faster. Beyond it
    the transverse field alone would need N * 2^N nonzeros, so the matrix-free
    LinearOperator is the only option.
    """
    if model.num_qubits <= MAX_QUBITS_SPARSE:
        return model.build_sparse_matrix()
    print(f"  (matrix-free Hamiltonian: {model.num_qubits} qubits, "
          f"{statevector_gb(model.num_qubits):.2f} GiB per state vector)")
    return model.build_linear_operator()


def hamiltonian_spo(model: SpinGlass2D) -> dict:
    """H = -sum J_ij Z_i Z_j - h sum X_i as an SPO.

    This is the operator propagated forward in ground-state mode: the
    Heisenberg picture moves the circuit onto the observable, and the
    observable we want the expectation value of is H itself.
    """
    spo: dict = {}
    for idx, (i, j) in enumerate(model.bonds):
        chars = ['I'] * model.num_qubits
        chars[i] = 'Z'
        chars[j] = 'Z'
        key = ''.join(chars)
        spo[key] = spo.get(key, 0.0) - model.J[idx]
    for q in range(model.num_qubits):
        key = make_observable_label(model.num_qubits, 'X', q)
        spo[key] = spo.get(key, 0.0) - model.h
    return spo


# ============================================================================
# Stage 1: Target SPO generation
# ============================================================================

def stage1_targets(config, model, out_dir):
    banner("STAGE 1: Target SPO generation")
    tgt = config['target']
    path = os.path.join(out_dir, f"targets_dt{tgt['delta_t']}.json")

    if os.path.exists(path):
        print(f"  Reusing cached targets: {path}")
        with open(path) as f:
            return json.load(f)

    print(f"  S{tgt['trotter_order']} Trotter, dt={tgt['dt']}, "
          f"delta_t={tgt['delta_t']}, cutoff={tgt['cutoff']}")
    print("  (this is a precision reference, not a hardware circuit)")

    generator = TargetGenerator(
        num_qubits=model.num_qubits, bonds=model.bonds,
        substep_bonds=model.substep_bonds, J=model.J, h=model.h,
    )
    t0 = time.time()
    targets, stats = generator.generate(
        delta_t=tgt['delta_t'], dt_trotter=tgt['dt'],
        order=tgt['trotter_order'], delta=tgt['cutoff'],
        observables=tgt['observables'], verbose=True,
    )
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print(f"  Truncation error estimate (Eq. B16): {stats.error_estimate:.3e}")

    serialisable = {k: {p: float(c) for p, c in v.items()}
                    for k, v in targets.items()}
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Saved: {path}")
    return targets


# ============================================================================
# Stage 2: Classical baselines
# ============================================================================

def stage2_baselines(config, model, out_dir):
    banner("STAGE 2: Classical baselines (ED + hardware Trotter)")
    if model.num_qubits > MAX_QUBITS_ED:
        print(f"  {model.num_qubits} qubits exceeds the ED limit "
              f"({MAX_QUBITS_ED}); skipping.")
        return None, None

    ht = config['hardware_trotter']
    time_points = ht['time_points']

    section("Exact diagonalisation")
    ed = ExactDiag(_hamiltonian_for_ed(model), model.num_qubits)
    E0 = ed.ground_energy()
    print(f"  Ground energy E0 = {E0:.8f}")

    psi0 = np.zeros(2 ** model.num_qubits)
    psi0[0] = 1.0
    ed_results = {'ground_energy': E0, 'time_points': {}}
    for t in time_points:
        psi_t = ed.time_evolve(psi0, t)
        obs = ed.local_observables(psi_t, model.bonds)
        ed_results['time_points'][str(t)] = {
            'X': obs['X'].tolist(), 'Z': obs['Z'].tolist(),
            'ZZ': obs['ZZ'].tolist(),
            'energy': ed.compute_energy(psi_t, model.bonds, model.J, model.h),
        }
        print(f"  t={t:.1f}: <X_0>={obs['X'][0]:+.6f}, <Z_0>={obs['Z'][0]:+.6f}")

    _, states = ed.ground_state(k=1)
    gs_obs = ed.local_observables(states[:, 0], model.bonds)
    ed_results['ground_state'] = {
        'X': gs_obs['X'].tolist(), 'Z': gs_obs['Z'].tolist(),
        'ZZ': gs_obs['ZZ'].tolist(), 'energy': E0,
    }
    with open(os.path.join(out_dir, 'ed_results.json'), 'w') as f:
        json.dump(ed_results, f, indent=2)

    section("Hardware Trotter (statevector)")
    from qiskit.quantum_info import Statevector
    builder = TrotterCircuit(model.get_pauli_terms(), model.num_qubits)
    sv_init = Statevector.from_label('0' * model.num_qubits)
    trotter_results = {}

    for dt in ht['dt_values']:
        trotter_results[str(dt)] = {}
        print(f"\n  dt={dt} (order {ht['order']}):")
        for t in time_points:
            # Ask the builder rather than re-deriving the step count here: it
            # rounds up and shrinks the step to t/steps so the circuit lands on
            # t exactly. A local int(round(...)) used to disagree with it and
            # skip t < dt/2 outright.
            steps = builder.num_steps(t, dt)
            qc = builder.build_circuit(t, dt, order=ht['order'])
            psi = np.array(sv_init.evolve(qc))
            obs = ed.local_observables(psi, model.bonds)
            trotter_results[str(dt)][str(t)] = {
                'X': obs['X'].tolist(), 'Z': obs['Z'].tolist(),
                'ZZ': obs['ZZ'].tolist(),
                'energy': ed.compute_energy(psi, model.bonds, model.J, model.h),
                'fidelity': ed.state_fidelity(psi, ed.time_evolve(psi0, t)),
                'depth': qc.depth(),
                'n_2q_gates': qc.count_ops().get('rzz', 0),
                'steps': steps,
            }
            r = trotter_results[str(dt)][str(t)]
            print(f"    t={t:.1f}: fid={r['fidelity']:.8f}, "
                  f"depth={r['depth']}, 2Q={r['n_2q_gates']}")

    with open(os.path.join(out_dir, 'trotter_results.json'), 'w') as f:
        json.dump(trotter_results, f, indent=2)
    return ed_results, trotter_results


# ============================================================================
# Stage 3: Training
# ============================================================================

def _make_trainer(config, model, mode, **kwargs):
    trunc = config['truncation']
    return BPPPSTrainer(
        num_qubits=model.num_qubits, bonds=model.bonds,
        substep_bonds=model.substep_bonds,
        n_layers=config['ansatz']['n_layers'],
        delta=trunc['initial_delta'], min_delta=trunc['min_delta'],
        adaptive_delta=trunc['adaptive'], delta_factor=trunc['factor'],
        error_ratio=trunc['error_ratio'], patience=trunc['patience'],
        lambda_ose=config['optimizer'].get('lambda_ose', 0.0),
        engine=trunc.get('engine', 'string'),
        mode=mode, **kwargs,
    )



def _load_params(out_dir: str, kind: str, n_layers: int) -> dict:
    """Load a trained-parameter record, or stop with the reason.

    Stage 4 validates one specific circuit, so silently validating a different
    layer count than the one configured would be worse than not running.
    """
    path = resolve_params_path(out_dir, kind, n_layers)
    if path is None:
        raise SystemExit(
            f"No '{kind}' parameters trained at n_layers={n_layers} in "
            f"{out_dir}. Run stage 3 at this layer count first, or point "
            f"--set ansatz.n_layers at one that has been trained.")
    print(f"  Using {os.path.basename(path)}")
    with open(path) as f:
        return json.load(f)


def stage3_train(config, model, targets, out_dir, parts=('te', 'gs')):
    """Run the BP-PPS training stage.

    ``parts`` selects which halves to (re)train. They are independent: the
    time-evolution half consumes the target SPOs, the ground-state half only
    needs the Hamiltonian. Iterating on a ground-state setting - the initial
    product state, the truncation delta - therefore does not need the ~78 min
    time-evolution run redone, and redoing it would also invalidate
    composition_fidelity.json for no reason.
    """
    banner("STAGE 3: BP-PPS training")
    opt = config['optimizer']
    n_layers = config['ansatz']['n_layers']
    delta_t = config['target']['delta_t']
    n_params = n_layers * (model.num_qubits + model.num_bonds)

    params_init, init_desc = build_initial_params(
        n_params, opt['init'], model.num_qubits, model.bonds,
        model.J, model.h, delta_t, n_layers,
    )
    print(f"  Initialisation: {init_desc}")

    te_record = gs_record = None

    if 'te' in parts:
        section("Time-evolution compression")
        te_trainer = _make_trainer(config, model, 'time_evolution',
                                   target_spos=targets)
        # checkpoint_path: written the moment Adam finishes, overwritten with
        # the complete record when L-BFGS-B also finishes. Without this, a
        # kill during stage 2 discards every completed Adam epoch along with
        # the angles that produced them -- see optimize()'s docstring and
        # results/4x4/gs_L5_aborted.json, which is exactly that failure.
        te_path = params_path(out_dir, 'te', n_layers)
        _, te_record = te_trainer.optimize(opt, params_init=params_init,
                                           checkpoint_path=te_path)
        te_record['delta_t'] = delta_t
        te_record['n_layers'] = n_layers
        with open(te_path, 'w') as f:
            json.dump(te_record, f, indent=2)
        print(f"  Saved: {os.path.basename(te_path)}")
        print("  NOTE: composition_fidelity.json is derived from this file - "
              "re-run `python scripts/plot_extended.py --part 1`.")

    if 'gs' not in parts:
        return te_record, gs_record

    section("Ground-state preparation")
    gs_init = opt.get('ground_state', {}).get('initial_state', 'plus')
    print(f"  Initial product state: |{'+' if gs_init == 'plus' else '0'}>^n")
    gs_trainer = _make_trainer(config, model, 'ground_state',
                               hamiltonian_spo=hamiltonian_spo(model),
                               initial_state=gs_init)
    gs_path = params_path(out_dir, 'gs', n_layers)
    _, gs_record = gs_trainer.optimize(opt, params_init=params_init,
                                       checkpoint_path=gs_path)
    gs_record['n_layers'] = n_layers
    with open(gs_path, 'w') as f:
        json.dump(gs_record, f, indent=2)
    print(f"  Saved: {os.path.basename(gs_path)}")

    return te_record, gs_record


# ============================================================================
# Stage 4: Validation
# ============================================================================

def stage4_validate(config, model, targets, te_record, gs_record,
                    ed_results, out_dir):
    banner("STAGE 4: Validation")
    n_layers = config['ansatz']['n_layers']
    delta_t = config['target']['delta_t']
    delta = config['truncation']['min_delta']

    te_params = np.array(te_record['params'])
    gs_params = np.array(gs_record['params'])
    hva = HVA(model.num_qubits, model.bonds, n_layers,
              model.Lx, model.Ly, J=model.J)

    section("HVA vs target SPO")
    gate_seq = build_hva_gate_sequence(
        model.num_qubits, model.bonds, model.substep_bonds, n_layers, te_params)
    stats = TruncationStats()
    total_loss = 0.0
    per_obs = {}
    for obs_key, target_spo in targets.items():
        pauli, q_str = obs_key.split('_')
        init_label = make_observable_label(model.num_qubits, pauli, int(q_str))
        evolved = propagate_forward({init_label: 1.0}, gate_seq, delta, stats)
        loss_g = sum((evolved.get(P, 0.0) - target_spo.get(P, 0.0)) ** 2
                     for P in set(evolved) | set(target_spo))
        total_loss += loss_g
        per_obs[obs_key] = loss_g
    print(f"  Total L_XZ (delta={delta:.0e}): {total_loss:.8f}")
    print(f"  Truncation error estimate:      {stats.error_estimate:.3e}")

    validation = {
        'time_evolution': {
            'total_loss': total_loss,
            'avg_loss_per_obs': total_loss / max(len(targets), 1),
            'truncation_error_estimate': stats.error_estimate,
            'per_observable_loss': per_obs,
        },
        'ground_state': {
            'bppps_final_energy': gs_record['final_loss'],
            'truncation_error_estimate': gs_record['truncation_error_estimate'],
        },
    }

    if model.num_qubits > MAX_QUBITS_ED or ed_results is None:
        print(f"\n  Statevector checks need ED; skipped at "
              f"{model.num_qubits} qubits.")
    else:
        from qiskit.quantum_info import Statevector
        ed = ExactDiag(_hamiltonian_for_ed(model), model.num_qubits)
        sv_init = Statevector.from_label('0' * model.num_qubits)
        psi0 = np.zeros(2 ** model.num_qubits)
        psi0[0] = 1.0

        section(f"HVA vs exact evolution (t={delta_t})")
        psi_hva = np.array(sv_init.evolve(hva.build_circuit(te_params)))
        psi_exact = ed.time_evolve(psi0, delta_t)
        fid = ed.state_fidelity(psi_hva, psi_exact)
        obs_hva = ed.local_observables(psi_hva, model.bonds)
        ref = ed_results['time_points'][str(delta_t)]
        x_err = float(np.mean(np.abs(obs_hva['X'] - np.array(ref['X']))))
        z_err = float(np.mean(np.abs(obs_hva['Z'] - np.array(ref['Z']))))
        E_hva = ed.compute_energy(psi_hva, model.bonds, model.J, model.h)
        print(f"  Fidelity              = {fid:.10f}")
        print(f"  mean |d<X_i>|         = {x_err:.8f}")
        print(f"  mean |d<Z_i>|         = {z_err:.8f}")
        print(f"  |dE|                  = {abs(E_hva - ref['energy']):.8f}")
        validation['time_evolution'].update({
            'fidelity': fid, 'mean_x_error': x_err, 'mean_z_error': z_err,
            'energy_hva': E_hva, 'energy_exact': ref['energy'],
        })

        section("Ground state")
        # Must start from the same product state the trainer optimised for,
        # otherwise the BP-PPS energy and the statevector energy are simply
        # two different quantities and the cross-check is meaningless.
        gs_init = gs_record.get('initial_state', 'zero')
        sv_gs_init = Statevector.from_label(
            ('+' if gs_init == 'plus' else '0') * model.num_qubits)
        print(f"  initial state         = |{'+' if gs_init == 'plus' else '0'}>^n")
        psi_gs = np.array(sv_gs_init.evolve(hva.build_circuit(gs_params)))
        E_sv = ed.compute_energy(psi_gs, model.bonds, model.J, model.h)
        E0 = ed.ground_energy()
        _, states = ed.ground_state(k=1)
        fid_gs = ed.state_fidelity(psi_gs, states[:, 0])
        print(f"  ED ground energy      = {E0:.6f}")
        print(f"  BP-PPS energy         = {gs_record['final_loss']:.6f}")
        print(f"  statevector energy    = {E_sv:.6f}")
        print(f"  |BP-PPS - statevector|= {abs(gs_record['final_loss'] - E_sv):.6f}"
              f"   (should be <= the truncation estimate "
              f"{gs_record['truncation_error_estimate']:.2e})")
        print(f"  GS fidelity           = {fid_gs:.10f}")
        validation['ground_state'].update({
            'ed_ground_energy': E0, 'gs_energy_statevector': E_sv,
            'gs_fidelity': fid_gs, 'initial_state': gs_init,
            'energy_gap': gs_record['final_loss'] - E0,
            'bppps_vs_statevector_gap': abs(gs_record['final_loss'] - E_sv),
        })

    path = os.path.join(out_dir, 'validation_results.json')
    with open(path, 'w') as f:
        json.dump(validation, f, indent=2)
    print(f"\n  Saved: {path}")
    return validation


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='section.key=value')
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4], help='Stages to run')
    parser.add_argument('--train', nargs='+', default=['te', 'gs'],
                        choices=['te', 'gs'],
                        help='Which halves of stage 3 to train. The two are '
                             'independent; "--train gs" skips the ~78 min '
                             'time-evolution run when only a ground-state '
                             'setting changed.')
    args = parser.parse_args()

    config = apply_overrides(load_config(), args.overrides)
    out_dir = output_dir(config)
    model, _ = load_model(out_dir)

    banner("SPIN GLASS QML — BP-PPS PIPELINE")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(describe(config))
    print(f"  output   : {out_dir}")
    print(f"  stages   : {args.stages}")
    if 3 in args.stages:
        print(f"  train    : {args.train}")

    t0 = time.time()
    targets = ed_results = te_record = gs_record = None

    if 1 in args.stages:
        targets = stage1_targets(config, model, out_dir)
    if 2 in args.stages:
        ed_results, _ = stage2_baselines(config, model, out_dir)
    if 3 in args.stages:
        if targets is None and 'te' in args.train:
            targets = stage1_targets(config, model, out_dir)
        te_record, gs_record = stage3_train(config, model, targets or {},
                                            out_dir, parts=tuple(args.train))
    if 4 in args.stages:
        if targets is None:
            targets = stage1_targets(config, model, out_dir)
        n_layers = config['ansatz']['n_layers']
        if te_record is None:
            te_record = _load_params(out_dir, 'te', n_layers)
        if gs_record is None:
            gs_record = _load_params(out_dir, 'gs', n_layers)
        if ed_results is None:
            ed_path = os.path.join(out_dir, 'ed_results.json')
            if os.path.exists(ed_path):
                with open(ed_path) as f:
                    ed_results = json.load(f)
        stage4_validate(config, model, targets, te_record, gs_record,
                        ed_results, out_dir)

    banner("PIPELINE COMPLETE")
    print(f"  Total: {time.time() - t0:.1f}s "
          f"({(time.time() - t0) / 60:.1f} min)")
    print(f"  Next : python scripts/01_build_hw_circuits.py")


if __name__ == '__main__':
    main()
