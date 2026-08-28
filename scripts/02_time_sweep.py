#!/usr/bin/env python3
"""T1-T: sweep the evolution time T and look for the quantum-advantage window.

The thesis of the project is that there exists a T where all three hold at once:

  (a) classical Pauli propagation / tensor networks fail,
  (b) the compressed circuit still fits the hardware error budget,
  (c) the compressed circuit is still accurate.

This script measures all three on the 4x4 lattice, where exact diagonalisation
still provides ground truth, so the answer can be checked rather than assumed.

Two ways of reaching a given T are compared:

  composition  U(theta; dt)^k     depth grows linearly in T -- the same scaling
                                  as Trotter, so compression buys only a
                                  constant factor.
  direct       U(theta; T)        trained outright at each T with a search over
                                  layer count. If the required depth grows
                                  sublinearly, that is the real compression
                                  result.

Stages:
  1  target series      T = k*chunk for every k, snapshotted from one sweep
  2  composition        reuse the T1 block, no training
  3  direct training    train at each T, find the minimum layer count
  4  analysis           SPD self-convergence, energy conservation, hardware budget

Usage:
    python scripts/02_time_sweep.py                    # all stages
    python scripts/02_time_sweep.py --stages 1         # targets only (long)
    python scripts/02_time_sweep.py --stages 2 4       # cheap stages only
    python scripts/02_time_sweep.py --set time_sweep.snapshots='[1,2,4]'

Stages 1 and 3 are long and checkpoint after every snapshot / every (T, layers)
pair, so the run can be interrupted and resumed.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from qiskit.quantum_info import Statevector                     # noqa: E402

from ansatz import HVA, TrotterCircuit                          # noqa: E402
from bppps import BPPPSTrainer, TargetGenerator                 # noqa: E402
from bppps.propagation import (                                 # noqa: E402
    TruncationStats, build_hva_gate_sequence, propagate_forward,
)
from bppps.pauli_utils import is_iz_only, make_observable_label  # noqa: E402
from bppps.warm_start import trotter_warm_start                 # noqa: E402
from classical_bench import ExactDiag, MAX_QUBITS_ED            # noqa: E402
from config import apply_overrides, load_config, output_dir     # noqa: E402
from hamiltonians import SpinGlass2D                            # noqa: E402


def banner(msg: str) -> None:
    print(f"\n{'=' * 74}\n  {msg}\n{'=' * 74}")


def section(msg: str) -> None:
    print(f"\n--- {msg} ---")


def load_model(out_dir: str) -> SpinGlass2D:
    path = os.path.join(out_dir, 'model_config.json')
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run scripts/00_build_model.py first.")
    with open(path) as f:
        return SpinGlass2D.from_config_dict(json.load(f))


def sweep_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, 'time_sweep')
    os.makedirs(path, exist_ok=True)
    return path


def target_path(sw_dir: str, k: int) -> str:
    return os.path.join(sw_dir, f'targets_k{k}.json')


def load_targets(sw_dir: str, k: int) -> dict:
    with open(target_path(sw_dir, k)) as f:
        return json.load(f)


def hamiltonian_spo(model: SpinGlass2D) -> dict:
    """H as an SPO -- the operator propagated forward in ground-state mode."""
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
# Stage 1: target series
# ============================================================================

def stage1_targets(config, model, sw_dir):
    banner("STAGE 1: target series (one sweep, snapshotted at every k)")
    ts = config['time_sweep']
    chunk = ts['chunk_delta_t']
    snapshots = list(ts['snapshots'])

    missing = [k for k in snapshots if not os.path.exists(target_path(sw_dir, k))]
    if not missing:
        print(f"  All {len(snapshots)} snapshots already present in {sw_dir}")
        return
    print(f"  Missing snapshots: {missing}")
    print(f"  (a partial run resumes from scratch: the sweep is sequential in k)")

    generator = TargetGenerator(
        num_qubits=model.num_qubits, bonds=model.bonds,
        substep_bonds=model.substep_bonds, J=model.J, h=model.h,
    )

    def checkpoint(k, targets_k, stats):
        payload = {key: {p: float(c) for p, c in spo.items()}
                   for key, spo in targets_k.items()}
        with open(target_path(sw_dir, k), 'w') as f:
            json.dump(payload, f)
        meta_path = os.path.join(sw_dir, 'targets_meta.json')
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        meta[str(k)] = {
            'T': k * chunk, 'n_terms': sum(len(v) for v in targets_k.values()),
            'largest_observable': max(len(v) for v in targets_k.values()),
            'truncation_error_estimate': stats.error_estimate,
            'dt': ts['dt'], 'order': ts['trotter_order'], 'cutoff': ts['cutoff'],
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"      saved {target_path(sw_dir, k)}")

    t0 = time.time()
    generator.generate_series(
        delta_t=chunk, snapshots=snapshots, dt_trotter=ts['dt'],
        order=ts['trotter_order'], delta=ts['cutoff'],
        observables=config['target']['observables'],
        verbose=True, checkpoint=checkpoint,
    )
    print(f"\n  Total: {time.time() - t0:.1f}s")

    _verify_against_t1(config, sw_dir, chunk)


def _verify_against_t1(config, sw_dir, chunk):
    """Cross-check the coarser sweep dt against the fine T1 target, if present."""
    tgt = config['target']
    if abs(tgt['delta_t'] - chunk) > 1e-12:
        return
    fine_path = os.path.join(os.path.dirname(sw_dir),
                             f"targets_dt{tgt['delta_t']}.json")
    if not os.path.exists(fine_path):
        print(f"\n  (no {os.path.basename(fine_path)} to cross-check dt against)")
        return

    section(f"dt cross-check: sweep dt={config['time_sweep']['dt']} "
            f"vs T1 dt={tgt['dt']}")
    with open(fine_path) as f:
        fine = json.load(f)
    coarse = load_targets(sw_dir, 1)
    worst_key, worst = None, 0.0
    for key in fine:
        a, b = coarse.get(key, {}), fine[key]
        dev = max((abs(a.get(P, 0.0) - b.get(P, 0.0))
                   for P in set(a) | set(b)), default=0.0)
        if dev > worst:
            worst_key, worst = key, dev
    print(f"  largest coefficient deviation: {worst_key} -> {worst:.3e}")
    if worst > 1e-6:
        print("  WARNING: coarser dt is not equivalent; lower time_sweep.dt.")
    else:
        print("  OK: the coarser sweep dt reproduces the fine target.")


# ============================================================================
# Stage 2: composition
# ============================================================================

def stage2_composition(config, model, out_dir, sw_dir, ed, psi0):
    banner("STAGE 2: composition  U(theta; chunk)^k  (no training)")
    ts = config['time_sweep']
    chunk = ts['chunk_delta_t']
    params_path = os.path.join(out_dir, 'trained_params.json')
    if not os.path.exists(params_path):
        print(f"  Missing {params_path}; run scripts/run_pipeline.py --stages 3 first.")
        return {}
    with open(params_path) as f:
        record = json.load(f)
    params = np.asarray(record.get('params', record.get('optimized_params')))
    n_layers = int(record['n_layers'])

    trained_chunk = record.get('delta_t', config['target']['delta_t'])
    if abs(trained_chunk - chunk) > 1e-12:
        print(f"  WARNING: trained_params.json was trained at delta_t="
              f"{trained_chunk} but time_sweep.chunk_delta_t={chunk}. "
              f"Composition results will be meaningless.")

    hva = HVA(model.num_qubits, model.bonds, n_layers,
              model.Lx, model.Ly, J=model.J)
    block = hva.build_circuit(params)
    sv0 = Statevector.from_label('0' * model.num_qubits)

    print(f"  block: {n_layers} layers, {block.count_ops().get('rzz', 0)} 2Q gates, "
          f"depth {block.depth()}")
    print(f"\n  {'T':>5} {'k':>3} {'fidelity':>12} {'1-F':>10} "
          f"{'mean|dX|':>10} {'mean|dZ|':>10} {'2Q':>6} {'depth':>6}")
    print("  " + "-" * 70)

    results = {}
    for k in sorted(ts['snapshots']):
        circuit = block.copy()
        for _ in range(k - 1):
            circuit = circuit.compose(block)
        T = k * chunk
        psi = np.array(sv0.evolve(circuit))
        psi_exact = ed.time_evolve(psi0, T)
        fid = ed.state_fidelity(psi, psi_exact)
        obs = ed.local_observables(psi, model.bonds)
        obs_ex = ed.local_observables(psi_exact, model.bonds)
        dx = float(np.mean(np.abs(obs['X'] - obs_ex['X'])))
        dz = float(np.mean(np.abs(obs['Z'] - obs_ex['Z'])))
        n2q = circuit.count_ops().get('rzz', 0)
        results[str(k)] = {
            'T': T, 'fidelity': fid, 'mean_x_error': dx, 'mean_z_error': dz,
            'n_2q_gates': n2q, 'depth': circuit.depth(), 'n_layers': n_layers * k,
        }
        print(f"  {T:5.1f} {k:3d} {fid:12.8f} {1 - fid:10.3e} "
              f"{dx:10.3e} {dz:10.3e} {n2q:6d} {circuit.depth():6d}")

    with open(os.path.join(sw_dir, 'composition.json'), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ============================================================================
# Stage 3: direct training
# ============================================================================

def stage3_direct(config, model, sw_dir, ed, psi0):
    banner("STAGE 3: direct training  U(theta; T)  (layer-count search)")
    ts = config['time_sweep']
    chunk = ts['chunk_delta_t']
    trunc = config['truncation']
    opt = config['optimizer']
    target_fid = ts['accuracy_target']

    out_path = os.path.join(sw_dir, 'direct.json')
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  resuming from {out_path} ({len(results)} entries done)")

    sv0 = Statevector.from_label('0' * model.num_qubits)

    for k in sorted(ts['snapshots']):
        T = k * chunk
        if not os.path.exists(target_path(sw_dir, k)):
            print(f"\n  T={T}: no target, skipped (run --stages 1)")
            continue
        targets = load_targets(sw_dir, k)
        psi_exact = ed.time_evolve(psi0, T)
        obs_ex = ed.local_observables(psi_exact, model.bonds)

        section(f"T = {T} (k={k})")
        for n_layers in ts['direct_layers']:
            tag = f"T{T}_L{n_layers}"
            if tag in results:
                r = results[tag]
                print(f"  L={n_layers:2d}: (cached) F={r['fidelity']:.6f}, "
                      f"2Q={r['n_2q_gates']}")
                continue

            n_params = n_layers * (model.num_qubits + model.num_bonds)
            trainer = BPPPSTrainer(
                num_qubits=model.num_qubits, bonds=model.bonds,
                substep_bonds=model.substep_bonds, n_layers=n_layers,
                delta=trunc['initial_delta'], min_delta=trunc['min_delta'],
                adaptive_delta=trunc['adaptive'], delta_factor=trunc['factor'],
                error_ratio=trunc['error_ratio'], patience=trunc['patience'],
                mode='time_evolution', target_spos=targets,
            )
            params_init = trotter_warm_start(
                model.num_qubits, model.bonds, model.J, model.h, T, n_layers)

            t0 = time.time()
            params, record = trainer.optimize(opt, params_init=params_init,
                                              verbose=False)
            hva = HVA(model.num_qubits, model.bonds, n_layers,
                      model.Lx, model.Ly, J=model.J)
            circuit = hva.build_circuit(params)
            psi = np.array(sv0.evolve(circuit))
            fid = ed.state_fidelity(psi, psi_exact)
            obs = ed.local_observables(psi, model.bonds)

            results[tag] = {
                'T': T, 'k': k, 'n_layers': n_layers, 'n_params': n_params,
                'params': params.tolist(),
                'final_loss': record['final_loss'],
                'truncation_error_estimate': record['truncation_error_estimate'],
                'final_delta': record['final_delta'],
                'fidelity': fid,
                'mean_x_error': float(np.mean(np.abs(obs['X'] - obs_ex['X']))),
                'mean_z_error': float(np.mean(np.abs(obs['Z'] - obs_ex['Z']))),
                'n_2q_gates': circuit.count_ops().get('rzz', 0),
                'depth': circuit.depth(),
                'training_time_s': time.time() - t0,
            }
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

            r = results[tag]
            flag = 'OK ' if fid >= target_fid else '   '
            print(f"  L={n_layers:2d}: {flag} F={fid:.6f}, L_XZ={r['final_loss']:.6f}, "
                  f"eps={r['truncation_error_estimate']:.2e}, "
                  f"2Q={r['n_2q_gates']:4d}, depth={r['depth']:3d}, "
                  f"{r['training_time_s']:.0f}s")

            if fid >= target_fid:
                print(f"       reached the {target_fid} target; "
                      f"skipping deeper circuits at this T")
                break

    return results


# ============================================================================
# Stage 4: analysis
# ============================================================================

def stage4_analysis(config, model, sw_dir, ed, psi0, composition, direct):
    banner("STAGE 4: analysis")
    ts = config['time_sweep']
    chunk = ts['chunk_delta_t']
    gate_err = ts['two_qubit_gate_error']
    summary = {'chunk_delta_t': chunk, 'per_T': {}}

    # --- (a) SPD self-convergence -----------------------------------------
    section("(a) SPD self-convergence and eps_emp calibration")
    print("  Observable: a single-site X and Z at the lattice centre, propagated")
    print("  through the target Trotter circuit. These genuinely evolve, unlike")
    print("  the energy, which is conserved and so cannot probe the truncation.")
    print("  At this size ED gives the true value, so eps_emp can be calibrated")
    print("  against a real error rather than only against its own Cauchy sequence.")

    from bppps.propagation import build_trotter_gate_sequence
    centre = (model.Ly // 2) * model.Lx + model.Lx // 2
    probes = [('X', centre), ('Z', centre)]

    print(f"\n  {'obs':>4} {'T':>5} {'delta':>8} {'f(delta)':>13} "
          f"{'|f-f_prev|':>11} {'eps_emp':>11} {'|f-ED|':>11} {'terms':>9}")
    print("  " + "-" * 82)

    spd = {}
    for k in sorted(ts['snapshots']):
        T = k * chunk
        n_steps = int(round(T / ts['dt']))
        seq = build_trotter_gate_sequence(
            model.num_qubits, model.substep_bonds, model.J, model.h,
            ts['dt'], n_steps, ts['trotter_order'])
        psi_exact = ed.time_evolve(psi0, T)
        obs_exact = ed.local_observables(psi_exact, model.bonds)

        for pauli, q in probes:
            truth = float(obs_exact[pauli][q])
            label = make_observable_label(model.num_qubits, pauli, q)
            prev, rows = None, []
            for delta in ts['delta_sweep']:
                stats = TruncationStats()
                evolved = propagate_forward({label: 1.0}, seq, delta, stats)
                f = sum(a for P, a in evolved.items() if is_iz_only(P))
                cauchy = abs(f - prev) if prev is not None else float('nan')
                true_err = abs(f - truth)
                rows.append({
                    'delta': delta, 'value': f, 'cauchy': cauchy,
                    'eps_emp': stats.error_estimate, 'true_error': true_err,
                    'n_terms': len(evolved),
                    'bounds_true_error': bool(stats.error_estimate >= true_err),
                })
                print(f"  {pauli}_{q:<2d} {T:5.1f} {delta:8.0e} {f:13.8f} "
                      f"{cauchy:11.3e} {stats.error_estimate:11.3e} "
                      f"{true_err:11.3e} {len(evolved):9d}")
                prev = f
            spd[f"{pauli}_{q}_k{k}"] = {'T': T, 'exact': truth, 'rows': rows}
        print()

    n_bound = sum(r['bounds_true_error'] for e in spd.values() for r in e['rows'])
    n_total = sum(len(e['rows']) for e in spd.values())
    print(f"  eps_emp upper-bounded the true error in {n_bound}/{n_total} cases.")
    print("  (Appendix B calls it an empirical estimate, not a bound; this is the")
    print("   calibration that justifies quoting it as an error bar at 10x10.)")
    summary['spd_convergence'] = spd

    # --- (b) energy conservation across initial states ---------------------
    section("(b) energy conservation across random product initial states")
    print("  <H> is conserved exactly; for a bitstring state "
          "E(0) = -sum J_ij s_i s_j is known analytically,")
    print("  so this diagnostic needs no classical reference at any lattice size.")
    rng = np.random.default_rng(ts['initial_state_seed'])
    n_states = ts['n_initial_states']
    states = rng.integers(0, 2, size=(n_states, model.num_qubits))

    best_direct = _best_per_T(direct, ts['accuracy_target'])
    econs = {}
    for k in sorted(ts['snapshots']):
        T = k * chunk
        entry = best_direct.get(T)
        if entry is None:
            continue
        hva = HVA(model.num_qubits, model.bonds, entry['n_layers'],
                  model.Lx, model.Ly, J=model.J)
        circuit = hva.build_circuit(np.asarray(entry['params']))
        drifts = []
        for s in states:
            label = ''.join('1' if b else '0' for b in s[::-1])
            sv = Statevector.from_label(label)
            E0 = -float(np.sum([model.J[idx] * (1 - 2 * ((s[i] + s[j]) & 1))
                                for idx, (i, j) in enumerate(model.bonds)]))
            psi = np.array(sv.evolve(circuit))
            ET = ed.compute_energy(psi, model.bonds, model.J, model.h)
            drifts.append(abs(ET - E0))
        econs[str(k)] = {'T': T, 'n_layers': entry['n_layers'],
                         'mean_drift': float(np.mean(drifts)),
                         'max_drift': float(np.max(drifts)),
                         'drifts': [float(d) for d in drifts]}
        print(f"  T={T:5.1f} (L={entry['n_layers']}): "
              f"mean |E(T)-E(0)| = {np.mean(drifts):.4e}, "
              f"max = {np.max(drifts):.4e}")
    summary['energy_conservation'] = econs

    # --- (c) the window ----------------------------------------------------
    section("(c) the advantage window")
    print(f"  {'T':>5} | {'compose 2Q':>10} {'F':>9} | {'direct L':>8} "
          f"{'2Q':>6} {'F':>9} | {'Trot 2Q':>8} | {'4x4 surv':>9} {'10x10 surv':>11}")
    print("  " + "-" * 92)
    trotter = TrotterCircuit(model.get_pauli_terms(), model.num_qubits)
    ht = config['hardware_trotter']
    dt_hw = ht['dt_values'][0]
    scale_10x10 = 180 / model.num_bonds

    for k in sorted(ts['snapshots']):
        T = k * chunk
        comp = composition.get(str(k), {})
        entry = best_direct.get(T)
        qc_t = trotter.build_circuit(T, dt_hw, order=ht['order'])
        n2q_t = qc_t.count_ops().get('rzz', 0)

        d_layers = entry['n_layers'] if entry else None
        d_n2q = entry['n_2q_gates'] if entry else None
        d_fid = entry['fidelity'] if entry else float('nan')
        ref_n2q = d_n2q if d_n2q else comp.get('n_2q_gates', 0)
        surv_small = np.exp(-ref_n2q * gate_err)
        surv_large = np.exp(-ref_n2q * scale_10x10 * gate_err)

        summary['per_T'][str(k)] = {
            'T': T,
            'composition': comp,
            'direct_best': entry and {kk: entry[kk] for kk in
                                      ('n_layers', 'n_2q_gates', 'depth',
                                       'fidelity', 'final_loss')},
            'trotter_2q': n2q_t,
            'survival_this_lattice': float(surv_small),
            'survival_10x10_scaled': float(surv_large),
        }
        print(f"  {T:5.1f} | {comp.get('n_2q_gates', 0):10d} "
              f"{comp.get('fidelity', float('nan')):9.6f} | "
              f"{str(d_layers):>8} {str(d_n2q):>6} {d_fid:9.6f} | "
              f"{n2q_t:8d} | {surv_small:9.2e} {surv_large:11.2e}")

    print("\n  survival = exp(-n_2q * gate_error), a crude global-fidelity proxy.")
    print("  Local observables decay far more slowly, so treat it as a lower bound.")
    print("  The 10x10 column rescales the 2Q count by the bond ratio "
          f"({scale_10x10:.1f}x) at equal layer count.")

    path = os.path.join(sw_dir, 'summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {path}")
    return summary


def _best_per_T(direct, target_fid):
    """Shallowest circuit per T that meets the accuracy target.

    If no layer count reached the target at some T, fall back to the highest
    fidelity seen there -- that point is still informative, it just marks a T
    where the ansatz ran out of expressivity within the layer budget.
    """
    passing, fallback = {}, {}
    for entry in direct.values():
        T = entry['T']
        if entry['fidelity'] >= target_fid:
            if T not in passing or entry['n_layers'] < passing[T]['n_layers']:
                passing[T] = entry
        if T not in fallback or entry['fidelity'] > fallback[T]['fidelity']:
            fallback[T] = entry
    return {T: passing.get(T, fallback[T]) for T in fallback}


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='section.key=value')
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3, 4],
                        choices=[1, 2, 3, 4])
    args = parser.parse_args()

    config = apply_overrides(load_config(), args.overrides)
    out_dir = output_dir(config)
    sw_dir = sweep_dir(out_dir)
    model = load_model(out_dir)

    banner("T1-T: TIME SWEEP")
    ts = config['time_sweep']
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  lattice  : {model.Lx}x{model.Ly} ({model.num_qubits} qubits)")
    print(f"  chunk    : {ts['chunk_delta_t']}, snapshots k={ts['snapshots']}")
    print(f"  T values : {[k * ts['chunk_delta_t'] for k in ts['snapshots']]}")
    print(f"  target   : S{ts['trotter_order']} dt={ts['dt']} cutoff={ts['cutoff']}")
    print(f"  output   : {sw_dir}")
    print(f"  stages   : {args.stages}")

    if model.num_qubits > MAX_QUBITS_ED:
        raise SystemExit(
            f"This sweep compares against exact diagonalisation and needs "
            f"<= {MAX_QUBITS_ED} qubits; this lattice has {model.num_qubits}."
        )

    ed = ExactDiag(model.build_sparse_matrix(), model.num_qubits)
    psi0 = np.zeros(2 ** model.num_qubits, dtype=complex)
    psi0[0] = 1.0

    t_start = time.time()
    composition, direct = {}, {}

    if 1 in args.stages:
        stage1_targets(config, model, sw_dir)
    if 2 in args.stages:
        composition = stage2_composition(config, model, out_dir, sw_dir, ed, psi0)
    if 3 in args.stages:
        direct = stage3_direct(config, model, sw_dir, ed, psi0)
    if 4 in args.stages:
        if not composition and os.path.exists(os.path.join(sw_dir, 'composition.json')):
            with open(os.path.join(sw_dir, 'composition.json')) as f:
                composition = json.load(f)
        if not direct and os.path.exists(os.path.join(sw_dir, 'direct.json')):
            with open(os.path.join(sw_dir, 'direct.json')) as f:
                direct = json.load(f)
        stage4_analysis(config, model, sw_dir, ed, psi0, composition, direct)

    banner("TIME SWEEP COMPLETE")
    print(f"  Total: {time.time() - t_start:.1f}s "
          f"({(time.time() - t_start) / 3600:.2f} h)")
    print(f"  Output: {sw_dir}")


if __name__ == '__main__':
    main()
