#!/usr/bin/env python3
"""Build Qiskit circuits for quantum hardware execution (IBM Nighthawk).

Reads the model from ``model_config.json`` and the trained parameters from
``trained_params.json``, then emits:

  * the HVA circuit U(theta*; delta_t), optionally composed k times to reach
    T = k * delta_t;
  * the hardware Trotter baselines for the same total time;
  * OpenQASM 2 dumps of each, plus a summary JSON of the depth comparison.

Usage:
    python scripts/01_build_hw_circuits.py
    python scripts/01_build_hw_circuits.py --repeats 1 2 3 4 5
    python scripts/01_build_hw_circuits.py --basis  # measure in the X basis
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from qiskit import QuantumCircuit, qasm2                     # noqa: E402

from ansatz import HVA, TrotterCircuit                       # noqa: E402
from config import (apply_overrides, load_config, output_dir,  # noqa: E402
                    resolve_params_path)
from hamiltonians import SpinGlass2D                         # noqa: E402


def load_trained_params(path: str) -> tuple:
    """Load a trained-parameter file written by either pipeline.

    The Python trainer writes ``params``; the Julia trainer writes
    ``optimized_params``. Accept both so the two halves stay interchangeable,
    and fail loudly rather than silently picking up the wrong key.

    Returns:
        (params array, n_layers, raw record)
    """
    with open(path) as f:
        data = json.load(f)

    for key in ('params', 'optimized_params'):
        if key in data:
            params = np.asarray(data[key], dtype=float)
            break
    else:
        raise KeyError(
            f"{path} has no parameter array; expected 'params' or "
            f"'optimized_params', found {sorted(data)}"
        )

    if 'n_layers' not in data:
        raise KeyError(f"{path} does not record 'n_layers'")
    return params, int(data['n_layers']), data


def build_hva_circuit(model: SpinGlass2D, params: np.ndarray, n_layers: int,
                      repeats: int = 1, measure_basis: str = 'Z'
                      ) -> QuantumCircuit:
    """Build U(theta; delta_t)^repeats with a terminal measurement.

    Composing the trained block k times realises U(theta; k * delta_t), which
    is how the ansatz reaches long times at constant per-block depth.
    """
    hva = HVA(num_qubits=model.num_qubits, bonds=model.bonds,
              n_layers=n_layers, Lx=model.Lx, Ly=model.Ly, J=model.J)
    block = hva.build_circuit(params)

    qc = QuantumCircuit(model.num_qubits)
    for _ in range(repeats):
        qc.compose(block, inplace=True)

    _append_measurement(qc, measure_basis)
    return qc


def build_trotter_circuit(model: SpinGlass2D, t: float, dt: float,
                          order: int = 2, measure_basis: str = 'Z'
                          ) -> QuantumCircuit:
    """Build the hardware Trotter baseline for total time t."""
    trotter = TrotterCircuit(hamiltonian_op=model.get_pauli_terms(),
                             num_qubits=model.num_qubits)
    qc = trotter.build_circuit(t, dt, order)
    _append_measurement(qc, measure_basis)
    return qc


def _append_measurement(qc: QuantumCircuit, basis: str) -> None:
    """Add a computational-basis measurement, rotating first if needed.

    <X_i> is not directly measurable: a Hadamard maps the X eigenbasis onto
    the computational basis first.
    """
    if basis.upper() == 'X':
        for q in range(qc.num_qubits):
            qc.h(q)
    elif basis.upper() != 'Z':
        raise ValueError(f"measure_basis must be 'X' or 'Z', got {basis!r}")
    qc.measure_all()


def _describe(qc: QuantumCircuit) -> dict:
    ops = dict(qc.count_ops())
    two_qubit = sum(v for k, v in ops.items()
                    if k in ('rzz', 'cx', 'cz', 'ecr', 'cy'))
    return {'depth': qc.depth(), 'n_2q_gates': two_qubit, 'ops': ops}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='section.key=value')
    parser.add_argument('--repeats', type=int, nargs='+', default=[1],
                        help='HVA block repetition counts to emit')
    parser.add_argument('--basis', choices=['Z', 'X'], default='Z',
                        help="Measurement basis ('X' inserts Hadamards)")
    args = parser.parse_args()

    config = apply_overrides(load_config(), args.overrides)
    out_dir = output_dir(config)

    config_path = os.path.join(out_dir, 'model_config.json')
    if not os.path.exists(config_path):
        raise SystemExit(f"Missing {config_path}. Run scripts/00_build_model.py first.")
    with open(config_path) as f:
        model_config = json.load(f)
    model = SpinGlass2D.from_config_dict(model_config)

    params_path = resolve_params_path(out_dir, 'te',
                                      config['ansatz']['n_layers'])
    if params_path is None:
        raise SystemExit(
            f"no time-evolution parameters at "
            f"n_layers={config['ansatz']['n_layers']} in {out_dir}")
    if not os.path.exists(params_path):
        raise SystemExit(f"Missing {params_path}. Train the ansatz first.")
    params, n_layers, record = load_trained_params(params_path)

    delta_t = model_config.get('delta_t', config['target']['delta_t'])
    expected = n_layers * (model.num_qubits + model.num_bonds)
    if params.size != expected:
        raise SystemExit(
            f"{params_path} holds {params.size} parameters but "
            f"{n_layers} layers on this lattice need {expected}."
        )

    print(f"  model  : {model.Lx}x{model.Ly}, {model.num_qubits} qubits, "
          f"{model.num_bonds} bonds")
    print(f"  params : {params.size} ({n_layers} layers), "
          f"final loss {record.get('final_loss', record.get('losses', [float('nan')])[-1]):.6g}")
    print(f"  basis  : {args.basis}")

    summary = {'delta_t': delta_t, 'n_layers': n_layers,
               'measure_basis': args.basis, 'hva': {}, 'trotter': {}}

    print("\n  [1] HVA circuits")
    for k in args.repeats:
        qc = build_hva_circuit(model, params, n_layers, repeats=k,
                               measure_basis=args.basis)
        info = _describe(qc)
        summary['hva'][str(k)] = {'t': k * delta_t, **info}
        name = f"hva_x{k}_{args.basis}.qasm"
        with open(os.path.join(out_dir, name), 'w') as f:
            qasm2.dump(qc, f)
        print(f"    T={k * delta_t:4.1f} (x{k}): depth={info['depth']:4d}, "
              f"2Q={info['n_2q_gates']:4d}  -> {name}")

    print("\n  [2] Trotter baselines")
    ht = config['hardware_trotter']
    for dt in ht['dt_values']:
        summary['trotter'][str(dt)] = {}
        for k in args.repeats:
            t = k * delta_t
            steps = int(round(t / dt))
            if steps == 0:
                print(f"    dt={dt}: T={t:.1f} rounds to 0 steps, skipped")
                continue
            qc = build_trotter_circuit(model, t, dt, ht['order'], args.basis)
            info = _describe(qc)
            summary['trotter'][str(dt)][str(k)] = {'t': t, 'steps': steps, **info}
            name = f"trotter_dt{dt}_T{t:g}_{args.basis}.qasm"
            with open(os.path.join(out_dir, name), 'w') as f:
                qasm2.dump(qc, f)
            print(f"    dt={dt}, T={t:4.1f}: depth={info['depth']:4d}, "
                  f"2Q={info['n_2q_gates']:4d}  -> {name}")

    print("\n  [3] Depth compression (vs hardware Trotter, same total time)")
    for dt, per_dt in summary['trotter'].items():
        for k, tinfo in per_dt.items():
            hva_info = summary['hva'][k]
            print(f"    T={tinfo['t']:4.1f}, dt={dt}: "
                  f"depth {tinfo['depth']:4d} -> {hva_info['depth']:3d} "
                  f"({tinfo['depth'] / hva_info['depth']:.1f}x), "
                  f"2Q {tinfo['n_2q_gates']:4d} -> {hva_info['n_2q_gates']:3d} "
                  f"({tinfo['n_2q_gates'] / hva_info['n_2q_gates']:.1f}x)")

    summary_path = os.path.join(out_dir, 'hw_circuits.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written: {summary_path}")


if __name__ == '__main__':
    main()
