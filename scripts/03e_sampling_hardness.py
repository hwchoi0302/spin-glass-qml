#!/usr/bin/env python3
"""Does U(theta;0.5)^k anticoncentrate, and does it entangle? -- goal 2's two
preconditions, measured on the real trained circuit rather than assumed.

Goal 2's advantage argument has three conditions (docs/benchmark_plan.md):
anticoncentration, classical failure, hardware survival. The third is a
budget calculation and the second needs a 7x7 tensor-network run, but the
first can be measured right now at 4x4 on the actual BP-PPS-trained block,
and it gates everything else: a distribution that stays concentrated on a few
bitstrings is classically easy to sample no matter how deep the circuit is.

Two quantities, both from the exact statevector of U(theta;0.5)^k|0...0>:

  collision probability  Z = 2^n * sum_x p(x)^2
      Z = 1     uniform
      Z ~ 2     Porter-Thomas, i.e. the Haar-random value that hardness
                arguments for random-circuit sampling assume
      Z >> 2    concentrated on few bitstrings -- classically easy, and the
                regime docs/issues/01-scale-plan.md suspects the shallow
                T=0.5 circuit is in

  half-lattice entanglement entropy S (qubits 0-7 | 8-15, a spatial cut)
      This is the direct proxy for tensor-network cost: an MPS/PEPS across
      that cut needs bond dimension ~ 2^S. S near its 8-bit ceiling means the
      cut is maximally entangled and TN has no room left.

Reported against k so the trend, not one point, is what gets read.

Usage:
    python scripts/03e_sampling_hardness.py
    python scripts/03e_sampling_hardness.py --k-max 12
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from classical_bench import ExactDiag                                   # noqa: E402
from config import load_config, output_dir, resolve_params_path                              # noqa: E402
from hamiltonians import SpinGlass2D                                     # noqa: E402
from qiskit.quantum_info import Statevector                              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'pilot', os.path.join(PROJECT_ROOT, 'scripts', '03_statevector_pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def collision_probability(psi):
    p = np.abs(psi) ** 2
    return float(len(psi) * np.sum(p ** 2))


def half_entropy(psi, n):
    """von Neumann entropy in bits across qubits [0, n/2) | [n/2, n).

    Little-endian indexing puts the low qubits in the fast axis, so a plain
    reshape already separates the two halves.
    """
    half = 2 ** (n // 2)
    s = np.linalg.svd(psi.reshape(half, half), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-16]
    return float(-np.sum(p * np.log2(p)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k-max', type=int, default=10)
    args = parser.parse_args()

    config = load_config()
    out_dir = output_dir(config, create=False)
    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))
    n_layers = config['ansatz']['n_layers']
    tp = resolve_params_path(out_dir, 'te', n_layers)
    if tp is None:
        raise SystemExit(f'no time-evolution parameters at n_layers={n_layers} '
                         f'in {out_dir}')
    with open(tp) as f:
        trained = json.load(f)

    n = model.num_qubits
    theta = np.array(trained['params'])
    L = trained['n_layers']
    dt_block = trained.get('delta_t', 0.5)
    patterns = [pilot.zz_pattern(i, j, n) for (i, j) in model.bonds]
    plan = pilot.build_plan(n, model.bonds)(L)
    ed = ExactDiag(model.build_sparse_matrix(), n)

    psi0 = np.array(Statevector.from_label('0' * n), dtype=complex)

    # Sanity: k=1 must reproduce the recorded validation fidelity, otherwise
    # the parameter layout here does not match the trained circuit's.
    psi1 = pilot.forward_pass(theta, plan, psi0, n, patterns)[-1]
    f1 = float(np.abs(np.vdot(ed.time_evolve(psi0, dt_block), psi1)) ** 2)
    print(f"  k=1 fidelity check: {f1:.8f} (validation_results.json: 0.99918239)")
    assert abs(f1 - 0.9991823919027341) < 1e-6, \
        "parameter layout mismatch -- these are not the trained circuit's params"

    print(f"\n  block = {L} layers, delta_t = {dt_block}, {len(plan)} gates/block")
    print(f"  Haar/Porter-Thomas reference: Z ~ 2.0;  uniform: Z = 1.0")
    print(f"  max possible half-cut entropy: {n // 2} bits\n")
    print(f"  {'k':>3} {'t':>5} {'fidelity':>11} {'Z':>9} {'S (bits)':>9} "
          f"{'chi~2^S':>9}")
    print("  " + "-" * 52)

    psi = psi0.copy()
    rows = []
    for k in range(1, args.k_max + 1):
        psi = pilot.forward_pass(theta, plan, psi, n, patterns)[-1]
        t = k * dt_block
        fid = float(np.abs(np.vdot(ed.time_evolve(psi0, t), psi)) ** 2)
        Z = collision_probability(psi)
        S = half_entropy(psi, n)
        rows.append({'k': k, 't': t, 'fidelity': fid, 'collision_Z': Z,
                     'half_entropy_bits': S})
        print(f"  {k:3d} {t:5.1f} {fid:11.6f} {Z:9.3f} {S:9.3f} "
              f"{2 ** S:9.0f}")

    # The Haar value for reference: a random state on n qubits has half-cut
    # entropy of about (n/2) - 1/(2 ln 2) bits.
    haar_S = n // 2 - 1 / (2 * np.log(2))
    print(f"\n  Haar-random reference: S ~ {haar_S:.3f} bits, Z ~ 2.0")

    path = os.path.join(out_dir, 'sampling_hardness.json')
    with open(path, 'w') as f:
        json.dump({'block_layers': L, 'delta_t': dt_block,
                   'haar_entropy_bits': haar_S, 'rows': rows}, f, indent=2)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()
