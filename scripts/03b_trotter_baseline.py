#!/usr/bin/env python3
"""Trotter fidelity at *matched* 2Q-gate count, for a like-for-like comparison
against scripts/03_statevector_pilot.py's HVA ceiling numbers.

Goal 1's actual claim (CLAUDE.md) is "more accurate per unit depth than
Trotter" -- a relative claim. The pilot only measured the HVA ansatz's own
fidelity ceiling; it never built a Trotter circuit, so by itself it cannot
say whether Trotter is doing better or worse at the same depth. This script
closes that gap.

Depths are matched by *time step*, not forced to equal 2Q count: for each L
used by the HVA pilot, Trotter runs with dt = T / L, so n_steps = L exactly
(no rounding). The 2Q-gate counts that result are NOT equal between the two,
though -- confirmed empirically here, not assumed: qiskit's generic
SuzukiTrotter synthesis treats every summand of the SparsePauliOp (each bond's
ZZ term, each qubit's X term) as a separate term in the symmetric product
formula, so every RZZ appears *twice* per Trotter step (once in each half of
the order-2 symmetric split) even though all the ZZ terms actually commute --
2*n_bonds=48 two-qubit gates per step. HVA's own layers apply one RZZ per
bond once, 24 per layer. Both counts are recorded and plotted on the same 2Q
axis rather than pretending n_steps=L is a fair count match.

Trotter order is fixed at 2 (Suzuki S2), matching every other Trotter
comparison already in this repo (docs/results_4x4.md, plot_results.py,
plot_extended.py) -- order=4 is only used for stage 1's near-exact target
generation, never as the "what would actually run on hardware" baseline.
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from ansatz.trotter import TrotterCircuit                        # noqa: E402
from classical_bench import ExactDiag                             # noqa: E402
from config import load_config, output_dir                        # noqa: E402
from hamiltonians import SpinGlass2D                               # noqa: E402
from qiskit.quantum_info import Statevector                        # noqa: E402


def main():
    config = load_config()
    out_dir = output_dir(config, create=False)
    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))

    n = model.num_qubits
    n_bonds = model.num_bonds
    H = model.build_sparse_matrix()
    ed = ExactDiag(H, n)
    trotter = TrotterCircuit(model.get_pauli_terms(), n)

    ts = config.get('time_sweep', {})
    T_list = [k * ts.get('chunk_delta_t', 0.5) for k in (1, 2, 4, 8)]
    layer_list = ts.get('direct_layers', [2, 3, 4, 6, 8])

    psi0 = np.array(Statevector.from_label('0' * n))

    pilot_path = os.path.join(out_dir, 'statevector_pilot.json')
    with open(pilot_path) as f:
        out = json.load(f)
    section = out.setdefault('trotter', {})

    print(f"{'T':>5} {'L':>3} {'n_steps':>7} {'2Q':>5} {'fidelity':>12}")
    print("-" * 40)
    for T in T_list:
        psi_exact = ed.time_evolve(np.array(psi0, dtype=complex), T)
        by_T = section.setdefault(str(T), {})
        for L in layer_list:
            dt = T / L
            n_2q = trotter.count_2q_gates(T, dt, order=2)
            qc = trotter.build_circuit(T, dt, order=2)
            psi_trotter = np.array(Statevector(psi0).evolve(qc))
            fid = float(np.abs(np.vdot(psi_exact, psi_trotter)) ** 2)
            by_T[str(L)] = {'fidelity': fid, 'n_2q': n_2q, 'n_steps': L}
            print(f"{T:5.1f} {L:3d} {L:7d} {n_2q:5d} {fid:12.8f}")

    with open(pilot_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {pilot_path} (added 'trotter' section)")


if __name__ == '__main__':
    main()
