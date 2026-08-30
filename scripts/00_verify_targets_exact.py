#!/usr/bin/env python3
"""Validate stage-1 targets against *exact* time evolution, independently.

scripts/00_verify_targets.py regenerates an observable with the same
BP-PPS propagation the cache was built with and compares -- a
self-consistency check. It would pass even if the Trotter sequence, the
propagation rules, or the sign conventions were all consistently wrong.
Nothing in the repo has ever checked stage-1 targets against a method that
shares no code with them.

This does. The identity used is

    <psi| Xtilde_i(T) |psi>  =  <psi(T)| X_i |psi(T)>,   psi(T) = exp(-iHT)|psi>

so the right-hand side needs exactly ONE exact statevector evolution (scipy
expm_multiply via ExactDiag, no Trotter, no truncation), while the left-hand
side is read straight off the target SPO as sum_P a_P <psi|P|psi>. Agreement
on random states tests the whole operator -- every Pauli term, coefficient
and sign at once -- without ever materialising the 2^n x 2^n operator that
building the target "directly by statevector" would require.

Note what this can and cannot conclude. It is a *contraction* of the target
against a state, so a discrepancy proves the target is wrong, while agreement
on random states is strong (not absolute) evidence it is right: an error
confined to Paulis that happen to be near-orthogonal to every test state
could hide. Random states make that unlikely, and more states make it more
unlikely; it is not a proof of term-by-term equality.

Usage:
    python scripts/00_verify_targets_exact.py
    python scripts/00_verify_targets_exact.py --observables X_0 Z_0 --states 8
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from bppps.pauli_utils import make_observable_label                    # noqa: E402
from bppps.propagation import (                                        # noqa: E402
    TruncationStats, build_trotter_gate_sequence, propagate_forward,
)
from classical_bench import ExactDiag                                   # noqa: E402
from config import apply_overrides, load_config, output_dir             # noqa: E402
from hamiltonians import SpinGlass2D                                     # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector               # noqa: E402


def pauli_expectations(labels, psi, n):
    """<psi|P|psi> for a list of Pauli labels, vectorised over basis states.

    Label convention follows the repo's SPO keys: label[k] is the operator on
    qubit k (little-endian), so it is reversed before handing to qiskit.
    Uses P = i^{|x&z|} X^x Z^z, which is validated against qiskit below rather
    than assumed.
    """
    idx = np.arange(len(psi), dtype=np.int64)
    conj_psi = np.conj(psi)
    out = np.empty(len(labels), dtype=complex)
    for t, label in enumerate(labels):
        x = z = 0
        for k, ch in enumerate(label):
            if ch in 'XY':
                x |= 1 << k
            if ch in 'ZY':
                z |= 1 << k
        n_y = bin(x & z).count('1')
        flipped = idx ^ x
        sign = 1.0 - 2.0 * (np.bitwise_count(flipped & z) & 1)
        out[t] = (1j ** n_y) * np.sum(conj_psi * sign * psi[flipped])
    return out


def check_pauli_helper(n, rng):
    """The helper above is the one piece of new maths here -- test it first."""
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    labels = [''.join(rng.choice(list('IXYZ'), size=n)) for _ in range(12)]
    mine = pauli_expectations(labels, psi, n)
    sv = Statevector(psi)
    ref = np.array([sv.expectation_value(SparsePauliOp(lb[::-1])) for lb in labels])
    dev = np.max(np.abs(mine - ref))
    print(f"  Pauli-expectation helper vs qiskit: max dev = {dev:.3e}")
    assert dev < 1e-10, "pauli_expectations disagrees with qiskit"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='section.key=value')
    parser.add_argument('--observables', nargs='+', default=['X_0', 'Z_0'])
    parser.add_argument('--states', type=int, default=6)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    args = parser.parse_args()

    config = apply_overrides(load_config(), args.overrides)
    out_dir = output_dir(config, create=False)
    tgt = config['target']
    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))
    n = model.num_qubits
    T = tgt['delta_t']

    rng = np.random.default_rng(0)
    print("=" * 72)
    print("  Stage-1 targets vs exact evolution (no shared code path)")
    print("=" * 72)
    check_pauli_helper(n, rng)

    n_steps = int(round(T / tgt['dt']))
    gate_seq = build_trotter_gate_sequence(
        n, model.substep_bonds, model.J, model.h,
        dt=tgt['dt'], n_steps=n_steps, order=tgt['trotter_order'])
    print(f"  target recipe: S{tgt['trotter_order']}, dt={tgt['dt']}, "
          f"{len(gate_seq)} gates, cutoff={tgt['cutoff']}, T={T}\n")

    ed = ExactDiag(model.build_sparse_matrix(), n)

    states = []
    for _ in range(args.states):
        v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        states.append(v / np.linalg.norm(v))
    evolved = [ed.time_evolve(np.array(v, dtype=complex), T) for v in states]

    worst = 0.0
    for key in args.observables:
        pauli, q_str = key.split('_')
        label = make_observable_label(n, pauli, int(q_str))
        stats = TruncationStats()
        t0 = time.time()
        spo = propagate_forward({label: 1.0}, gate_seq, tgt['cutoff'], stats)
        gen_s = time.time() - t0
        labels = list(spo.keys())
        coeffs = np.array([spo[k] for k in labels])
        print(f"  {key}: {len(labels)} terms, eps_emp={stats.error_estimate:.3e}, "
              f"{gen_s:.1f}s")

        q = int(q_str)
        for si, (v, vT) in enumerate(zip(states, evolved)):
            from_spo = float(np.real(coeffs @ pauli_expectations(labels, v, n)))
            exact = float(np.real(
                pauli_expectations([make_observable_label(n, pauli, q)], vT, n)[0]))
            dev = abs(from_spo - exact)
            worst = max(worst, dev)
            print(f"    state {si}: SPO={from_spo:+.8f}  exact={exact:+.8f}  "
                  f"dev={dev:.2e}")

    print(f"\n  worst deviation over all states/observables: {worst:.3e}")
    if worst < args.tolerance:
        print("  PASS -- targets agree with exact evolution")
    else:
        print(f"  FAIL -- exceeds tolerance {args.tolerance:.0e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
