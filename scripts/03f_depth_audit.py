"""Is the HVA depth advantage over Trotter real, or a scheduling artefact?

``HVA.circuit_depth()`` returns an analytic ``5 * n_layers`` -- it assumes the
RZZ gates are scheduled by the 4-colouring. ``TrotterCircuit.circuit_depth()``
returns ``qc.depth()`` on qiskit's own gate order, which is not colour-sorted.
Comparing those two numbers gives our ansatz a free win, in the same way that
quoting qiskit's Suzuki synthesis (48 RZZ/step) against our grouped builder
(24 RZZ/step) did on the gate-count axis.

This script schedules *both* circuits with one ASAP scheduler, from the same
``substep_bonds`` colouring, and reports the depth each actually needs.

Run:  .venv/bin/python scripts/03f_depth_audit.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import numpy as np

from bppps.propagation import build_hva_gate_sequence, build_trotter_gate_sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', '4x4')


def merge_adjacent_1q(seq):
    """Fuse consecutive single-qubit rotations on the same qubit.

    Any transpiler does this, and S2 steps butt an RX(tau/2) against the next
    step's RX(tau/2). Not fusing them would inflate the Trotter depth.
    """
    out = []
    last_rx_slot = {}   # qubit -> index in out
    blocked = set()     # qubits touched by a 2Q gate since their last RX
    for g in seq:
        if g[0] == 'rx':
            q = g[1]
            if q in last_rx_slot and q not in blocked:
                i = last_rx_slot[q]
                merged = list(out[i])
                merged[2] = out[i][2] + g[2]
                out[i] = tuple(merged)
                continue
            out.append(g)
            last_rx_slot[q] = len(out) - 1
            blocked.discard(q)
        else:
            out.append(g)
            blocked.add(g[1])
            blocked.add(g[2])
    return out


def asap_depth(seq, n):
    """Parallel depth: every gate starts as early as its qubits allow."""
    ready = np.zeros(n, dtype=int)
    for g in seq:
        qs = (g[1],) if g[0] == 'rx' else (g[1], g[2])
        t = max(ready[q] for q in qs) + 1
        for q in qs:
            ready[q] = t
    return int(ready.max())


def asap_depth_2q(seq, n):
    """Same, counting only 2Q layers -- the channel that dominates error."""
    ready = np.zeros(n, dtype=int)
    for g in seq:
        if g[0] == 'rx':
            continue
        i, j = g[1], g[2]
        t = max(ready[i], ready[j]) + 1
        ready[i] = ready[j] = t
    return int(ready.max())


def main():
    with open(os.path.join(RES, 'model_config.json')) as f:
        mc = json.load(f)
    n = mc['num_qubits']
    bonds = [tuple(b) for b in mc['bonds']]
    J = np.array(mc['J'])
    h = mc['h']
    sb = {int(k): [tuple(x) for x in v] for k, v in mc['substep_bonds'].items()}

    print(f"{n} qubits, {len(bonds)} bonds, 4-colour substeps "
          f"{[len(sb[s]) for s in range(1, 5)]}\n")

    rows = []
    print(f"{'circuit':>18} {'unit':>5} {'2Q':>5} {'depth':>6} {'2Q depth':>9} "
          f"{'depth/2Q':>9}")
    print("-" * 60)

    rng = np.random.default_rng(0)
    for L in [1, 2, 3, 4, 6, 8, 12, 16]:
        seq = build_hva_gate_sequence(n, bonds, sb, L,
                                      rng.normal(size=L * (n + len(bonds))))
        seq = merge_adjacent_1q(seq)
        n2q = sum(1 for g in seq if g[0] == 'rzz')
        d, d2 = asap_depth(seq, n), asap_depth_2q(seq, n)
        rows.append(('hva', L, n2q, d, d2))
        print(f"{'HVA':>18} {L:5d} {n2q:5d} {d:6d} {d2:9d} {d / n2q:9.3f}")

    print()
    for steps in [1, 2, 3, 4, 6, 8, 12, 16]:
        seq = build_trotter_gate_sequence(n, sb, J, h, dt=0.5 / steps,
                                          n_steps=steps, order=2)
        seq = merge_adjacent_1q(seq)
        n2q = sum(1 for g in seq if g[0] == 'rzz')
        d, d2 = asap_depth(seq, n), asap_depth_2q(seq, n)
        rows.append(('trotter_s2', steps, n2q, d, d2))
        print(f"{'grouped S2':>18} {steps:5d} {n2q:5d} {d:6d} {d2:9d} "
              f"{d / n2q:9.3f}")

    print()
    for steps in [1, 2, 3, 4]:
        seq = build_trotter_gate_sequence(n, sb, J, h, dt=0.5 / steps,
                                          n_steps=steps, order=4)
        seq = merge_adjacent_1q(seq)
        n2q = sum(1 for g in seq if g[0] == 'rzz')
        d, d2 = asap_depth(seq, n), asap_depth_2q(seq, n)
        rows.append(('trotter_s4', steps, n2q, d, d2))
        print(f"{'grouped S4':>18} {steps:5d} {n2q:5d} {d:6d} {d2:9d} "
              f"{d / n2q:9.3f}")

    out = os.path.join(RES, 'depth_audit.json')
    with open(out, 'w') as f:
        json.dump({'num_qubits': n, 'num_bonds': len(bonds),
                   'rows': [{'circuit': c, 'unit': u, 'n_2q': q,
                             'depth': d, 'depth_2q': d2}
                            for c, u, q, d, d2 in rows]}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
