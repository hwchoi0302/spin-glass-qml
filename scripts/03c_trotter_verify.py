#!/usr/bin/env python3
"""Independent re-verification of scripts/03b_trotter_baseline.py's numbers.

03b used qiskit's TrotterCircuit (generic SuzukiTrotter synthesis over the
SparsePauliOp). That synthesis is *term-agnostic*: it symmetrises over every
summand separately, so each bond's RZZ is emitted twice per step (once in
each half of the order-2 product) even though all the ZZ terms mutually
commute and could share a single rotation. The repo's own
`build_trotter_gate_sequence` (src/bppps/propagation.py, used for stage 1
targets) instead splits H = H_X + H_ZZ into two commuting groups, giving
RX(half); RZZ(full); RX(half) -- 24 RZZ per step instead of 48.

That difference is worth a factor of two on the 2Q-gate axis, which is
exactly the axis the goal-1 claim is measured on. Quoting the wasteful
decomposition as "Trotter" would flatter our own ansatz for free. So this
script rebuilds the baseline with the *better* Trotter available and checks
several things that could each independently invalidate 03b:

  A. sign/convention: both builders must converge to ED as dt -> 0. A flipped
     J or h sign shows up here as a fidelity that never approaches 1.
  B. the two builders' 2Q counts, measured (not assumed).
  C. Suzuki order-2 convergence rate: global amplitude error ~ T*dt^2, so
     infidelity ~ dt^4. A fitted log-log slope far from 4 means one of the
     builders is not actually the order it claims.
  D. the corrected head-to-head, sweeping n_steps so both methods span the
     same 2Q range instead of being compared at one arbitrary point. S4 is
     included because it is the repo's default for target generation and is
     the strongest Trotter baseline available.

Writes the corrected curves back into results/4x4/statevector_pilot.json
under 'trotter_grouped_s2' / 'trotter_grouped_s4', leaving 03b's original
'trotter' (qiskit generic) section in place for comparison.
"""

import importlib.util
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from ansatz.trotter import TrotterCircuit                              # noqa: E402
from bppps.propagation import build_trotter_gate_sequence              # noqa: E402
from classical_bench import ExactDiag                                   # noqa: E402
from config import load_config, output_dir                              # noqa: E402
from hamiltonians import SpinGlass2D                                     # noqa: E402
from hamiltonians.spin_glass_2d import classify_substep_bonds            # noqa: E402
from qiskit.quantum_info import Statevector                              # noqa: E402

# The pilot's own statevector primitives, reused so that any error in them
# would show up identically in both the HVA and the Trotter numbers rather
# than biasing the comparison one way.
_spec = importlib.util.spec_from_file_location(
    'pilot', os.path.join(PROJECT_ROOT, 'scripts', '03_statevector_pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def evolve_sequence(psi, seq, n, pat_map):
    """Apply a repo gate sequence in circuit order (list[0] applied first)."""
    for g in seq:
        if g[0] == 'rx':
            psi = pilot.apply_rx(psi, g[1], g[2], n)
        elif g[0] == 'rzz':
            _, i, j, theta, _ = g
            psi = pilot.apply_rzz(psi, theta, pat_map[(i, j)])
    return psi


def n_2q(seq):
    return sum(1 for g in seq if g[0] == 'rzz')


def main():
    config = load_config()
    out_dir = output_dir(config, create=False)
    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))

    n = model.num_qubits
    H = model.build_sparse_matrix()
    ed = ExactDiag(H, n)
    qk = TrotterCircuit(model.get_pauli_terms(), n)
    substep = classify_substep_bonds(model.bonds, model.Lx)
    pat_map = {(i, j): pilot.zz_pattern(i, j, n) for (i, j) in model.bonds}

    psi0 = np.array(Statevector.from_label('0' * n))

    def fid_grouped(T, n_steps, order):
        seq = build_trotter_gate_sequence(n, substep, model.J, model.h,
                                          T / n_steps, n_steps, order)
        psi = evolve_sequence(np.array(psi0, dtype=complex), seq, n, pat_map)
        exact = ed.time_evolve(np.array(psi0, dtype=complex), T)
        return float(np.abs(np.vdot(exact, psi)) ** 2), n_2q(seq)

    def fid_qiskit(T, n_steps, order):
        qc = qk.build_circuit(T, T / n_steps, order=order)
        psi = np.array(Statevector(psi0).evolve(qc))
        exact = ed.time_evolve(np.array(psi0, dtype=complex), T)
        ops = qc.count_ops()
        return (float(np.abs(np.vdot(exact, psi)) ** 2),
                ops.get('rzz', 0) + ops.get('cx', 0) + ops.get('cz', 0))

    print("=" * 72)
    print("  TEST A -- convergence to ED as dt -> 0 (catches sign/convention errors)")
    print("=" * 72)
    T = 0.5
    for steps in (1, 4, 16, 64, 256):
        fg, gg = fid_grouped(T, steps, 2)
        fq, gq = fid_qiskit(T, steps, 2)
        print(f"  T={T} steps={steps:4d} | grouped S2 F={fg:.10f} (2Q={gg:5d}) "
              f"| qiskit S2 F={fq:.10f} (2Q={gq:5d})")
    fg, _ = fid_grouped(T, 256, 2)
    fq, _ = fid_qiskit(T, 256, 2)
    assert fg > 0.999999 and fq > 0.999999, \
        "a builder does not converge to ED -- sign/convention error"
    print("  PASS: both builders converge to ED\n")

    print("=" * 72)
    print("  TEST B -- 2Q gates per step, measured")
    print("=" * 72)
    _, g4 = fid_grouped(1.0, 4, 2)
    _, q4 = fid_qiskit(1.0, 4, 2)
    _, g4s4 = fid_grouped(1.0, 4, 4)
    print(f"  grouped S2 : {g4 // 4:3d} RZZ/step  (n_bonds={model.num_bonds})")
    print(f"  qiskit  S2 : {q4 // 4:3d} RZZ/step")
    print(f"  grouped S4 : {g4s4 // 4:3d} RZZ/step")
    print(f"  -> qiskit's generic synthesis costs "
          f"{q4 / g4:.1f}x the 2Q gates of the grouped split\n")

    print("=" * 72)
    print("  TEST C -- order-2 convergence rate (expect infidelity ~ dt^4)")
    print("=" * 72)
    steps_list = [8, 16, 32, 64]
    infids, dts = [], []
    for steps in steps_list:
        f, _ = fid_grouped(1.0, steps, 2)
        infids.append(1.0 - f)
        dts.append(1.0 / steps)
    slope = np.polyfit(np.log(dts), np.log(infids), 1)[0]
    for dt, inf in zip(dts, infids):
        print(f"  dt={dt:.5f}  infidelity={inf:.3e}")
    print(f"  fitted log-log slope = {slope:.2f} (expected ~4.0)")
    assert 3.5 < slope < 4.5, f"grouped S2 is not order 2 (slope {slope})"
    print("  PASS\n")

    print("=" * 72)
    print("  TEST D -- corrected head-to-head vs the HVA ceiling")
    print("=" * 72)
    with open(os.path.join(out_dir, 'statevector_pilot.json')) as f:
        out = json.load(f)
    te = out['time_evolution']
    sec2 = out.setdefault('trotter_grouped_s2', {})
    sec4 = out.setdefault('trotter_grouped_s4', {})

    for T in sorted(te.keys(), key=float):
        Tf = float(T)
        print(f"\n  --- T = {T} ---")
        hva = te[T]
        for L in sorted(hva.keys(), key=int):
            print(f"    HVA         L={int(L):2d}  2Q={24 * int(L):4d}  "
                  f"F={hva[L]['fidelity']:.8f}  infid={1 - hva[L]['fidelity']:.3e}")
        by2, by4 = sec2.setdefault(T, {}), sec4.setdefault(T, {})
        for steps in (1, 2, 3, 4, 6, 8, 12, 16):
            f2, g2 = fid_grouped(Tf, steps, 2)
            by2[str(steps)] = {'fidelity': f2, 'n_2q': g2, 'n_steps': steps}
            print(f"    grouped S2  s={steps:2d}  2Q={g2:4d}  F={f2:.8f}  "
                  f"infid={1 - f2:.3e}")
        for steps in (1, 2, 3, 4):
            f4, g4_ = fid_grouped(Tf, steps, 4)
            by4[str(steps)] = {'fidelity': f4, 'n_2q': g4_, 'n_steps': steps}
            print(f"    grouped S4  s={steps:2d}  2Q={g4_:4d}  F={f4:.8f}  "
                  f"infid={1 - f4:.3e}")

    path = os.path.join(out_dir, 'statevector_pilot.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
