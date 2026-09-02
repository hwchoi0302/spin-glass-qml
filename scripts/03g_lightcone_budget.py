"""How much of the circuit can actually corrupt a local observable?

The hardware budget tables in docs/issues/ all use ``exp(-n_2Q * eps)``, the
probability that *no gate anywhere in the circuit* errs. For a claim about the
global state that is the right quantity. For a local observable it is far too
pessimistic: a depolarising error on a gate outside the backward light cone of
O commutes past O and leaves <O> untouched.

This script measures the cone exactly -- walk the gate sequence backwards from
a single-site observable, keep the support, and count only the 2Q gates that
touch it -- for the composed circuit U(theta; 0.5)^k on 4x4 and 10x10.

Run:  .venv/bin/python scripts/03g_lightcone_budget.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import numpy as np

from hamiltonians.spin_glass_2d import classify_substep_bonds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', '4x4')


def lattice(Lx, Ly):
    bonds = []
    for y in range(Ly):
        for x in range(Lx - 1):
            bonds.append((y * Lx + x, y * Lx + x + 1))
    for x in range(Lx):
        for y in range(Ly - 1):
            bonds.append((y * Lx + x, (y + 1) * Lx + x))
    return bonds, classify_substep_bonds(bonds, Lx)


def hva_sequence(sb, n_layers, k):
    """2Q gates of U(theta; dt)^k, in circuit order. Angles are irrelevant here."""
    seq = []
    for _ in range(k):
        for _ in range(n_layers):
            for s in range(1, 5):
                for _, i, j in sb[s]:
                    seq.append((i, j))
    return seq


def cone(seq, site):
    """2Q gates inside the backward light cone of a single-site observable.

    Heisenberg picture: the observable sits at the circuit output, so walk the
    sequence in reverse. A gate matters only if it touches the current support;
    when it does, it drags its partner into the support too.
    """
    support = {site}
    n_in = 0
    for i, j in reversed(seq):
        if i in support or j in support:
            n_in += 1
            support.add(i)
            support.add(j)
    return n_in, len(support)


def main():
    rows = []
    hdr = (f"{'lattice':>8} {'L':>2} {'k':>3} {'t':>5} {'2Q tot':>7} "
           f"{'cone':>6} {'cone%':>7} {'supp':>5}")
    for eps in (3e-3, 1e-3):
        hdr += " | {:>10} {:>7} {:>9}".format('Fg@%g' % eps, 'Floc', 'shots@10%')
    print(hdr)
    print("-" * len(hdr))

    for Lx, Ly in [(4, 4), (10, 10)]:
        bonds, sb = lattice(Lx, Ly)
        centre = (Ly // 2) * Lx + (Lx // 2)
        for n_layers in (2, 3):
            for k in (1, 2, 4, 8):
                seq = hva_sequence(sb, n_layers, k)
                n_tot = len(seq)
                n_cone, n_supp = cone(seq, centre)
                line = (f"{Lx}x{Ly:<5d} {n_layers:2d} {k:3d} {0.5 * k:5.1f} "
                        f"{n_tot:7d} {n_cone:6d} {n_cone / n_tot:6.1%} "
                        f"{n_supp:5d}")
                row = {'Lx': Lx, 'Ly': Ly, 'n_layers': n_layers, 'k': k,
                       't': 0.5 * k, 'n_2q_total': n_tot, 'n_2q_cone': n_cone,
                       'support': n_supp, 'n_qubits': Lx * Ly}
                for eps in (3e-3, 1e-3):
                    f_glob = float(np.exp(-n_tot * eps))
                    f_loc = float(np.exp(-n_cone * eps))
                    # Linear XEB: Fhat = (2^n/M) sum p_ideal(x_i) - 1. For a
                    # Porter-Thomas ideal distribution 2^n p has unit mean and
                    # unit variance, so Var(Fhat) ~ 1/M and SE = 1/sqrt(M).
                    # SNR 1 needs M ~ 1/F^2; 10% relative precision needs 100x
                    # that. The 10% column is the operationally useful one.
                    shots = 100.0 / f_glob ** 2 if f_glob > 0 else float('inf')
                    line += f" | {f_glob:10.4f} {f_loc:7.3f} {shots:9.2g}"
                    row[f'eps_{eps:g}'] = {
                        'F_global': f_glob, 'F_local': f_loc,
                        'xeb_shots_snr1': 1.0 / f_glob ** 2 if f_glob > 0 else None,
                        'xeb_shots_10pct': shots}
                print(line)
                rows.append(row)
        print()

    out = os.path.join(RES, 'lightcone_budget.json')
    with open(out, 'w') as f:
        json.dump({'note': 'cone = 2Q gates in the backward light cone of one '
                           'centre-site observable through U(theta;0.5)^k',
                   'rows': rows}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
