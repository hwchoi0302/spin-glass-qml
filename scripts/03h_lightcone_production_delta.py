#!/usr/bin/env python3
"""Re-measure the light-cone saturation table at production truncation.

docs/issues/01-scale-plan.md records that at fixed T the term count saturates
with lattice size (3x3 9,945 -> 7x7 58,593, flat from 5x5 on), which is what
makes T2 affordable: 7x7 costs ~2x per observable against 4x4 rather than
exploding the way T did. That table was taken on the laptop at delta=1e-5,
and the doc flags the gap explicitly -- production is delta=1e-8, a thousand
times tighter, and smaller coefficients surviving longer could push saturation
out to a larger lattice. The qualitative claim rests on the light cone and is
safe; the *ratio* is what T2's cost estimate uses, and that is what needs
re-checking before committing to 7x7.

Run on the desktop: at 1e-8 the corner observable alone reached 1.2M terms in
the 4x4 T1-T run, so this is not a laptop job.

Guards, because delta=1e-8 on a 7x7 is exactly the regime that filled memory
during T1-T stage 1: MemAvailable is checked between lattices and the sweep
stops before starting one that looks unsafe, and --max-terms bails out of a
size that has clearly run away. Partial results are written after every
lattice, so an abort still leaves the ratios measured so far.

Usage:
    python scripts/03h_lightcone_production_delta.py
    python scripts/03h_lightcone_production_delta.py --sizes 3 4 5 --delta 1e-8
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
    TruncationStats, build_trotter_gate_sequence,
)
from bppps.propagation_numba import (                                  # noqa: E402
    empty_dict, from_numba_dict, make_key, propagate_forward_numba,
)
from bppps.propagation_packed import label_to_xz                        # noqa: E402
from hamiltonians import SpinGlass2D                                     # noqa: E402
from hamiltonians.spin_glass_2d import classify_substep_bonds            # noqa: E402


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 ** 2
    return float('inf')


def centre_qubit(L):
    return (L // 2) * L + (L // 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[3, 4, 5, 6, 7])
    parser.add_argument('--delta', type=float, default=1e-8)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--order', type=int, default=4)
    parser.add_argument('--max-terms', type=int, default=40_000_000)
    parser.add_argument('--mem-floor-gb', type=float, default=3.0)
    args = parser.parse_args()

    out_path = os.path.join(PROJECT_ROOT, 'results', 'lightcone_production_delta.json')
    rows = []
    n_steps = int(round(args.T / args.dt))
    print(f"  T={args.T}, dt={args.dt}, S{args.order}, delta={args.delta:.0e}, "
          f"{n_steps} Trotter steps")
    print(f"  guards: max_terms={args.max_terms:,}, mem_floor={args.mem_floor_gb} GB\n")
    print(f"  {'lattice':>8} {'qubits':>7} {'centre X':>12} {'ratio':>7} "
          f"{'corner X':>12} {'eps_emp':>10} {'s':>7} {'memGB':>7}")
    print("  " + "-" * 82)

    for L in args.sizes:
        avail = mem_available_gb()
        if avail < args.mem_floor_gb:
            print(f"  MEMORY FLOOR: {avail:.2f} GB < {args.mem_floor_gb} GB, "
                  f"stopping before {L}x{L}")
            break

        model = SpinGlass2D(L, L, h=1.0, coupling_type='ea_bimodal', seed=42)
        substep = classify_substep_bonds(model.bonds, model.Lx)
        seq = build_trotter_gate_sequence(model.num_qubits, substep, model.J,
                                          model.h, dt=args.dt, n_steps=n_steps,
                                          order=args.order)
        counts = {}
        t0 = time.time()
        eps = None
        for tag, q in (('centre', centre_qubit(L)), ('corner', 0)):
            label = make_observable_label(model.num_qubits, 'X', q)
            x, z = label_to_xz(label)
            init = empty_dict()
            init[make_key(x, z)] = 1.0
            stats = TruncationStats()
            coeffs = propagate_forward_numba(init, seq, args.delta, stats)
            counts[tag] = len(coeffs)
            if tag == 'centre':
                eps = stats.error_estimate
            del coeffs, init
        elapsed = time.time() - t0

        prev = rows[-1]['centre_terms'] if rows else None
        ratio = counts['centre'] / prev if prev else float('nan')
        rows.append({'L': L, 'num_qubits': model.num_qubits,
                     'centre_terms': counts['centre'],
                     'corner_terms': counts['corner'],
                     'ratio_vs_prev': ratio, 'eps_emp_centre': eps,
                     'seconds': elapsed, 'delta': args.delta, 'dt': args.dt,
                     'T': args.T, 'trotter_order': args.order})
        print(f"  {L}x{L:<5} {model.num_qubits:7d} {counts['centre']:12,} "
              f"{ratio:7.2f} {counts['corner']:12,} {eps:10.2e} "
              f"{elapsed:7.0f} {mem_available_gb():7.2f}")

        with open(out_path, 'w') as f:
            json.dump({'rows': rows, 'args': vars(args)}, f, indent=2)

        if counts['centre'] > args.max_terms:
            print(f"  MAX TERMS exceeded at {L}x{L}; stopping (partial results kept)")
            break

    print(f"\n  Saved: {out_path}")
    if len(rows) >= 3:
        print("  Saturation is the claim being tested: the ratio column should "
              "fall toward 1.0.")


if __name__ == '__main__':
    main()
