#!/usr/bin/env python3
"""Spot-check a cached targets_dt<T>.json against a freshly generated observable.

Target generation is the expensive stage (hours), so the file is cached and
gitignored. Before committing to a long training run on a cache produced by an
older revision, regenerate one or two observables and compare.

Why this is normally enough: the S2 and S4 Trotter sequences are palindromic,
so the gate-ordering fix of 2026-08-28 left the targets bit-identical — see
``scripts/00_validate_small.py`` TEST 14, which proves the invariance
structurally. This script is the empirical confirmation on the actual file.

The corner observables (X_0, Z_0) are the cheapest: they have the fewest
neighbours, so their operators stay smallest under propagation.

Usage:
    python scripts/00_verify_targets.py
    python scripts/00_verify_targets.py --observables X_0 Z_0 X_5
    python scripts/00_verify_targets.py --tolerance 1e-9
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from bppps.ose_regularizer import operator_norm_sq             # noqa: E402
from bppps.pauli_utils import make_observable_label            # noqa: E402
from bppps.propagation import (                                # noqa: E402
    TruncationStats, build_trotter_gate_sequence, propagate_forward,
)
from config import apply_overrides, load_config, output_dir    # noqa: E402
from hamiltonians import SpinGlass2D                           # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='section.key=value')
    parser.add_argument('--observables', nargs='+', default=['X_0'],
                        help="Observables to regenerate, e.g. X_0 Z_0")
    parser.add_argument('--tolerance', type=float, default=1e-10,
                        help='Max allowed coefficient deviation')
    args = parser.parse_args()

    config = apply_overrides(load_config(), args.overrides)
    out_dir = output_dir(config, create=False)
    tgt = config['target']

    targets_path = os.path.join(out_dir, f"targets_dt{tgt['delta_t']}.json")
    if not os.path.exists(targets_path):
        raise SystemExit(
            f"No cache at {targets_path}.\n"
            f"Generate it with: python scripts/run_pipeline.py --stages 1"
        )

    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))

    size_mb = os.path.getsize(targets_path) / 1024 ** 2
    print(f"  cache   : {targets_path} ({size_mb:.1f} MB)")
    t0 = time.time()
    with open(targets_path) as f:
        cached = json.load(f)
    print(f"  loaded  : {len(cached)} observables, "
          f"{sum(len(v) for v in cached.values())} terms, "
          f"{time.time() - t0:.1f}s")

    # An exactly propagated single-site Pauli keeps ||O||_rHS = 1; the shortfall
    # is what truncation removed, so it is a free global health check.
    print("\n  norm check (||O|| = 1 exactly; deficit = truncation loss)")
    worst = None
    for key in sorted(cached, key=lambda k: (k[0], int(k.split('_')[1]))):
        deficit = 1.0 - operator_norm_sq(cached[key]) ** 0.5
        if worst is None or deficit > worst[1]:
            worst = (key, deficit)
    print(f"    largest norm deficit: {worst[0]} -> {worst[1]:.3e}")
    if worst[1] > 1e-3:
        print("    WARNING: that is large for a cutoff of "
              f"{tgt['cutoff']:.0e}; the cache may have been built with a "
              "looser threshold.")

    # Build the precision Trotter sequence once and propagate only the
    # requested observables through it; TargetGenerator would do all 32.
    n_steps = int(round(tgt['delta_t'] / tgt['dt']))
    gate_seq = build_trotter_gate_sequence(
        model.num_qubits, model.substep_bonds, model.J, model.h,
        dt=tgt['dt'], n_steps=n_steps, order=tgt['trotter_order'],
    )

    print(f"\n  regenerating {args.observables} "
          f"(S{tgt['trotter_order']}, dt={tgt['dt']}, "
          f"{len(gate_seq)} gates, cutoff={tgt['cutoff']})")
    failures = 0
    for key in args.observables:
        if key not in cached:
            print(f"    {key}: NOT IN CACHE")
            failures += 1
            continue
        pauli, q_str = key.split('_')
        label = make_observable_label(model.num_qubits, pauli, int(q_str))
        stats = TruncationStats()
        t0 = time.time()
        spo = propagate_forward({label: 1.0}, gate_seq, tgt['cutoff'], stats)
        ref = cached[key]
        max_dev = max(
            (abs(spo.get(P, 0.0) - ref.get(P, 0.0))
             for P in set(spo) | set(ref)), default=0.0)
        ok = max_dev <= args.tolerance
        failures += not ok
        print(f"    {key}: {len(spo)} vs {len(ref)} terms, "
              f"max deviation {max_dev:.3e}, "
              f"eps_trunc {stats.error_estimate:.3e}, "
              f"{time.time() - t0:.1f}s  {'OK' if ok else 'MISMATCH'}")

    print()
    if failures:
        raise SystemExit(
            f"  {failures} observable(s) did not match. Regenerate the cache:\n"
            f"    rm {targets_path}\n"
            f"    python scripts/run_pipeline.py --stages 1"
        )
    print("  Cache is consistent with the current code. Safe to train on.")


if __name__ == '__main__':
    main()
