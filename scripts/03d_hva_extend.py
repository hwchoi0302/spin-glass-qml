#!/usr/bin/env python3
"""Extend the HVA ceiling to deeper circuits at large T.

03c's corrected head-to-head left one real gap: the HVA ceiling was only ever
measured out to L=8 (192 2Q gates), because 03_statevector_pilot.py stops at
config's `direct_layers`. Grouped Trotter S2 keeps improving past that point
(at T=4.0 it reaches infidelity 6.5e-2 by 384 gates), so at T=4.0 there is
currently no HVA number at a comparable gate count and the comparison there
is simply unresolved -- not a win and not a loss.

This fills in L=12 and L=16 (288 and 384 2Q gates) at T=2.0 and T=4.0 so both
methods span the same budget. T=4.0 L=8 is also re-run with a much larger
time budget, because the original 300s cap makes its 0.4148 a lower bound on
the ceiling rather than the ceiling itself -- if that number moves a lot, the
original run was optimiser-limited rather than ansatz-limited, which would
matter for how the T=4.0 row is read.

Merges into results/4x4/statevector_pilot.json's existing time_evolution
section, so the plot script picks the new points up with no changes.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from bppps.warm_start import trotter_warm_start                        # noqa: E402
from classical_bench import ExactDiag                                   # noqa: E402
from config import load_config, output_dir                              # noqa: E402
from hamiltonians import SpinGlass2D                                     # noqa: E402
from qiskit.quantum_info import Statevector                              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'pilot', os.path.join(PROJECT_ROOT, 'scripts', '03_statevector_pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

# Two gaps, both artefacts of how 03_statevector_pilot.py sampled rather than
# of the physics:
#
#  - Small T has almost no points. time_evolution_pilot() `break`s as soon as
#    a layer count clears `accuracy_target` (0.99), so T=0.5 stopped after
#    L=2 (F=0.99966) with a single point and T=1.0 after L=3. That was fine
#    for "how deep must it be", but it makes the fidelity-vs-2Q curve look
#    like deeper circuits were never worth measuring at small T. They are:
#    the comparison against Trotter runs out to 384+ gates, so HVA needs
#    points there too or the curves simply do not overlap.
#  - Large T stops at L=8 (config's deepest `direct_layers`), while grouped
#    Trotter keeps improving past 192 gates.
#
# Ordered cheapest-first so the most informative points land early.
PAIRS = [
    (0.5, 3), (0.5, 4), (0.5, 6), (0.5, 8), (0.5, 12), (0.5, 16),
    (1.0, 4), (1.0, 6), (1.0, 8), (1.0, 12), (1.0, 16),
    (2.0, 12), (2.0, 16),
    (4.0, 8), (4.0, 12), (4.0, 16),
]
MAX_SECONDS = 1800.0


def main():
    config = load_config()
    out_dir = output_dir(config, create=False)
    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))

    n = model.num_qubits
    patterns = [pilot.zz_pattern(i, j, n) for (i, j) in model.bonds]
    plan_for = pilot.build_plan(n, model.bonds)
    ed = ExactDiag(model.build_sparse_matrix(), n)
    psi0 = np.array(Statevector.from_label('0' * n))

    path = os.path.join(out_dir, 'statevector_pilot.json')

    for T, L in PAIRS:
        psi_exact = ed.time_evolve(np.array(psi0, dtype=complex), T)
        plan = plan_for(L)
        params_init = trotter_warm_start(n, model.bonds, model.J, model.h, T, L)
        t0 = time.time()
        x, elapsed = pilot.optimise(
            lambda th: pilot.fidelity_cost_grad(th, plan, psi0, psi_exact, n, patterns),
            params_init, MAX_SECONDS, maxiter=3000)
        fid = 1.0 - pilot.fidelity_cost_grad(x, plan, psi0, psi_exact, n, patterns)[0]
        print(f"T={T} L={L:2d} 2Q={24 * L:4d} fidelity={fid:.8f} "
              f"infid={1 - fid:.3e} ({elapsed:.0f}s)", flush=True)

        # Re-read/re-write each time: 03c and this script both touch the file,
        # and the retrains running alongside make a lost update easy.
        with open(path) as f:
            out = json.load(f)
        out['time_evolution'].setdefault(str(T), {})[str(L)] = {
            'fidelity': fid, 'n_params': len(plan), 'time_s': elapsed,
            'max_seconds': MAX_SECONDS,
        }
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)

    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
