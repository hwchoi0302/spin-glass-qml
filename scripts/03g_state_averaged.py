"""Is the HVA advantage over Trotter an artefact of scoring on |0...0>?

Every goal-1 number in the repo is the fidelity of U|0...0> against the exact
evolution of that one state. That choice is not neutral:

  * the statevector pilot *optimised* that exact quantity, so it is scored on
    its own objective and looks as good as it possibly can;
  * BP-PPS optimises L_XZ, a distance between operators, which is an
    infinite-temperature average over all initial states -- it never sees
    |0...0> and is scored on something it did not train for;
  * Trotter optimises nothing and is state-independent by construction.

So the |0...0> comparison flatters the pilot and penalises BP-PPS, and the
124x gap between them may be mostly metric mismatch rather than a training
deficiency. The claim in CLAUDE.md is time evolution "from an arbitrary
product state", which is the averaged quantity, not the |0...0> one.

This measures the average over Haar-random single-qubit product states of
|<psi_exact(T)|U|psi_0>|^2 for the trained BP-PPS blocks and for grouped
second-order Trotter at matched 2Q gate counts. Same states for every circuit,
so the comparison is paired.

The BP-PPS block only exists at dt = 0.5, so any T here must be a whole number
of blocks and the block is composed that many times. --T 1.0 scores
U(theta;0.5)^2 at 2*24*L gates, not one block at half the time.

Usage:
    python scripts/03g_state_averaged.py
    python scripts/03g_state_averaged.py --n-states 40 --T 0.5
    python scripts/03g_state_averaged.py --T 1.0 --out results/4x4/state_averaged_T1.0.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hamiltonians.spin_glass_2d import SpinGlass2D          # noqa: E402
from classical_bench.exact_diag import ExactDiag            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'pilot', os.path.join(ROOT, 'scripts', '03_statevector_pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def random_product_states(n, count, seed):
    """Haar-random single-qubit states, tensored. Shape (count, 2**n)."""
    rng = np.random.default_rng(seed)
    out = np.empty((count, 1 << n), dtype=np.complex128)
    for s in range(count):
        psi = np.ones(1, dtype=np.complex128)
        for _ in range(n):
            # Haar on the Bloch sphere: cos(theta/2), e^{i phi} sin(theta/2)
            u, phi = rng.random(), rng.uniform(0, 2 * np.pi)
            theta = 2 * np.arccos(np.sqrt(u))
            qubit = np.array([np.cos(theta / 2),
                              np.exp(1j * phi) * np.sin(theta / 2)])
            psi = np.kron(qubit, psi)          # little-endian: new qubit is high
        out[s] = psi
    return out


def apply_hva(psi, theta, plan, n, patterns):
    for g in plan:
        if g[0] == 'rx':
            psi = pilot.apply_rx(psi, g[1], theta[g[2]], n)
        else:
            _, i, j, pidx, bidx = g
            psi = pilot.apply_rzz(psi, theta[pidx], patterns[bidx])
    return psi


def apply_grouped_s2(psi, T, steps, n, bonds, J, h, patterns):
    """M steps of RX(dt/2) RZZ(dt) RX(dt/2), merging adjacent half-layers.

    H = H_X + H_ZZ with H_X = -h sum X and H_ZZ = -sum J ZZ, both internally
    commuting, so this is the standard second-order splitting and costs
    |bonds| two-qubit gates per step -- not the 2x that a generic Suzuki
    synthesis emits (see docs/issues/01-scale-plan.md).
    """
    dt = T / steps
    for m in range(steps):
        frac = 0.5 if m == 0 else 1.0        # merged half-layers in the middle
        if m == 0:
            for q in range(n):
                psi = pilot.apply_rx(psi, q, -2.0 * h * dt * frac, n)
        for b, _ in enumerate(bonds):
            psi = pilot.apply_rzz(psi, -2.0 * J[b] * dt, patterns[b])
        last = (m == steps - 1)
        for q in range(n):
            psi = pilot.apply_rx(psi, q, -2.0 * h * dt * (0.5 if last else 1.0), n)
    return psi


def apply_grouped_s4(psi, T, steps, n, bonds, J, h, patterns):
    """Suzuki's 4th-order recursion built from grouped S2 steps.

    S4(dt) = S2(p dt)^2 S2((1-4p) dt) S2(p dt)^2 with p = 1/(4 - 4^(1/3)), so
    one S4 step costs five S2 steps -- 5*|bonds| two-qubit gates. That is the
    price of the higher order and it is why S4 only pays off when the accuracy
    target is tight enough for its steeper convergence to overcome the 5x
    per-step cost.
    """
    w = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    sub = [w, w, 1.0 - 4.0 * w, w, w]
    dt = T / steps
    for _ in range(steps):
        for c in sub:
            psi = apply_grouped_s2(psi, c * dt, 1, n, bonds, J, h, patterns)
    return psi


# The BP-PPS time-evolution block is trained at this chunk size.
BLOCK_DT = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-states', type=int, default=32)
    ap.add_argument('--T', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, 'results/4x4/model_config.json')))
    model = SpinGlass2D.from_config_dict(cfg)
    n, T = model.num_qubits, args.T
    patterns = [pilot.zz_pattern(i, j, n) for (i, j) in model.bonds]
    plan_for = pilot.build_plan(n, model.bonds)
    ed = ExactDiag(model.build_sparse_matrix(), n)

    states = random_product_states(n, args.n_states, args.seed)
    zero = np.zeros(1 << n, dtype=np.complex128); zero[0] = 1.0
    exact = [ed.time_evolve(s, T) for s in states]
    exact_zero = ed.time_evolve(zero, T)

    def score(apply_fn, label, n_2q):
        f_avg = float(np.mean([abs(np.vdot(e, apply_fn(s))) ** 2
                               for s, e in zip(states, exact)]))
        f_zero = float(abs(np.vdot(exact_zero, apply_fn(zero))) ** 2)
        print(f"  {label:<26} {n_2q:5d} {1 - f_zero:12.3e} {1 - f_avg:12.3e} "
              f"{(1 - f_avg) / (1 - f_zero):8.2f}")
        return {'label': label, 'n_2q': n_2q, 'infid_zero': 1 - f_zero,
                'infid_avg': 1 - f_avg}

    print(f"4x4, T={T}, {args.n_states} Haar-random product states (seed {args.seed})\n")
    print(f"  {'circuit':<26} {'2Q':>5} {'infid |0..0>':>12} "
          f"{'infid 평균':>12} {'비율':>8}")
    print("  " + "-" * 68)

    # The BP-PPS block is trained at dt = 0.5 only; longer times are reached by
    # composing it, U(theta; 0.5)^k. Applying it once at T = 1.0 would score a
    # half-time circuit against a full-time target and report a spurious
    # failure, so k is derived from T here rather than assumed to be 1.
    if abs(T / BLOCK_DT - round(T / BLOCK_DT)) > 1e-9:
        raise SystemExit(
            f"T={T} is not a whole number of {BLOCK_DT} blocks; the BP-PPS "
            f"circuit only exists at multiples of the trained chunk.")
    k = int(round(T / BLOCK_DT))

    rows = []
    for fname, L in (('trained_params.json', 3), ('te_trained_params_L2.json', 2)):
        path = os.path.join(ROOT, 'results/4x4', fname)
        if not os.path.exists(path):
            continue
        rec = json.load(open(path))
        th = np.asarray(rec.get('params', rec.get('optimized_params')))
        plan = plan_for(L)

        def composed(psi, th=th, pl=plan, k=k):
            for _ in range(k):
                psi = apply_hva(psi, th, pl, n, patterns)
            return psi

        lab = f"BP-PPS HVA L={L}" if k == 1 else f"BP-PPS HVA L={L}, k={k}"
        rows.append(score(composed, lab, 24 * L * k))

    for steps in (2, 3, 4, 6, 8, 12, 16):
        rows.append(score(
            lambda p, m=steps: apply_grouped_s2(p, T, m, n, model.bonds,
                                                model.J, model.h, patterns),
            f"grouped S2, {steps} steps", 24 * steps))

    for steps in (1, 2, 3, 4, 6):
        rows.append(score(
            lambda p, m=steps: apply_grouped_s4(p, T, m, n, model.bonds,
                                                model.J, model.h, patterns),
            f"grouped S4, {steps} steps", 5 * 24 * steps))

    out = args.out or os.path.join(ROOT, 'results/4x4', 'state_averaged.json')
    with open(out, 'w') as f:
        json.dump({'T': T, 'n_states': args.n_states, 'seed': args.seed,
                   'rows': rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
