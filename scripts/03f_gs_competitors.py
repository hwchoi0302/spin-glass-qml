"""Goal 3 under FTQC assumptions: HVA vs adiabatic Trotter vs QAOA.

Hardware error is assumed away, so the only currency is *algorithmic*: how many
2-qubit gates does each route need to reach a given ground-state energy? All
three routes below emit exactly the same gate alphabet on the same lattice --
a layer of RX on every qubit and a layer of RZZ on every bond -- so a layer is
24 two-qubit gates at 4x4 regardless of route, and comparing at equal layer
count is comparing at equal 2Q gate count. That is what makes the comparison
fair; the routes differ only in how the angles are chosen.

    HVA        L layers, L*(n+|bonds|) free angles, energy minimised
    QAOA       p layers, 2p free angles (all RX in a layer share beta, all RZZ
               share gamma scaled by its own J_b), energy minimised
    adiabatic  M Trotter steps of H(s) = (1-s)(-Gamma sum X) + s H, angles
               fixed by the schedule, nothing optimised

QAOA is a strict sub-family of HVA (the same circuit with angles tied), so
HVA >= QAOA holds by construction and is not a result. It is measured anyway
because the interesting question is the *price* of the tying: QAOA has 40x
fewer parameters at 4x4, which is what decides whether it can be trained at
100 qubits at all.

All three start from |+...+>, which is the ground state of the driver and, by
Perron-Frobenius on this stoquastic H, lies in the same parity sector as the
true ground state (see docs/issues/01-scale-plan.md).

Usage:
    python scripts/03f_gs_competitors.py                    # 4x4
    python scripts/03f_gs_competitors.py --lattice 3        # 3x3, quick
    python scripts/03f_gs_competitors.py --max-seconds 60
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hamiltonians.spin_glass_2d import SpinGlass2D          # noqa: E402
from classical_bench.exact_diag import ExactDiag            # noqa: E402

# The pilot module's name starts with a digit, so it cannot be imported by name.
_spec = importlib.util.spec_from_file_location(
    'pilot', os.path.join(ROOT, 'scripts', '03_statevector_pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


def plus_state(n):
    """|+...+>, the driver ground state and the +1 eigenstate of prod_i X_i."""
    return np.full(1 << n, 2.0 ** (-n / 2.0), dtype=np.complex128)


def evolve_only(theta, plan, psi0, n, patterns):
    """Final state only, discarding intermediates.

    pilot.forward_pass keeps every intermediate state because the adjoint
    gradient needs them, which is 40*L states per call. The adiabatic sweep
    optimises nothing and so needs no gradient, and at M=100 on 4x4 the stored
    tape would be 4 GB. This keeps two arrays live instead.
    """
    psi = psi0
    for g in plan:
        if g[0] == 'rx':
            psi = pilot.apply_rx(psi, g[1], theta[g[2]], n)
        else:
            _, i, j, pidx, bidx = g
            psi = pilot.apply_rzz(psi, theta[pidx], patterns[bidx])
    return psi


def energy_of(psi, bonds, J, h, n, patterns):
    """<psi|H|psi>."""
    return float(np.real(np.vdot(psi, pilot.apply_H(psi, bonds, J, h, n,
                                                    patterns))))


# ============================================================================
# Angle builders. Each returns a full HVA-layout parameter vector, so all three
# routes run through the *same* forward pass and are guaranteed to be compared
# on identical circuit machinery.
# ============================================================================

def qaoa_angles(gamma_beta, n_layers, n, J, n_bonds):
    """Expand 2p QAOA angles into the L*(n+|bonds|) HVA layout.

    Layer l applies exp(-i beta_l sum_q X_q) then exp(+i gamma_l sum_b J_b Z Z),
    i.e. every RX in the layer shares one angle and every RZZ is that layer's
    gamma scaled by its own coupling.
    """
    theta = np.empty(n_layers * (n + n_bonds))
    per = n + n_bonds
    for l in range(n_layers):
        beta, gamma = gamma_beta[2 * l], gamma_beta[2 * l + 1]
        theta[l * per: l * per + n] = beta
        theta[l * per + n: (l + 1) * per] = gamma * J
    return theta


def qaoa_cost_grad(gamma_beta, plan, psi0, bonds, J, h, n, patterns, n_layers):
    """Energy and gradient in the 2p QAOA parameters, by chain rule."""
    n_bonds = len(bonds)
    theta = qaoa_angles(gamma_beta, n_layers, n, J, n_bonds)
    energy, grad_theta = pilot.energy_cost_grad(
        theta, plan, psi0, bonds, J, h, n, patterns)

    per = n + n_bonds
    grad = np.zeros_like(gamma_beta)
    for l in range(n_layers):
        block = grad_theta[l * per: (l + 1) * per]
        grad[2 * l] = block[:n].sum()                       # d/d beta_l
        grad[2 * l + 1] = float(np.dot(block[n:], J))       # d/d gamma_l
    return energy, grad


def adiabatic_angles(n_layers, T_A, n, J, h, gamma_driver):
    """Fixed angles for a linear-schedule adiabatic sweep, M = n_layers steps.

    H(s) = (1-s)(-Gamma sum_q X_q) + s H,  H = -sum_b J_b Z Z - h sum_q X_q,
    so the transverse coefficient at s is c(s) = (1-s) Gamma + s h and the
    coupling coefficient is s J_b. Each step is one grouped second-order
    Trotter step, evaluated at the midpoint of its own interval, which keeps
    the discretisation error second order in dt rather than first.

    Angle conventions follow bppps.warm_start: exp(+i c X dt) is RX(-2 c dt),
    exp(+i s J ZZ dt) is RZZ(-2 s J dt).
    """
    dt = T_A / n_layers
    theta = np.empty(n_layers * (n + len(J)))
    per = n + len(J)
    for l in range(n_layers):
        s = (l + 0.5) / n_layers
        c = (1.0 - s) * gamma_driver + s * h
        theta[l * per: l * per + n] = -2.0 * c * dt
        theta[l * per + n: (l + 1) * per] = -2.0 * s * J * dt
    return theta


# ============================================================================
# Runs
# ============================================================================

def run_hva(model, plan_for, patterns, psi0, layers, max_seconds, e0, psi_gs):
    print("\n--- HVA (all angles free) ---")
    print(f"{'L':>3} {'2Q':>5} {'params':>7} {'energy':>12} {'dE':>10} "
          f"{'F0':>8} {'s':>7}")
    out = {}
    for L in layers:
        plan = plan_for(L)
        n_params = L * (model.num_qubits + model.num_bonds)
        best = None
        for seed in (0, 1):
            init = np.random.default_rng(seed).uniform(-0.1, 0.1, n_params)
            cg = lambda x: pilot.energy_cost_grad(
                x, plan, psi0, model.bonds, model.J, model.h,
                model.num_qubits, patterns)
            th, elapsed = pilot.optimise(cg, init, max_seconds)
            e = cg(th)[0]
            if best is None or e < best[1]:
                best = (th, e, elapsed)
        th, e, dt_s = best
        psi = evolve_only(th, plan, psi0, model.num_qubits, patterns)
        f0 = float(abs(np.vdot(psi_gs, psi)) ** 2)
        out[str(L)] = {'n_2q': model.num_bonds * L, 'n_layers': L, 'n_params': n_params,
                       'energy': e, 'gap': e - e0, 'fidelity': f0}
        print(f"{L:3d} {out[str(L)]['n_2q']:5d} {n_params:7d} {e:12.6f} "
              f"{e - e0:10.6f} {f0:8.4f} {dt_s:7.1f}")
    return out


def run_qaoa(model, plan_for, patterns, psi0, depths, max_seconds, e0, psi_gs):
    print("\n--- QAOA (2p angles, structurally a sub-family of HVA) ---")
    print(f"{'p':>3} {'2Q':>5} {'params':>7} {'energy':>12} {'dE':>10} "
          f"{'F0':>8} {'s':>7}")
    out = {}
    for p in depths:
        plan = plan_for(p)
        # Annealing-inspired initialisation, plus random restarts.
        #
        # QAOA at large p does not optimise from a random start -- the landscape
        # is full of poor local minima and the standard fix is to start from a
        # discretised anneal, which QAOA contains as a special case (beta
        # decreasing as the driver is switched off, gamma increasing as the
        # problem is switched on). Measured at 4x4 with random starts only,
        # p=10 stalled at dE=0.87 while p=4 reached 0.96 -- essentially no gain
        # from six extra layers, which is an optimiser artefact and would
        # understate QAOA badly.
        anneal = np.empty(2 * p)
        for l in range(p):
            s = (l + 0.5) / p
            anneal[2 * l] = -2.0 * (1.0 - s) * 1.0 * (1.0 / p)   # beta_l
            anneal[2 * l + 1] = -2.0 * s * (1.0 / p)             # gamma_l
        inits = [anneal] + [np.random.default_rng(100 + s).uniform(-0.5, 0.5, 2 * p)
                            for s in (0, 1)]
        best = None
        for init in inits:
            cg = lambda x: qaoa_cost_grad(
                x, plan, psi0, model.bonds, model.J, model.h,
                model.num_qubits, patterns, p)
            gb, elapsed = pilot.optimise(cg, init, max_seconds)
            e = cg(gb)[0]
            if best is None or e < best[1]:
                best = (gb, e, elapsed)
        gb, e, dt_s = best
        th = qaoa_angles(gb, p, model.num_qubits, model.J, model.num_bonds)
        psi = evolve_only(th, plan, psi0, model.num_qubits, patterns)
        f0 = float(abs(np.vdot(psi_gs, psi)) ** 2)
        out[str(p)] = {'n_2q': model.num_bonds * p, 'n_layers': p,
                       'n_params': 2 * p, 'energy': e, 'gap': e - e0,
                       'fidelity': f0}
        print(f"{p:3d} {out[str(p)]['n_2q']:5d} {2 * p:7d} {e:12.6f} "
              f"{e - e0:10.6f} {f0:8.4f} {dt_s:7.1f}")
    return out


def run_adiabatic(model, plan_for, patterns, psi0, steps, T_list,
                  gamma_driver, e0, psi_gs):
    print(f"\n--- Adiabatic Trotter (no optimisation, Gamma={gamma_driver}) ---")
    print("  best total anneal time T_A at each step count")
    print(f"{'M':>4} {'2Q':>5} {'best T_A':>9} {'energy':>12} {'dE':>10} "
          f"{'F0':>8}")
    out = {}
    for M in steps:
        plan = plan_for(M)
        best = None
        for T_A in T_list:
            th = adiabatic_angles(M, T_A, model.num_qubits, model.J,
                                  model.h, gamma_driver)
            psi = evolve_only(th, plan, psi0, model.num_qubits, patterns)
            e = energy_of(psi, model.bonds, model.J, model.h,
                          model.num_qubits, patterns)
            if best is None or e < best[1]:
                best = (T_A, e, psi)
        T_A, e, psi = best
        f0 = float(abs(np.vdot(psi_gs, psi)) ** 2)
        out[str(M)] = {'n_2q': model.num_bonds * M, 'n_layers': M,
                       'n_params': 0, 'best_T_A': T_A, 'energy': e,
                       'gap': e - e0, 'fidelity': f0}
        print(f"{M:4d} {out[str(M)]['n_2q']:5d} {T_A:9.2f} {e:12.6f} "
              f"{e - e0:10.6f} {f0:8.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lattice', type=int, default=4)
    ap.add_argument('--max-seconds', type=float, default=120.0)
    ap.add_argument('--gamma-driver', type=float, default=1.0)
    # Sweeping the disorder realisation is the point of the mechanism
    # experiment: the claim is that HVA's per-bond angles absorb disorder that
    # a schedule cannot, so its advantage over the anneal should widen as the
    # E1-E0 gap closes. That needs several seeds on the same lattice.
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    L = args.lattice
    model = SpinGlass2D(L, L, h=1.0, coupling_type='ea_bimodal', seed=args.seed)
    n = model.num_qubits
    print(f"{L}x{L}, {n} qubits, {model.num_bonds} bonds "
          f"({model.num_bonds} 2Q gates per layer)")

    ed = ExactDiag(model.build_sparse_matrix(), n)
    energies, states = ed.ground_state(k=2)
    e0, e1 = float(energies[0]), float(energies[1])
    psi_gs = states[:, 0]
    print(f"ED: E0 = {e0:.6f}, E1 = {e1:.6f}, gap = {e1 - e0:.6f}")

    patterns = [pilot.zz_pattern(i, j, n) for (i, j) in model.bonds]
    plan_for = pilot.build_plan(n, model.bonds)
    psi0 = plus_state(n)

    layers = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    anneal_steps = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]
    T_list = list(np.arange(0.5, 30.1, 0.5))

    res = {
        'lattice': f"{L}x{L}", 'n_qubits': n, 'n_bonds': model.num_bonds,
        'seed': args.seed,
        'ed_ground_energy': e0, 'ed_first_excited': e1,
        'gamma_driver': args.gamma_driver,
        'adiabatic': run_adiabatic(model, plan_for, patterns, psi0,
                                   anneal_steps, T_list, args.gamma_driver,
                                   e0, psi_gs),
        'qaoa': run_qaoa(model, plan_for, patterns, psi0, layers,
                         args.max_seconds, e0, psi_gs),
        'hva': run_hva(model, plan_for, patterns, psi0, layers,
                       args.max_seconds, e0, psi_gs),
    }

    suffix = '' if args.seed == 42 else f'_seed{args.seed}'
    out = args.out or os.path.join(ROOT, 'results', f'{L}x{L}',
                                   f'gs_competitors{suffix}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
