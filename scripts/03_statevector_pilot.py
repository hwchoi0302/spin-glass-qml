#!/usr/bin/env python3
"""Statevector pilot: a fast, target-free lower bound for two open questions.

Both T1-T's "does the required HVA depth grow sublinearly with T" and goal 3's
"does the ground-state fidelity ceiling rise with more layers" are, at bottom,
questions about the *ansatz*, not about BP-PPS's ability to train it. At 16
qubits both can be answered directly: build the statevector, optimise the
circuit against the true cost (state fidelity, or energy) with exact
gradients, no Trotter target and no coefficient truncation at all.

  - Time evolution: optimise U(theta; T)|0...0> against ED's exp(-iHT)|0...0>
    directly (fidelity, not L_XZ). This is what T1-T's stage 3 actually scores
    (`accuracy_target: 0.99`), so the layer count found here is a genuine
    lower bound on what BP-PPS would need -- if this alone grows linearly in
    T, BP-PPS cannot do better, and stage 1's target generation is not worth
    chasing further for this question.
  - Ground state: optimise U(theta)|+...+> against ED's exact ground energy
    at several layer counts, to see whether the fidelity ceiling (0.724 at 3
    layers, see docs/results_4x4.md SS5) actually rises with depth.

Gradients are exact analytic adjoint differentiation (the standard
backprop-through-a-statevector method: forward-propagate storing every
intermediate state, then walk the gate list backward accumulating
Im[<lambda|generator|phi>] and un-rotating lambda by each gate's inverse).
This is O(n_params) forward+backward passes total, not O(n_params) *forward
passes each O(n_params) gates* the way parameter-shift is -- the difference
between a ~13s and a ~1h optimisation at 8 layers (320 parameters).

Because a sign or transpose error here would silently produce a plausible
but wrong "compression scaling" conclusion -- this script's actual point --
`self_check()` validates the hand-rolled forward pass against qiskit's
Statevector for the real 4x4 lattice, and the adjoint gradient against the
provably-exact (if much slower) parameter-shift rule, before anything else
runs. Both must pass or the script aborts.

Usage:
    python scripts/03_statevector_pilot.py
    python scripts/03_statevector_pilot.py --part te     # time evolution only
    python scripts/03_statevector_pilot.py --part gs     # ground state only
    python scripts/03_statevector_pilot.py --max-seconds 120   # per (T,L) cap
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from ansatz import HVA                                          # noqa: E402
from bppps.warm_start import trotter_warm_start                 # noqa: E402
from classical_bench import ExactDiag                            # noqa: E402
from config import load_config, output_dir                       # noqa: E402
from hamiltonians import SpinGlass2D                              # noqa: E402
from qiskit.quantum_info import Statevector                       # noqa: E402


# ============================================================================
# Fast numpy statevector primitives (little-endian: qubit q = bit q of the
# basis-state index, matching Qiskit's Statevector convention exactly -- this
# is asserted, not assumed, in self_check()).
# ============================================================================

def apply_rx(psi, q, theta, n):
    """Full rotation exp(-i*theta/2 * X_q), returns a new array."""
    psi_r = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    a0, a1 = psi_r[:, 0, :], psi_r[:, 1, :]
    out = np.empty_like(psi_r)
    out[:, 0, :] = c * a0 - 1j * s * a1
    out[:, 1, :] = -1j * s * a0 + c * a1
    return out.reshape(-1)


def apply_x(psi, q, n):
    """Bare generator X_q (no phase), returns a new array."""
    psi_r = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    return psi_r[:, ::-1, :].reshape(-1).copy()


def zz_pattern(i, j, n):
    """+1 where qubits i,j agree, -1 where they differ -- eigenvalues of Z_i Z_j."""
    idx = np.arange(2 ** n)
    bi = (idx >> i) & 1
    bj = (idx >> j) & 1
    return (1 - 2 * (bi ^ bj)).astype(np.float64)


def apply_rzz(psi, theta, pattern):
    """Full rotation exp(-i*theta/2 * Z_i Z_j), returns a new array."""
    return psi * np.exp(-1j * theta / 2 * pattern)


def apply_zz(psi, pattern):
    """Bare generator Z_i Z_j (no phase), returns a new array."""
    return psi * pattern


def apply_H(psi, bonds, J, h, n, patterns):
    """H = -sum_b J_b Z_i Z_j - h sum_q X_q, applied directly (no matrix)."""
    out = np.zeros_like(psi)
    for bidx, (i, j) in enumerate(bonds):
        out -= J[bidx] * (patterns[bidx] * psi)
    for q in range(n):
        out -= h * apply_x(psi, q, n)
    return out


# ============================================================================
# Gate plan: same parameter layout as ansatz.HVA.build_circuit (RX block then
# RZZ block per layer, RZZ indexed by the bond's position in `bonds`, not by
# substep -- substep order only matters for circuit scheduling, and RZZ gates
# are all diagonal so they commute regardless of order).
# ============================================================================

def build_plan(n_qubits, bonds):
    params_per_layer = n_qubits + len(bonds)

    def plan_for(n_layers):
        plan = []
        for layer in range(n_layers):
            offset = layer * params_per_layer
            for q in range(n_qubits):
                plan.append(('rx', q, offset + q))
            for bidx, (i, j) in enumerate(bonds):
                plan.append(('rzz', i, j, offset + n_qubits + bidx, bidx))
        return plan
    return plan_for


def forward_pass(theta, plan, psi0, n, patterns):
    """Returns [phi_0, phi_1, ..., phi_M] (M = len(plan))."""
    phis = [psi0]
    psi = psi0
    for g in plan:
        if g[0] == 'rx':
            psi = apply_rx(psi, g[1], theta[g[2]], n)
        else:
            _, i, j, pidx, bidx = g
            psi = apply_rzz(psi, theta[pidx], patterns[bidx])
        phis.append(psi)
    return phis


def adjoint_grad(theta, plan, phis, lam0, n, patterns):
    """dC/dtheta for C = <phi_M|O|phi_M>, given lam0 = O|phi_M>.

    Standard adjoint method: walk the gate list backward, reading off
    Im[<lambda|generator|phi_after_gate>] at each gate, then un-rotate
    lambda by that gate's inverse (same gate, negated angle) before moving
    to the previous one.
    """
    grad = np.zeros(len(theta))
    lam = lam0
    for k in range(len(plan) - 1, -1, -1):
        g = plan[k]
        phi_after = phis[k + 1]
        if g[0] == 'rx':
            q, pidx = g[1], g[2]
            grad[pidx] = np.imag(np.vdot(lam, apply_x(phi_after, q, n)))
            lam = apply_rx(lam, q, -theta[pidx], n)
        else:
            _, i, j, pidx, bidx = g
            grad[pidx] = np.imag(np.vdot(lam, apply_zz(phi_after, patterns[bidx])))
            lam = apply_rzz(lam, -theta[pidx], patterns[bidx])
    return grad


# ============================================================================
# Cost functions (return (cost, grad) for scipy's jac=True convention)
# ============================================================================

def fidelity_cost_grad(theta, plan, psi0, target, n, patterns):
    """1 - |<target|U(theta)|psi0>|^2 and its gradient, for minimisation."""
    phis = forward_pass(theta, plan, psi0, n, patterns)
    overlap = np.vdot(target, phis[-1])
    fidelity = float(np.abs(overlap) ** 2)
    lam0 = target * overlap  # O|phi_M> for O = |target><target|
    grad_fid = adjoint_grad(theta, plan, phis, lam0, n, patterns)
    return 1.0 - fidelity, -grad_fid


def energy_cost_grad(theta, plan, psi0, bonds, J, h, n, patterns):
    """<psi0|U(theta)^dag H U(theta)|psi0> and its gradient."""
    phis = forward_pass(theta, plan, psi0, n, patterns)
    Hphi = apply_H(phis[-1], bonds, J, h, n, patterns)
    energy = float(np.real(np.vdot(phis[-1], Hphi)))
    grad_e = adjoint_grad(theta, plan, phis, Hphi, n, patterns)
    return energy, grad_e


# ============================================================================
# Reference (slow, provably-exact) parameter-shift gradient, for validation only
# ============================================================================

def parameter_shift_grad(cost_only_fn, theta):
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        orig = theta[i]
        theta[i] = orig + np.pi / 2
        f_plus = cost_only_fn(theta)
        theta[i] = orig - np.pi / 2
        f_minus = cost_only_fn(theta)
        theta[i] = orig
        grad[i] = (f_plus - f_minus) / 2
    return grad


# ============================================================================
# Self-check: must pass before any real optimisation is trusted
# ============================================================================

def self_check(model):
    n = model.num_qubits
    bonds = model.bonds
    plan_for = build_plan(n, bonds)
    plan = plan_for(2)
    n_params = len(plan)
    patterns = [zz_pattern(i, j, n) for (i, j) in bonds]

    rng = np.random.default_rng(0)
    theta = rng.uniform(-1, 1, n_params)

    print("  [1/3] forward pass vs qiskit Statevector ...")
    hva = HVA(n, bonds, 2, model.Lx, model.Ly, J=model.J)
    qc = hva.build_circuit(theta)
    for label in ('0' * n, '+' * n):
        psi0 = np.array(Statevector.from_label(label))
        phis = forward_pass(theta, plan, psi0, n, patterns)
        psi_qiskit = np.array(Statevector.from_label(label).evolve(qc))
        dev = np.max(np.abs(phis[-1] - psi_qiskit))
        print(f"      |{label[0]}...{label[0]}> start: max |mine - qiskit| = {dev:.3e}")
        assert dev < 1e-9, f"forward pass does not match qiskit for |{label}>"

    print("  [2/3] adjoint gradient vs parameter-shift (fidelity cost) ...")
    psi0 = np.array(Statevector.from_label('0' * n))
    target = np.array(Statevector.from_label('+' * n))  # arbitrary fixed target
    _, grad_adj = fidelity_cost_grad(theta, plan, psi0, target, n, patterns)

    def cost_only(th):
        phis = forward_pass(th, plan, psi0, n, patterns)
        return 1.0 - float(np.abs(np.vdot(target, phis[-1])) ** 2)
    grad_ps = parameter_shift_grad(cost_only, theta.copy())
    dev = np.max(np.abs(grad_adj - grad_ps))
    print(f"      max |adjoint - parameter-shift| = {dev:.3e}  ({n_params} params)")
    assert dev < 1e-8, "adjoint gradient (fidelity) does not match parameter-shift"

    print("  [3/3] adjoint gradient vs parameter-shift (energy cost) ...")
    psi0p = np.array(Statevector.from_label('+' * n))
    _, grad_adj_e = energy_cost_grad(theta, plan, psi0p, bonds, model.J, model.h, n, patterns)

    def energy_only(th):
        phis = forward_pass(th, plan, psi0p, n, patterns)
        Hphi = apply_H(phis[-1], bonds, model.J, model.h, n, patterns)
        return float(np.real(np.vdot(phis[-1], Hphi)))
    grad_ps_e = parameter_shift_grad(energy_only, theta.copy())
    dev_e = np.max(np.abs(grad_adj_e - grad_ps_e))
    print(f"      max |adjoint - parameter-shift| = {dev_e:.3e}")
    assert dev_e < 1e-8, "adjoint gradient (energy) does not match parameter-shift"

    print("  self-check PASSED\n")


# ============================================================================
# Optimisation driver
# ============================================================================

def optimise(cost_grad_fn, params_init, max_seconds, maxiter=300):
    t0 = time.time()
    state = {'x': params_init.copy()}

    def cb(xk):
        state['x'] = xk.copy()
        if time.time() - t0 > max_seconds:
            raise StopIteration

    try:
        res = minimize(cost_grad_fn, params_init, jac=True, method='L-BFGS-B',
                       callback=cb,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        x = res.x
    except StopIteration:
        x = state['x']
    return x, time.time() - t0


# ============================================================================
# Pilots
# ============================================================================

def time_evolution_pilot(model, ed, plan_for, patterns, T_list, layer_list,
                         accuracy_target, max_seconds, out):
    n = model.num_qubits
    psi0 = np.array(Statevector.from_label('0' * n))
    section = out.setdefault('time_evolution', {})

    print(f"\n  {'T':>5} {'L':>3} {'fidelity':>12} {'n_params':>9} {'time_s':>8}")
    print("  " + "-" * 45)
    for T in T_list:
        psi_exact = ed.time_evolve(np.array(psi0, dtype=complex), T)
        by_T = section.setdefault(str(T), {})
        for L in layer_list:
            plan = plan_for(L)
            params_init = trotter_warm_start(n, model.bonds, model.J, model.h, T, L)
            t0 = time.time()
            x, elapsed = optimise(
                lambda th: fidelity_cost_grad(th, plan, psi0, psi_exact, n, patterns),
                params_init, max_seconds)
            fid = 1.0 - fidelity_cost_grad(x, plan, psi0, psi_exact, n, patterns)[0]
            by_T[str(L)] = {
                'fidelity': fid, 'n_params': len(plan), 'time_s': elapsed,
            }
            print(f"  {T:5.1f} {L:3d} {fid:12.8f} {len(plan):9d} {elapsed:8.1f}")
            with open(os.path.join(os.path.dirname(out['_path']), 'statevector_pilot.json'), 'w') as f:
                json.dump({k: v for k, v in out.items() if not k.startswith('_')}, f, indent=2)
            if fid >= accuracy_target:
                print(f"        reached {accuracy_target} target; skipping deeper layers at T={T}")
                break


def ground_state_pilot(model, ed, plan_for, patterns, layer_list, out):
    n = model.num_qubits
    psi0 = np.array(Statevector.from_label('+' * n))
    e0, states0 = ed.ground_state(k=1)
    e0 = float(e0[0])
    psi_gs = states0[:, 0]
    section = out.setdefault('ground_state', {})

    print(f"\n  {'L':>3} {'energy':>14} {'gap':>10} {'fidelity':>12} {'n_params':>9} {'time_s':>8}")
    print("  " + "-" * 62)
    for L in layer_list:
        plan = plan_for(L)
        params_init = trotter_warm_start(n, model.bonds, model.J, model.h, 0.5, L)
        x, elapsed = optimise(
            lambda th: energy_cost_grad(th, plan, psi0, model.bonds, model.J, model.h, n, patterns),
            params_init, max_seconds=600)
        phis = forward_pass(x, plan, psi0, n, patterns)
        energy = float(np.real(np.vdot(phis[-1], apply_H(phis[-1], model.bonds, model.J, model.h, n, patterns))))
        fid = float(np.abs(np.vdot(psi_gs, phis[-1])) ** 2)
        section[str(L)] = {
            'energy': energy, 'gap': energy - e0, 'fidelity': fid,
            'n_params': len(plan), 'time_s': elapsed,
        }
        print(f"  {L:3d} {energy:14.6f} {energy - e0:10.4f} {fid:12.6f} {len(plan):9d} {elapsed:8.1f}")
        with open(os.path.join(os.path.dirname(out['_path']), 'statevector_pilot.json'), 'w') as f:
            json.dump({k: v for k, v in out.items() if not k.startswith('_')}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['te', 'gs', 'both'], default='both')
    parser.add_argument('--max-seconds', type=float, default=120.0,
                        help='wall-clock cap per (T, L) time-evolution optimisation')
    args = parser.parse_args()

    config = load_config()
    out_dir = output_dir(config, create=False)
    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))

    print("=" * 70)
    print("  STATEVECTOR PILOT -- target-free lower bound (16 qubits, exact)")
    print("=" * 70)
    self_check(model)

    n = model.num_qubits
    patterns = [zz_pattern(i, j, n) for (i, j) in model.bonds]
    plan_for = build_plan(n, model.bonds)
    H = model.build_sparse_matrix()
    ed = ExactDiag(H, n)

    out = {'_path': os.path.join(out_dir, 'statevector_pilot.json')}

    ts = config.get('time_sweep', {})
    # Independent of configs/time_sweep.yaml's (reduced) `snapshots` -- this
    # pilot's point is to explore T values stage 1 never got targets for.
    T_list = [k * ts.get('chunk_delta_t', 0.5) for k in (1, 2, 4, 8)]
    layer_list_te = ts.get('direct_layers', [2, 3, 4, 6, 8])
    accuracy_target = ts.get('accuracy_target', 0.99)

    if args.part in ('te', 'both'):
        print("\n" + "=" * 70)
        print("  TIME EVOLUTION: layers needed vs T (goal 1 / T1-T lower bound)")
        print("=" * 70)
        time_evolution_pilot(model, ed, plan_for, patterns, T_list, layer_list_te,
                             accuracy_target, args.max_seconds, out)

    if args.part in ('gs', 'both'):
        print("\n" + "=" * 70)
        print("  GROUND STATE: fidelity ceiling vs layers (goal 3)")
        print("=" * 70)
        ground_state_pilot(model, ed, plan_for, patterns,
                           layer_list=[1, 2, 3, 4, 5, 6, 8], out=out)

    print(f"\n  Saved: {out['_path']}")


if __name__ == '__main__':
    main()
