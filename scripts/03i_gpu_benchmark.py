#!/usr/bin/env python3
"""CPU vs GPU, per workload, with the 2026-08-30 confounds removed.

That first benchmark answered "is cupy faster than numpy for Pauli
propagation" with a clear no (100.3 s vs 12.0 s at 31.6K terms) and then left
the interesting half open, because the retry at ~1.2M terms -- the scale T1-T
actually produced -- was killed at 42 minutes under CPU contention. Two things
were wrong with it beyond the contention:

  1. `propagate_forward_sorted(..., stats=...)` evaluates
     `float(xp.sum(coeffs ** 2))` twice per gate. On a GPU each of those is a
     device-to-host sync that drains the pipeline, so the measurement was
     partly of synchronisation, not of the kernel. Here stats is None.
  2. The CPU side was carrying `np.union1d`, which in numpy 2.5 is ~70x
     slower than the sort-and-mask it should be. That has since been fixed
     (propagation_sorted._union_sorted), so the CPU baseline this compares
     against is 3-6x faster than the one cupy beat^H^Hlost to.

It also measures the workload the earlier benchmark never looked at, which is
the one with the best case for a GPU: the *statevector* the comparison models
run on (Trotter, QAOA, adiabatic). That kernel is fixed-shape, branch-free
and 1 MB, i.e. the opposite of Pauli propagation in every way that matters,
and 03f_gs_competitors.py is wall-clock-budgeted rather than
iteration-budgeted -- so throughput there buys competitor-curve quality, not
just a shorter run.

Nothing here is a physics result; it decides which engine the physics runs on.
Run it on the desktop, idle. Roughly 10-20 minutes.

Usage:
    python scripts/03i_gpu_benchmark.py
    python scripts/03i_gpu_benchmark.py --part prop      # Pauli propagation
    python scripts/03i_gpu_benchmark.py --part sv        # statevector
    python scripts/03i_gpu_benchmark.py --part train     # trainer inner loop
    python scripts/03i_gpu_benchmark.py --max-terms 1200000
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

from bppps.propagation_sorted import (                             # noqa: E402
    apply_rx_sorted, apply_rzz_sorted, propagate_forward_sorted,
)
from config import load_config, output_dir                          # noqa: E402
from hamiltonians.spin_glass_2d import SpinGlass2D                  # noqa: E402


def get_cupy():
    """cupy plus a usable device, or None. Never raises."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        cp.zeros(4).sum()
        return cp
    except Exception as exc:
        print(f"  [no GPU: {type(exc).__name__}: {exc}]")
        return None


def sync(xp):
    if xp is not np:
        xp.cuda.Stream.null.synchronize()


def bench(fn, xp, reps, warmup=1):
    """Median-of-reps seconds, GPU-synced, warm-up excluded."""
    for _ in range(warmup):
        fn()
    sync(xp)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(xp)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ============================================================================
# Part A -- Pauli propagation (target generation, and the trainer's gradient)
# ============================================================================

def part_prop(cp, max_terms, out):
    print("\n" + "=" * 72)
    print("A. Pauli propagation: sorted-array engine, numpy vs cupy")
    print("=" * 72)
    print("   Per gate this is ~35 whole-array calls with data-dependent")
    print("   shapes, two sorts and two binary searches. stats=None, so no")
    print("   device sync is forced inside the gate loop.\n")

    sizes = [n for n in (30_000, 100_000, 300_000, 1_200_000, 3_800_000)
             if n <= max_terms]
    rows = []
    rng = np.random.default_rng(0)
    print(f"{'N terms':>11} {'numpy RX':>11} {'numpy RZZ':>11} "
          f"{'cupy RX':>11} {'cupy RZZ':>11} {'GPU speedup':>12}")
    for N in sizes:
        keys_h = np.unique(rng.integers(0, 1 << 40, size=N, dtype=np.uint64))
        coeffs_h = rng.standard_normal(keys_h.size)
        reps = 9 if N <= 300_000 else 3

        t_rx = bench(lambda: apply_rx_sorted(keys_h, coeffs_h, 3, 0.3, 1e-12),
                     np, reps)
        t_rzz = bench(lambda: apply_rzz_sorted(keys_h, coeffs_h, 2, 5, 0.3, 1e-12),
                      np, reps)
        row = {'n_terms': int(keys_h.size), 'numpy_rx_ms': t_rx * 1e3,
               'numpy_rzz_ms': t_rzz * 1e3}

        if cp is not None:
            keys_d, coeffs_d = cp.asarray(keys_h), cp.asarray(coeffs_h)
            g_rx = bench(lambda: apply_rx_sorted(keys_d, coeffs_d, 3, 0.3,
                                                 1e-12, xp=cp), cp, reps)
            g_rzz = bench(lambda: apply_rzz_sorted(keys_d, coeffs_d, 2, 5, 0.3,
                                                   1e-12, xp=cp), cp, reps)
            row.update(cupy_rx_ms=g_rx * 1e3, cupy_rzz_ms=g_rzz * 1e3,
                       speedup=(t_rx + t_rzz) / (g_rx + g_rzz))
            print(f"{keys_h.size:>11,} {t_rx*1e3:>9.2f}ms {t_rzz*1e3:>9.2f}ms "
                  f"{g_rx*1e3:>9.2f}ms {g_rzz*1e3:>9.2f}ms "
                  f"{row['speedup']:>11.2f}x")
            del keys_d, coeffs_d
            cp.get_default_memory_pool().free_all_blocks()
        else:
            print(f"{keys_h.size:>11,} {t_rx*1e3:>9.2f}ms {t_rzz*1e3:>9.2f}ms "
                  f"{'--':>11} {'--':>11} {'--':>12}")
        rows.append(row)

    print("\n   A gate is not the unit that matters -- target generation is a")
    print("   sequential chain of 14,000 of them (dt=0.01) and cannot batch,")
    print("   because gate k+1 consumes gate k's output. Projected wall clock")
    print("   for one 14,000-gate chunk at fixed N (an underestimate: N grows):")
    for r in rows:
        line = (f"     N={r['n_terms']:>9,}  numpy "
                f"{14000 * (r['numpy_rx_ms'] + r['numpy_rzz_ms']) / 2 / 1e3:8.1f} s")
        if 'cupy_rx_ms' in r:
            line += (f"   cupy "
                     f"{14000 * (r['cupy_rx_ms'] + r['cupy_rzz_ms']) / 2 / 1e3:8.1f} s")
        print(line)
    out['propagation'] = rows


# ============================================================================
# Part B -- statevector (the comparison models: Trotter, QAOA, adiabatic)
# ============================================================================
#
# Backend-generic copies of 03_statevector_pilot.py's primitives. They are not
# imported from it because that module hardcodes `np.`; correctness here is
# established by checking against it, below, rather than by sharing code.

def sv_rx(psi, q, theta, n, xp):
    r = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    c, s = float(np.cos(theta / 2)), float(np.sin(theta / 2))
    out = xp.empty_like(r)
    out[:, 0, :] = c * r[:, 0, :] - 1j * s * r[:, 1, :]
    out[:, 1, :] = -1j * s * r[:, 0, :] + c * r[:, 1, :]
    return out.reshape(-1)


def sv_x(psi, q, n, xp):
    r = psi.reshape(2 ** (n - q - 1), 2, 2 ** q)
    return r[:, ::-1, :].reshape(-1).copy()


def sv_rzz(psi, theta, pattern, xp):
    c, s = float(np.cos(theta / 2)), float(np.sin(theta / 2))
    return psi * (c - 1j * s * pattern)


def sv_pattern(i, j, n, xp):
    idx = xp.arange(2 ** n)
    return xp.where(((idx >> i) & 1) == ((idx >> j) & 1), 1.0, -1.0)


def sv_H(psi, bonds, J, h, n, patterns, xp):
    out = xp.zeros_like(psi)
    for bidx in range(len(bonds)):
        out -= float(J[bidx]) * (patterns[bidx] * psi)
    for q in range(n):
        out -= h * sv_x(psi, q, n, xp)
    return out


def sv_energy_grad(theta, plan, psi0, bonds, J, h, n, patterns, xp):
    """Forward tape + adjoint sweep, the same shape as the pilot's."""
    phis = [psi0]
    psi = psi0
    for g in plan:
        psi = (sv_rx(psi, g[1], theta[g[2]], n, xp) if g[0] == 'rx'
               else sv_rzz(psi, theta[g[3]], patterns[g[4]], xp))
        phis.append(psi)
    lam = sv_H(phis[-1], bonds, J, h, n, patterns, xp)
    energy = float(xp.real(xp.vdot(phis[-1], lam)))
    grad = np.zeros(len(theta))
    for k in range(len(plan) - 1, -1, -1):
        g = plan[k]
        if g[0] == 'rx':
            gen, pidx = sv_x(phis[k + 1], g[1], n, xp), g[2]
        else:
            gen, pidx = phis[k + 1] * patterns[g[4]], g[3]
        grad[pidx] += float(xp.real(xp.vdot(lam, -1j * 0.5 * gen))) * 2.0
        lam = (sv_rx(lam, g[1], -theta[g[2]], n, xp) if g[0] == 'rx'
               else sv_rzz(lam, -theta[g[3]], patterns[g[4]], xp))
    return energy, grad


def part_sv(cp, out):
    print("\n" + "=" * 72)
    print("B. Statevector: the comparison models' kernel, numpy vs cupy")
    print("=" * 72)

    model = SpinGlass2D(4, 4, seed=42)
    n, bonds, J, h = model.num_qubits, model.bonds, model.J, model.h
    per = n + len(bonds)

    spec = importlib.util.spec_from_file_location(
        'pilot', os.path.join(ROOT, 'scripts', '03_statevector_pilot.py'))
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    plan_for = pilot.build_plan(n, bonds)

    pats_h = [sv_pattern(i, j, n, np) for (i, j) in bonds]
    psi0_h = np.full(1 << n, 2.0 ** (-n / 2.0), dtype=np.complex128)

    # This file's primitives must reproduce the validated pilot exactly before
    # any of its timings mean anything.
    theta = np.random.default_rng(0).uniform(-0.3, 0.3, 3 * per)
    e_ref, g_ref = pilot.energy_cost_grad(theta, plan_for(3), psi0_h, bonds, J,
                                          h, n, pats_h)
    e_mine, g_mine = sv_energy_grad(theta, plan_for(3), psi0_h, bonds, J, h, n,
                                    pats_h, np)
    d_e, d_g = abs(e_ref - e_mine), float(np.max(np.abs(g_ref - g_mine)))
    print(f"  vs 03_statevector_pilot: |dE| = {d_e:.3e}, "
          f"max |dgrad| = {d_g:.3e}")
    assert d_e < 1e-10 and d_g < 1e-10, "backend-generic primitives disagree"

    if cp is not None:
        pats_d = [cp.asarray(p) for p in pats_h]
        psi0_d = cp.asarray(psi0_h)
        e_g, g_g = sv_energy_grad(theta, plan_for(3), psi0_d, bonds, J, h, n,
                                  pats_d, cp)
        print(f"  cupy vs numpy:           |dE| = {abs(e_mine - e_g):.3e}, "
              f"max |dgrad| = {float(np.max(np.abs(g_mine - g_g))):.3e}")

    print("\n  B1. HVA / QAOA -- optimised, so wall-clock-capped. The layer")
    print("      list is 03f_gs_competitors.py's own: [1..6, 8, 10, 12].")
    print(f"\n{'L':>4} {'2Q':>6} {'gates':>7} {'tape MB':>9} "
          f"{'numpy':>11} {'cupy':>11} {'speedup':>9} {'iters/120s':>22}")
    rows = []
    for L in (2, 3, 5, 8, 10, 12):
        plan = plan_for(L)
        th = np.random.default_rng(0).uniform(-0.1, 0.1, L * per)
        tape = 16 * (1 << n) * (len(plan) + 1) / 1e6
        reps = 5 if L <= 8 else 2
        t_c = bench(lambda: sv_energy_grad(th, plan, psi0_h, bonds, J, h, n,
                                           pats_h, np), np, reps)
        row = {'n_layers': L, 'n_2q': L * len(bonds), 'n_gates': len(plan),
               'tape_mb': tape, 'numpy_s': t_c}
        cell_g, cell_s = '--', '--'
        iters = f"cpu {120/t_c:>5.0f}"
        if cp is not None:
            t_g = bench(lambda: sv_energy_grad(th, plan, psi0_d, bonds, J, h,
                                               n, pats_d, cp), cp, reps)
            row.update(cupy_s=t_g, speedup=t_c / t_g)
            cell_g, cell_s = f"{t_g*1e3:>9.1f}ms", f"{t_c/t_g:>8.1f}x"
            iters = f"cpu {120/t_c:>5.0f} -> gpu {120/t_g:>6.0f}"
        print(f"{L:>4} {L*len(bonds):>6} {len(plan):>7} {tape:>9.0f} "
              f"{t_c*1e3:>9.1f}ms {cell_g:>11} {cell_s:>9} {iters:>22}")
        rows.append(row)

    print("\n      120 s per point, maxiter 300. Wherever 'iters/120s' is under")
    print("      300 the optimiser is cut off mid-descent and throughput buys")
    print("      convergence, not a shorter run.")
    out['statevector'] = rows
    part_sv_adiabatic(cp, model, plan_for, pats_h, psi0_h, out,
                      None if cp is None else pats_d)


def sv_evolve(thetas, plan, psi0, n, patterns, xp):
    """Final state only, for a *batch* of angle vectors on a leading axis.

    The adiabatic sweep optimises nothing: it scans 60 independent anneal
    times per step count. That is the only workload in this project with no
    data dependence between its pieces, and therefore the only one that can
    be batched at all -- which matters because a lone 2^16 statevector is far
    too small to occupy a GPU.
    """
    B = thetas.shape[0]
    psi = xp.broadcast_to(psi0, (B, psi0.size)).copy()
    for g in plan:
        if g[0] == 'rx':
            q = g[1]
            r = psi.reshape(B, 2 ** (n - q - 1), 2, 2 ** q)
            c = xp.cos(thetas[:, g[2]] / 2)[:, None, None]
            s = xp.sin(thetas[:, g[2]] / 2)[:, None, None]
            a0, a1 = r[:, :, 0, :], r[:, :, 1, :]
            o = xp.empty_like(r)
            o[:, :, 0, :] = c * a0 - 1j * s * a1
            o[:, :, 1, :] = -1j * s * a0 + c * a1
            psi = o.reshape(B, -1)
        else:
            c = xp.cos(thetas[:, g[3]] / 2)[:, None]
            s = xp.sin(thetas[:, g[3]] / 2)[:, None]
            psi = psi * (c - 1j * s * patterns[g[4]][None, :])
    return psi


def part_sv_adiabatic(cp, model, plan_for, pats_h, psi0_h, out, pats_d):
    n, per = model.num_qubits, model.num_qubits + model.num_bonds
    n_anneal = 60                      # len(np.arange(0.5, 30.1, 0.5))
    print("\n  B2. Adiabatic anneal-time scan -- not optimised, and 60-way")
    print("      independent. On CPU, batching those 60 is a *loss* (the")
    print("      working set leaves L2); on a GPU it is the whole point.")
    print(f"\n{'M':>4} {'2Q':>6} {'serial CPU':>12} {'batched CPU':>12} "
          f"{'batched GPU':>12} {'best vs serial CPU':>20}")
    rows = []
    for M in (8, 24, 32):
        plan = plan_for(M)
        th = np.random.default_rng(0).uniform(-0.1, 0.1, (n_anneal, M * per))
        t_ser = bench(lambda: [sv_evolve(th[i:i+1], plan, psi0_h, n, pats_h, np)
                               for i in range(n_anneal)], np, 1)
        t_bat = bench(lambda: sv_evolve(th, plan, psi0_h, n, pats_h, np), np, 2)
        row = {'n_steps': M, 'n_2q': M * model.num_bonds,
               'serial_cpu_s': t_ser, 'batched_cpu_s': t_bat}
        cell_g, cell_b = '--', f"{t_ser/t_bat:>19.2f}x"
        if cp is not None:
            th_d = cp.asarray(th)
            psi0_d = cp.asarray(psi0_h)
            t_gpu = bench(lambda: sv_evolve(th_d, plan, psi0_d, n, pats_d, cp),
                          cp, 2)
            row.update(batched_gpu_s=t_gpu, gpu_speedup=t_ser / t_gpu)
            cell_g = f"{t_gpu:>10.3f}s"
            cell_b = f"{t_ser/min(t_bat, t_gpu):>19.2f}x"
        print(f"{M:>4} {M*model.num_bonds:>6} {t_ser:>10.3f}s {t_bat:>10.3f}s "
              f"{cell_g:>12} {cell_b:>20}")
        rows.append(row)
    print(f"\n      Whole sweep as configured: 60 anneal times x 40 gates x")
    print(f"      sum(M)=466 = 1,118,400 gate applications.")
    out['adiabatic_scan'] = rows


# ============================================================================
# Part C -- the trainer's inner loop (CPU only; this one has no GPU question)
# ============================================================================

def part_train(out):
    print("\n" + "=" * 72)
    print("C. Trainer inner loop: the target scan that is now gone (CPU)")
    print("=" * 72)
    print("   BPPPSTrainer._time_evolution_step used to walk the whole target")
    print("   SPO once per observable per iteration, to add up the")
    print("   target-only strings. The target is fixed for the entire")
    print("   optimisation and `evolved` is small, so that is now")
    print("   ||t||^2 - sum over the intersection, with ||t||^2 precomputed.")
    print("   No array backend was ever going to touch this -- it was a")
    print("   Python interpreter loop, and the fix is an identity.\n")

    rng = np.random.default_rng(0)
    rows = []
    print(f"{'target':>10} {'evolved':>9} {'old (scan)':>12} {'new (identity)':>15} "
          f"{'speedup':>9} {'loss deviation':>16}")
    for n_tgt in (300_000, 1_200_000):
        keys = rng.integers(0, 1 << 40, size=n_tgt, dtype=np.uint64).tolist()
        tgt = dict(zip(keys, (rng.standard_normal(n_tgt) * 1e-3).tolist()))
        t_sq = sum(c * c for c in tgt.values())
        for n_ev in (3_000, 30_000):
            ev = {k: tgt[k] + 1e-5 for k in keys[:n_ev]}

            def old():
                loss, seed = 0.0, {}
                for P, a in ev.items():
                    d = a - tgt.get(P, 0.0)
                    loss += d * d
                    if d != 0.0:
                        seed[P] = 2.0 * d
                for P, t in tgt.items():
                    if P not in ev:
                        loss += t * t
                return loss

            def new():
                loss, hit, seed = 0.0, 0.0, {}
                for P, a in ev.items():
                    t = tgt.get(P, 0.0)
                    d = a - t
                    loss += d * d
                    hit += t * t
                    if d != 0.0:
                        seed[P] = 2.0 * d
                return loss + (t_sq - hit)

            t_old = bench(old, np, 2)
            t_new = bench(new, np, 20)
            dev = abs(old() - new())
            print(f"{n_tgt:>10,} {n_ev:>9,} {t_old*1e3:>10.1f}ms "
                  f"{t_new*1e3:>13.2f}ms {t_old/t_new:>8.0f}x {dev:>16.2e}")
            rows.append({'n_target': n_tgt, 'n_evolved': n_ev,
                         'old_s': t_old, 'new_s': t_new,
                         'speedup': t_old / t_new, 'loss_deviation': dev})

    worst = max(r['old_s'] for r in rows)
    best = max(r['new_s'] for r in rows)
    print(f"\n   Over 32 observables x 1000 iterations that is "
          f"{32*worst*1000/60:.0f} min -> {32*best*1000:.0f} s.")
    out['trainer_inner_loop'] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['all', 'prop', 'sv', 'train'],
                    default='all')
    ap.add_argument('--max-terms', type=int, default=1_200_000)
    args = ap.parse_args()

    print("=" * 72)
    print("CPU vs GPU by workload -- see docs/issues/03-engine-performance.md")
    print("=" * 72)
    print(f"  numpy {np.__version__}")
    cp = get_cupy()
    if cp is not None:
        dev = cp.cuda.Device(0)
        free, total = dev.mem_info
        print(f"  cupy {cp.__version__}, compute capability "
              f"{dev.compute_capability}, VRAM {total/1e9:.1f} GB "
              f"({free/1e9:.1f} free)")

    out = {'numpy_version': np.__version__,
           'cupy_version': None if cp is None else cp.__version__}
    if args.part in ('all', 'prop'):
        part_prop(cp, args.max_terms, out)
    if args.part in ('all', 'sv'):
        part_sv(cp, out)
    if args.part in ('all', 'train'):
        part_train(out)

    # Merge rather than overwrite, so `--part sv` does not discard an
    # earlier `--part prop` run's numbers.
    path = os.path.join(output_dir(load_config()), 'gpu_benchmark.json')
    if os.path.exists(path):
        with open(path) as fh:
            merged = json.load(fh)
        merged.update(out)
        out = merged
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
