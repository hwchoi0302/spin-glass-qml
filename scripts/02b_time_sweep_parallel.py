#!/usr/bin/env python3
"""T1-T stage 1, parallelised across observables, with a live memory watchdog.

scripts/02_time_sweep.py's generate_series propagates the 32 observables one
at a time through each chunk (see docs/issues/01-scale-plan.md, "T1-T 진행
상황"). A py-spy dump on that process showed it still on the 2nd of 32
observables after ~11 hours of CPU time for k=1 alone -- the 32 observables
are fully independent of each other (only the *same* observable's own chunks
are sequential in k), so this is an embarrassingly parallel loop that was
just never split across the desktop's cores.

This script does not touch propagation.py or target_generator.py -- both are
the validated oracle (00_validate_small.py TEST 10/11/14) -- it only adds a
process-pool orchestration layer around the same build_trotter_gate_sequence
/ propagate_forward functions those modules already use.

Design:
  - One worker process per observable (from a pool of --workers), each
    running that observable through every chunk up to k_max, saving its
    result to disk *as soon as each requested snapshot k completes* (not
    only at the end of its own job) -- an aborted worker only loses its
    own in-flight chunk, not chunks already snapshotted.
  - The parent polls /proc/meminfo every few seconds. If available memory
    drops under --mem-floor-gb, it kills the whole pool immediately
    (pool.terminate()) and moves on to assembling whatever is on disk.
    Parallelism buys wall-clock time, not memory headroom -- N workers
    can hold up to N large SPOs at once, which is *worse* on the memory
    axis than the sequential original, hence the aggressive floor.
  - After the pool ends (complete or aborted), assembles per-k
    targets_k<k>.json from whatever partial files exist and writes
    targets_meta.json noting which of the 32 observables made it into
    each k, honestly, rather than only writing complete snapshots.

Usage:
    python -u scripts/02b_time_sweep_parallel.py --snapshots 1 2 4 8 --workers 6
    python -u scripts/02b_time_sweep_parallel.py --snapshots 1 2 4 8 --workers 6 --mem-floor-gb 3
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from bppps.pauli_utils import make_observable_label              # noqa: E402
from bppps.propagation import (                                  # noqa: E402
    TruncationStats, build_trotter_gate_sequence,
)
from bppps.propagation_numba import (                             # noqa: E402
    empty_dict, make_key, propagate_forward_numba,
)
from bppps.propagation_packed import label_to_xz, xz_to_label     # noqa: E402
from config import apply_overrides, load_config, output_dir      # noqa: E402
from hamiltonians import SpinGlass2D                              # noqa: E402


def partial_path(sw_dir, k, pauli, q):
    return os.path.join(sw_dir, '_partial', f"k{k}_{pauli}_{q}.json")


def worker(args):
    """Propagate one observable through every chunk, snapshotting as it goes.

    Runs in a child process, using the numba-JIT bit-packed engine
    (propagation_numba, TEST 18 in 00_validate_small.py validates it
    term-for-term against the string-dict oracle) -- ~3.5x faster per gate
    than the string engine on this model, measured on this machine while
    under CPU contention from other work. Returns
    (pauli, q, list_of_k_done, elapsed_s) so the parent can log progress; the
    actual data goes straight to disk so an aborted worker still leaves
    whatever snapshots it reached. Already-saved (k, pauli, q) snapshots are
    skipped, so a re-run after an abort resumes rather than redoing work.
    """
    (pauli, q, num_qubits, block, k_max, snapshots, delta, sw_dir) = args
    label = make_observable_label(num_qubits, pauli, q)
    x0, z0 = label_to_xz(label)
    coeffs = empty_dict()
    coeffs[make_key(x0, z0)] = 1.0
    stats = TruncationStats()
    t0 = time.time()
    done = []
    os.makedirs(os.path.join(sw_dir, '_partial'), exist_ok=True)
    for k in range(1, k_max + 1):
        path = partial_path(sw_dir, k, pauli, q)
        coeffs = propagate_forward_numba(coeffs, block, delta, stats)
        if k in snapshots:
            if os.path.exists(path):
                done.append(k)
                continue
            spo = {}
            for key, v in coeffs.items():
                x = int(key & 0xFFFFFFFF)
                z = int((key >> 32) & 0xFFFFFFFF)
                spo[xz_to_label(x, z, num_qubits)] = v
            tmp = path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({
                    'spo': spo, 'n_terms': len(spo),
                    'eps_emp': stats.error_estimate,
                    'elapsed_s': time.time() - t0,
                }, f)
            os.replace(tmp, path)
            done.append(k)
            print(f"    [{pauli}_{q}] k={k} done: {len(spo)} terms, "
                  f"eps_emp={stats.error_estimate:.3e}, "
                  f"{time.time() - t0:.1f}s", flush=True)
    return pauli, q, done, time.time() - t0


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024 / 1024
    return float('inf')  # unknown -- do not block on it


def assemble(sw_dir, k_max, snapshots, obs_list, chunk, dt, order, cutoff):
    """Build targets_k<k>.json + targets_meta.json from whatever is on disk."""
    meta = {}
    for k in snapshots:
        payload = {}
        for pauli, q in obs_list:
            path = partial_path(sw_dir, k, pauli, q)
            if os.path.exists(path):
                with open(path) as f:
                    payload[f"{pauli}_{q}"] = json.load(f)['spo']
        complete = len(payload) == len(obs_list)
        out_path = os.path.join(sw_dir, f"targets_k{k}.json")
        with open(out_path, 'w') as f:
            json.dump(payload, f)
        meta[str(k)] = {
            'T': k * chunk,
            'n_observables_present': len(payload),
            'n_observables_total': len(obs_list),
            'complete': complete,
            'n_terms': sum(len(v) for v in payload.values()) if payload else 0,
            'largest_observable': max((len(v) for v in payload.values()), default=0),
            'dt': dt, 'order': order, 'cutoff': cutoff,
        }
        flag = 'complete' if complete else f"PARTIAL ({len(payload)}/{len(obs_list)})"
        print(f"  k={k} (T={k*chunk}): {flag} -> {out_path}")
    meta_path = os.path.join(sw_dir, 'targets_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--snapshots', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--mem-floor-gb', type=float, default=2.5,
                        help='abort the whole pool if available memory drops below this')
    parser.add_argument('--poll-seconds', type=float, default=5.0)
    args = parser.parse_args()

    config = apply_overrides(load_config(), args.overrides)
    out_dir = output_dir(config, create=False)
    sw_dir = os.path.join(out_dir, 'time_sweep')
    os.makedirs(sw_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'model_config.json')) as f:
        model = SpinGlass2D.from_config_dict(json.load(f))

    ts = config['time_sweep']
    chunk = ts['chunk_delta_t']
    dt, order, cutoff = ts['dt'], ts['trotter_order'], ts['cutoff']
    snapshots = sorted(set(args.snapshots))
    k_max = snapshots[-1]

    n_steps = int(round(chunk / dt))
    block = build_trotter_gate_sequence(
        model.num_qubits, model.substep_bonds, model.J, model.h,
        dt=dt, n_steps=n_steps, order=order,
    )
    print(f"  chunk={chunk}, dt={dt}, order={order}, {len(block)} gates/chunk")
    print(f"  snapshots k={snapshots} (T={[k*chunk for k in snapshots]})")
    print(f"  workers={args.workers}, mem_floor={args.mem_floor_gb}GB, "
          f"available_now={mem_available_gb():.1f}GB")

    obs_list = ([('X', q) for q in range(model.num_qubits)] +
                [('Z', q) for q in range(model.num_qubits)])
    tasks = [(pauli, q, model.num_qubits, block, k_max, snapshots, cutoff, sw_dir)
             for pauli, q in obs_list]

    ctx = mp.get_context('fork')
    pool = ctx.Pool(processes=args.workers)
    async_result = pool.map_async(worker, tasks)
    pool.close()

    t0 = time.time()
    aborted = False
    while not async_result.ready():
        time.sleep(args.poll_seconds)
        avail = mem_available_gb()
        elapsed = time.time() - t0
        print(f"  [{elapsed:7.0f}s] MemAvailable={avail:.2f}GB", flush=True)
        if avail < args.mem_floor_gb:
            print(f"  MEMORY FLOOR HIT ({avail:.2f}GB < {args.mem_floor_gb}GB) "
                  f"-- terminating pool, keeping whatever is on disk", flush=True)
            pool.terminate()
            pool.join()
            aborted = True
            break

    if not aborted:
        results = async_result.get()
        for pauli, q, done, elapsed in results:
            print(f"  [{pauli}_{q}] finished all requested k in {elapsed:.1f}s: {done}")
        pool.join()

    print("\n  Assembling targets_k<k>.json from disk ...")
    assemble(sw_dir, k_max, snapshots, obs_list, chunk, dt, order, cutoff)
    print("  ABORTED (memory floor)" if aborted else "  COMPLETE")


if __name__ == '__main__':
    main()
