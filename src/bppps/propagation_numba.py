"""Numba-JIT'd Pauli propagation on the bit-packed representation.

propagation_packed.py proved the (x,z) bit algebra is correct but, as a plain
Python dict of (int,int) tuples, is *slower* than the string engine -- every
CPython object (even a small int) carries ~28 bytes of interpreter overhead,
so packing bits into Python ints buys nothing without also getting the data
out of Python objects entirely (docs/issues/03-engine-performance.md).

This module keeps the exact same, already-validated bit algebra but stores
each term as a single uint64 key (x in the low 32 bits, z in the high 32 --
valid for n <= 32, comfortably covering 4x4 now and 7x7 later; the 100-qubit
case needs two uint64 per mask and is out of scope for this session) in a
numba.typed.Dict, with the per-gate hot loop JIT-compiled. The outer
gate-sequence walk stays plain Python (the string/packed engines are
structured the same way) -- only the O(n_terms) inner loop, where all the
time actually goes, is compiled.

Correctness is re-validated here rather than assumed from propagation_packed:
self_check() compares this engine's output to the string-dict oracle
directly, term-for-term, the same way TEST 17 does for the pure-Python
packed engine.
"""

from typing import List, Optional

import numpy as np
from numba import njit, types
from numba.typed import Dict

U64 = types.uint64
F64 = types.float64
MASK32 = np.uint64(0xFFFFFFFF)


def make_key(x: int, z: int) -> np.uint64:
    return np.uint64(x) | (np.uint64(z) << np.uint64(32))


def split_key(key: np.uint64):
    x = int(key & MASK32)
    z = int((key >> np.uint64(32)) & MASK32)
    return x, z


def empty_dict():
    return Dict.empty(key_type=U64, value_type=F64)


def to_numba_dict(packed: dict) -> "Dict":
    """{(x,z): coeff} -> numba typed Dict[uint64, float64]."""
    d = empty_dict()
    for (x, z), c in packed.items():
        d[make_key(x, z)] = c
    return d


def from_numba_dict(d) -> dict:
    return {split_key(k): v for k, v in d.items()}


@njit(cache=True)
def _apply_rx(coeffs, qubit, theta, thresh):
    new_coeffs = Dict.empty(key_type=U64, value_type=F64)
    processed = Dict.empty(key_type=U64, value_type=types.boolean)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    kbit = np.uint64(1) << np.uint64(qubit)
    discarded_sq = 0.0

    for key in coeffs:
        if key in processed:
            continue
        a_P = coeffs[key]
        x = key & MASK32
        z = (key >> np.uint64(32)) & MASK32

        if (z & kbit) == np.uint64(0):
            if abs(a_P) > thresh:
                new_coeffs[key] = a_P
            else:
                discarded_sq += a_P * a_P
        else:
            Rkey = (x ^ kbit) | (z << np.uint64(32))
            sign = 1.0 if (x & kbit) != np.uint64(0) else -1.0
            a_R = coeffs[Rkey] if Rkey in coeffs else 0.0

            new_a_P = cos_t * a_P + sign * sin_t * a_R
            new_a_R = -sign * sin_t * a_P + cos_t * a_R

            if abs(new_a_P) > thresh:
                new_coeffs[key] = new_a_P
            else:
                discarded_sq += new_a_P * new_a_P
            if abs(new_a_R) > thresh:
                new_coeffs[Rkey] = new_a_R
            else:
                discarded_sq += new_a_R * new_a_R
            processed[key] = True
            processed[Rkey] = True
    return new_coeffs, discarded_sq


@njit(cache=True)
def _apply_rzz(coeffs, qi, qj, theta, thresh):
    new_coeffs = Dict.empty(key_type=U64, value_type=F64)
    processed = Dict.empty(key_type=U64, value_type=types.boolean)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    ibit = np.uint64(1) << np.uint64(qi)
    jbit = np.uint64(1) << np.uint64(qj)
    m = ibit | jbit
    discarded_sq = 0.0

    for key in coeffs:
        if key in processed:
            continue
        a_P = coeffs[key]
        x = key & MASK32
        z = (key >> np.uint64(32)) & MASK32
        anti_i = (x & ibit) != np.uint64(0)
        anti_j = (x & jbit) != np.uint64(0)

        if anti_i == anti_j:
            if abs(a_P) > thresh:
                new_coeffs[key] = a_P
            else:
                discarded_sq += a_P * a_P
        else:
            Rz = z ^ m
            Rkey = x | (Rz << np.uint64(32))
            k_anti_bit = ibit if anti_i else jbit
            sign = -1.0 if (z & k_anti_bit) != np.uint64(0) else 1.0

            a_R = coeffs[Rkey] if Rkey in coeffs else 0.0
            new_a_P = cos_t * a_P + sign * sin_t * a_R
            new_a_R = -sign * sin_t * a_P + cos_t * a_R

            if abs(new_a_P) > thresh:
                new_coeffs[key] = new_a_P
            else:
                discarded_sq += new_a_P * new_a_P
            if abs(new_a_R) > thresh:
                new_coeffs[Rkey] = new_a_R
            else:
                discarded_sq += new_a_R * new_a_R
            processed[key] = True
            processed[Rkey] = True
    return new_coeffs, discarded_sq


def propagate_forward_numba(init_coeffs, gate_sequence: List[tuple],
                            delta: float = 0.0, stats: Optional["object"] = None):
    """Same reverse-order walk as propagation.propagate_forward (hard rule 3).

    `init_coeffs` and the return value are numba typed Dict[uint64,float64].
    If `stats` is a propagation.TruncationStats, it accumulates the same
    Appendix B quantity (sum of squared discarded coefficients) the string
    engine tracks, computed inside the JIT'd loop and added in here.
    """
    thresh = max(delta, 1e-15)
    coeffs = init_coeffs
    for gate in reversed(gate_sequence):
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            coeffs, disc = _apply_rx(coeffs, np.int64(q), np.float64(theta), np.float64(thresh))
        elif gate[0] == 'rzz':
            _, qi, qj, theta, _ = gate
            coeffs, disc = _apply_rzz(coeffs, np.int64(qi), np.int64(qj),
                                      np.float64(theta), np.float64(thresh))
        else:
            continue
        if stats is not None:
            stats.sum_sq += disc
            stats.n_gates += 1
    return coeffs


def self_check(seed: int = 0, n: int = 4, n_gates: int = 60) -> None:
    """Compare directly against the string-dict oracle, term-for-term."""
    from .propagation import apply_rx_forward, apply_rzz_forward

    rng = np.random.default_rng(seed)
    gates = []
    for _ in range(n_gates):
        if rng.random() < 0.5:
            q = int(rng.integers(0, n))
            gates.append(('rx', q, float(rng.uniform(-1, 1)), -1))
        else:
            qi, qj = rng.choice(n, size=2, replace=False)
            gates.append(('rzz', int(qi), int(qj), float(rng.uniform(-1, 1)), -1))

    label0 = 'X' + 'I' * (n - 1)
    spo_str = {label0: 1.0}
    for gate in reversed(gates):
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            spo_str = apply_rx_forward(spo_str, q, theta, 1e-12, None)
        else:
            _, qi, qj, theta, _ = gate
            spo_str = apply_rzz_forward(spo_str, qi, qj, theta, 1e-12, None)

    from .propagation_packed import label_to_xz
    x0, z0 = label_to_xz(label0)
    init = empty_dict()
    init[make_key(x0, z0)] = 1.0
    result = propagate_forward_numba(init, gates, 1e-12)
    result_packed = from_numba_dict(result)

    from .propagation_packed import xz_to_label
    result_labels = {xz_to_label(x, z, n): c for (x, z), c in result_packed.items()}

    keys_diff = set(spo_str) ^ set(result_labels)
    max_dev = max(
        (abs(spo_str.get(k, 0.0) - result_labels.get(k, 0.0))
         for k in set(spo_str) | set(result_labels)), default=0.0)
    print(f"  n={n}, {n_gates} gates: string={len(spo_str)} terms, "
          f"numba={len(result_labels)} terms, "
          f"symmetric-diff={len(keys_diff)}, max coeff dev={max_dev:.3e}")
    assert not keys_diff, f"numba engine produced different terms: {keys_diff}"
    assert max_dev < 1e-9, "numba engine coefficients diverge from string engine"


if __name__ == '__main__':
    print("propagation_numba self-check vs the string-dict oracle:")
    for seed in range(5):
        self_check(seed=seed, n=4, n_gates=80)
    for seed in range(3):
        self_check(seed=seed, n=6, n_gates=150)
    print("ALL NUMBA-ENGINE EQUIVALENCE CHECKS PASSED")
