"""Bit-packed Pauli propagation engine (BP-PPS Appendix C representation).

CLAUDE.md hard rule 4: propagation.py's string-keyed dict is the oracle
every other engine is checked against, and bitpacking is *required* from 7x7
up because the string representation stops fitting there. Below 7x7 any
engine may be used once it has been proven term-for-term equal to the oracle
at 4x4 -- which is what self_check() below and TEST 17 do, while
propagation.py is still small enough to trust by inspection.

This particular module is a negative result and is kept as the oracle for
the bit algebra rather than for speed: a (x, z) tuple key in pure Python
carries ~28 bytes of object overhead regardless of its logical bit width, so
it is no smaller and no faster than the strings it replaced. The
representation only pays off once the keys live in an unboxed container --
propagation_numba.py (typed dict) or propagation_sorted.py (parallel
arrays), both of which reuse the algebra derived here.

Representation: a Pauli string P in {I,X,Y,Z}^n becomes a pair of integer
bitmasks (x, z), one bit per qubit:

    I = (0,0)   X = (1,0)   Z = (0,1)   Y = (1,1)

Python's arbitrary-precision int makes this representation-agnostic in n --
the same code handles 16 or 100 qubits without a UInt64/UInt128 split. A term
is keyed by the tuple (x, z) instead of a 16-character string: for n=16 that
is roughly 56 bytes (two small-int objects + tuple) against ~89 bytes for an
interned-looking 16-char str, and the real win, per docs/issues/03-engine-
performance.md, is that commutation and multiplication become popcount/XOR
instead of character comparisons.

Gate action (re-derived from, and checked against, apply_rx_forward /
apply_rzz_forward in propagation.py -- NOT from the pseudocode summary in
docs/benchmark_plan.md, which has the RX sign backwards):

    RX on qubit k (generator X_k):
        commutes iff z_k == 0 (P is I or X)
        anti-commuting partner R = (x ^ (1<<k), z)       [Y <-> Z at k]
        sign = +1 if x_k == 1 (P is Y), else -1 (P is Z)

    RZZ on (i, j) (generator Z_i Z_j):
        commutes iff (x_i, x_j) agree (both I/Z or both X/Y at i,j)
        anti-commuting partner R = (x, z ^ ((1<<i)|(1<<j)))
        sign = +1 if the X/Y-carrying position (i or j) has z==0 (is X),
               -1 if it has z==1 (is Y)

Both are proven equal to the string engine's output for random circuits in
self_check() below (exact term-for-term match, not just close).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# A packed SPO term key: (x_mask, z_mask), both Python ints (n bits used).
PackedSPO = Dict[Tuple[int, int], float]

# This module keeps x and z as separate arbitrary-precision Python ints, so it
# has no size limit. Its two descendants do: propagation_numba.py and
# propagation_sorted.py both fold the pair into ONE uint64 as
# ``x | (z << 32)``, which is lossless only while both masks fit in 32 bits.
PACKED_MAX_QUBITS = 32


def check_gate_sequence_packable(gate_sequence, engine: str) -> int:
    """Refuse a circuit the single-uint64 key cannot represent.

    Above 32 qubits ``x | (z << 32)`` is not injective: x needs bit 32, which
    is where z's bit 0 lands. ``Z_0`` (x=0, z=1) and ``X_32`` (x=2**32, z=0)
    both pack to 2**32, and the engines write ``d[key] = value``, so one term
    silently overwrites the other -- the term count comes out low and the
    surviving coefficient belongs to the wrong Pauli string. No exception, no
    warning, and the numbers still look plausible.

    Checked here, on the gate sequence, rather than at key construction. A key
    built at the entry point is a single-site observable with one bit set and
    always passes; the collision only appears later, once propagation has
    spread the support past qubit 31, and by then the numba engine is building
    every new key inside its ``@njit`` loop with raw bit ops that never call
    make_key(). The circuit's own qubit indices are the one quantity available
    *before* any of that runs.

    This is not hypothetical. results/4x4/../lightcone_production_delta.json's
    L=6 row (36 qubits) was produced without this check; measured afterwards
    with the string engine, that propagation puts support on qubit 33, so the
    row is corrupt. See docs/issues/03-engine-performance.md.

    Args:
        gate_sequence: Gates in circuit order, ``('rx', q, theta, pidx)`` or
            ``('rzz', qi, qj, theta, pidx)``.
        engine: Name used in the error message.

    Returns:
        The highest qubit index the sequence touches.

    Raises:
        ValueError: if that index needs bit 32 or above.
    """
    hi = -1
    for gate in gate_sequence:
        if gate[0] == 'rx':
            if gate[1] > hi:
                hi = gate[1]
        elif gate[0] == 'rzz':
            if gate[1] > hi:
                hi = gate[1]
            if gate[2] > hi:
                hi = gate[2]
    if hi >= PACKED_MAX_QUBITS:
        raise ValueError(
            f"{engine} packs (x, z) into one uint64 as x | (z << 32), which "
            f"holds at most {PACKED_MAX_QUBITS} qubits, but this circuit "
            f"touches qubit {hi}. Distinct Pauli strings would collide on one "
            f"key and silently overwrite each other. Use the string engine "
            f"(propagation.py), or widen the key to two uint64 per mask "
            f"first. 5x5 (25 qubits) fits; 6x6 (36) and 7x7 (49) do not.")
    return hi


def label_to_xz(label: str) -> Tuple[int, int]:
    """String Pauli label (propagation.py's convention) -> (x, z) bitmasks."""
    x = z = 0
    for k, c in enumerate(label):
        if c == 'X':
            x |= 1 << k
        elif c == 'Z':
            z |= 1 << k
        elif c == 'Y':
            x |= 1 << k
            z |= 1 << k
    return x, z


def xz_to_label(x: int, z: int, n: int) -> str:
    """(x, z) bitmasks -> string Pauli label, inverse of label_to_xz."""
    chars = []
    for k in range(n):
        xb, zb = (x >> k) & 1, (z >> k) & 1
        if not xb and not zb:
            chars.append('I')
        elif xb and not zb:
            chars.append('X')
        elif not xb and zb:
            chars.append('Z')
        else:
            chars.append('Y')
    return ''.join(chars)


def spo_to_packed(spo: Dict[str, float]) -> PackedSPO:
    return {label_to_xz(label): a for label, a in spo.items()}


def packed_to_spo(packed: PackedSPO, n: int) -> Dict[str, float]:
    return {xz_to_label(x, z, n): a for (x, z), a in packed.items()}


def apply_rx_forward_packed(coeffs: PackedSPO, qubit: int, theta: float,
                            delta: float = 0.0,
                            stats: Optional["object"] = None) -> PackedSPO:
    """Bit-packed equivalent of propagation.apply_rx_forward."""
    new_coeffs: PackedSPO = {}
    processed = set()
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    thresh = max(delta, 1e-15)
    kbit = 1 << qubit

    for (x, z), a_P in coeffs.items():
        if (x, z) in processed:
            continue
        if not (z & kbit):
            # Commuting (I or X at k): still subject to the threshold.
            if abs(a_P) > thresh:
                new_coeffs[(x, z)] = a_P
            elif stats is not None:
                stats.discard(a_P)
        else:
            # Anti-commuting (Y or Z at k): R flips the x-bit at k.
            R = (x ^ kbit, z)
            sign = 1 if (x & kbit) else -1
            a_R = coeffs.get(R, 0.0)

            new_a_P = cos_t * a_P + sign * sin_t * a_R
            new_a_R = -sign * sin_t * a_P + cos_t * a_R

            if abs(new_a_P) > thresh:
                new_coeffs[(x, z)] = new_a_P
            elif stats is not None:
                stats.discard(new_a_P)
            if abs(new_a_R) > thresh:
                new_coeffs[R] = new_a_R
            elif stats is not None:
                stats.discard(new_a_R)

            processed.add((x, z))
            processed.add(R)

    if stats is not None:
        stats.n_gates += 1
    return new_coeffs


def apply_rzz_forward_packed(coeffs: PackedSPO, qi: int, qj: int, theta: float,
                             delta: float = 0.0,
                             stats: Optional["object"] = None) -> PackedSPO:
    """Bit-packed equivalent of propagation.apply_rzz_forward."""
    new_coeffs: PackedSPO = {}
    processed = set()
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    thresh = max(delta, 1e-15)
    ibit, jbit = 1 << qi, 1 << qj
    m = ibit | jbit

    for (x, z), a_P in coeffs.items():
        if (x, z) in processed:
            continue
        anti_i = bool(x & ibit)
        anti_j = bool(x & jbit)

        if anti_i == anti_j:
            # Both commute (I/Z at both) or both anti-commute (X/Y at both).
            if abs(a_P) > thresh:
                new_coeffs[(x, z)] = a_P
            elif stats is not None:
                stats.discard(a_P)
        else:
            R = (x, z ^ m)
            k_anti = qi if anti_i else qj
            sign = -1 if (z & (1 << k_anti)) else 1  # X -> +1, Y -> -1

            a_R = coeffs.get(R, 0.0)
            new_a_P = cos_t * a_P + sign * sin_t * a_R
            new_a_R = -sign * sin_t * a_P + cos_t * a_R

            if abs(new_a_P) > thresh:
                new_coeffs[(x, z)] = new_a_P
            elif stats is not None:
                stats.discard(new_a_P)
            if abs(new_a_R) > thresh:
                new_coeffs[R] = new_a_R
            elif stats is not None:
                stats.discard(new_a_R)

            processed.add((x, z))
            processed.add(R)

    if stats is not None:
        stats.n_gates += 1
    return new_coeffs


def propagate_forward_packed(init_spo: PackedSPO, gate_sequence: List[tuple],
                             delta: float = 0.0,
                             stats: Optional["object"] = None) -> PackedSPO:
    """Bit-packed equivalent of propagation.propagate_forward.

    Same reverse-order walk (CLAUDE.md hard rule 3): the observable sits at
    the circuit output, so this conjugates by the last gate first.
    """
    spo = dict(init_spo)
    for gate in reversed(gate_sequence):
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            spo = apply_rx_forward_packed(spo, q, theta, delta, stats)
        elif gate[0] == 'rzz':
            _, qi, qj, theta, _ = gate
            spo = apply_rzz_forward_packed(spo, qi, qj, theta, delta, stats)
    return spo


def self_check(seed: int = 0, n: int = 4, n_gates: int = 60) -> None:
    """Random small circuit: packed engine must match the string engine
    term-for-term, not just approximately -- these are two representations
    of the *same* deterministic computation.
    """
    from .propagation import TruncationStats, apply_rx_forward, apply_rzz_forward

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
    stats_str = TruncationStats()
    for gate in reversed(gates):
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            spo_str = apply_rx_forward(spo_str, q, theta, 1e-12, stats_str)
        else:
            _, qi, qj, theta, _ = gate
            spo_str = apply_rzz_forward(spo_str, qi, qj, theta, 1e-12, stats_str)

    packed0 = spo_to_packed({label0: 1.0})
    stats_packed = TruncationStats()
    spo_packed = propagate_forward_packed(packed0, gates, 1e-12, stats_packed)
    spo_packed_as_labels = packed_to_spo(spo_packed, n)

    keys_diff = set(spo_str) ^ set(spo_packed_as_labels)
    max_dev = max(
        (abs(spo_str.get(k, 0.0) - spo_packed_as_labels.get(k, 0.0))
         for k in set(spo_str) | set(spo_packed_as_labels)), default=0.0)

    print(f"  n={n}, {n_gates} gates: string={len(spo_str)} terms, "
          f"packed={len(spo_packed_as_labels)} terms, "
          f"symmetric-diff keys={len(keys_diff)}, max coeff dev={max_dev:.3e}, "
          f"eps_emp match={abs(stats_str.error_estimate - stats_packed.error_estimate):.3e}")
    assert not keys_diff, f"packed engine produced different terms: {keys_diff}"
    assert max_dev < 1e-9, "packed engine coefficients diverge from string engine"


if __name__ == '__main__':
    print("propagation_packed self-check vs the string-dict oracle:")
    for seed in range(5):
        self_check(seed=seed, n=4, n_gates=80)
    for seed in range(3):
        self_check(seed=seed, n=6, n_gates=150)
    print("ALL PACKED-ENGINE EQUIVALENCE CHECKS PASSED")
