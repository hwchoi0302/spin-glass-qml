"""Sorted-array Pauli propagation: the vectorised (and GPU-portable) engine.

propagation_numba.py's numba.typed.Dict is faster than a Python dict (JIT
compiles the per-term Python loop) but is still fundamentally a hashmap
walked one term at a time. docs/issues/03-engine-performance.md is explicit
that GPU only becomes meaningful after this step: representing an SPO as two
*sorted, parallel arrays* (keys, coeffs) and applying a gate as whole-array
numpy operations (union, searchsorted, boolean masking) instead of a
per-term loop -- a data-parallel kernel with no Python-level loop at all.

The array API used here (union1d, searchsorted, where, boolean indexing) is
close enough between numpy and cupy that this same code runs on the GPU by
swapping which module `xp` is bound to -- see propagation_gpu.py, which
imports this module's *_xp functions with xp=cupy. That is the actual
apples-to-apples comparison the CPU-vs-GPU question needs: same algorithm,
same code, different array backend.

Derivation of the per-key update rule (why this can be a whole-array
operation with no term-by-term pairing/dedup logic, unlike the dict
engines): for the anti-commuting branch, every key P in the union of
{old anti-commuting keys} and {their partners P^flip} gets

    new_val(P) = cos(theta) * old(P) + sign(P) * sin(theta) * old(partner(P))

computed independently, using *old* (pre-gate) values throughout. Applying
this formula to P and to its partner R = partner(P) independently
reproduces exactly propagation.py's paired update (new_a_P, new_a_R) --
sign(R) = -sign(P) falls out automatically because R has the opposite
generator eigenvalue at the flipped position. See the module docstring
derivation retained in propagation_packed.py for the underlying (x,z) bit
algebra itself; this module only changes *how* that algebra is applied
(whole-array vs. per-term).
"""

from typing import Optional, Tuple

import numpy as np

MASK32 = np.uint64(0xFFFFFFFF)


def to_sorted_arrays(packed: dict, xp=np):
    """{(x,z): coeff} -> (sorted keys array, aligned coeffs array)."""
    if not packed:
        return xp.zeros(0, dtype=xp.uint64), xp.zeros(0, dtype=xp.float64)
    items = sorted(packed.items())
    keys = xp.asarray([np.uint64(x) | (np.uint64(z) << np.uint64(32))
                       for (x, z), _ in items], dtype=xp.uint64)
    coeffs = xp.asarray([c for _, c in items], dtype=xp.float64)
    return keys, coeffs


def from_sorted_arrays(keys, coeffs, xp=np) -> dict:
    keys_h = keys.get() if xp is not np else keys
    coeffs_h = coeffs.get() if xp is not np else coeffs
    out = {}
    for k, c in zip(keys_h.tolist(), coeffs_h.tolist()):
        k = np.uint64(k)
        x = int(k & MASK32)
        z = int((k >> np.uint64(32)) & MASK32)
        out[(x, z)] = c
    return out


def _lookup(keys, coeffs, query_keys, xp):
    """old(query_keys), 0.0 where query_keys is absent from `keys`."""
    n = keys.shape[0]
    if n == 0:
        return xp.zeros_like(query_keys, dtype=xp.float64)
    pos = xp.searchsorted(keys, query_keys)
    pos_c = xp.minimum(pos, n - 1)
    found = (pos < n) & (keys[pos_c] == query_keys)
    return xp.where(found, coeffs[pos_c], 0.0)


def apply_rx_sorted(keys, coeffs, qubit: int, theta: float, thresh: float, xp=np):
    if keys.shape[0] == 0:
        return keys, coeffs
    kbit = xp.uint64(1) << xp.uint64(qubit)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    z = (keys >> xp.uint64(32)) & MASK32
    x = keys & MASK32
    anti = (z & kbit) != xp.uint64(0)

    commute_keys, commute_coeffs = keys[~anti], coeffs[~anti]

    anti_keys = keys[anti]
    partner_keys = anti_keys ^ kbit  # flip x-bit only (low 32 bits)
    union_keys = xp.union1d(anti_keys, partner_keys)

    old_self = _lookup(keys, coeffs, union_keys, xp)
    partner_of_union = union_keys ^ kbit
    old_partner = _lookup(keys, coeffs, partner_of_union, xp)
    sign = xp.where((union_keys & kbit) != xp.uint64(0), 1.0, -1.0)
    new_anti_vals = cos_t * old_self + sign * sin_t * old_partner

    out_keys = xp.concatenate([commute_keys, union_keys])
    out_vals = xp.concatenate([commute_coeffs, new_anti_vals])

    keep = xp.abs(out_vals) > thresh
    out_keys, out_vals = out_keys[keep], out_vals[keep]
    order = xp.argsort(out_keys)
    return out_keys[order], out_vals[order]


def apply_rzz_sorted(keys, coeffs, qi: int, qj: int, theta: float, thresh: float, xp=np):
    if keys.shape[0] == 0:
        return keys, coeffs
    ibit = xp.uint64(1) << xp.uint64(qi)
    jbit = xp.uint64(1) << xp.uint64(qj)
    m32 = ibit | jbit
    m_hi = m32 << xp.uint64(32)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    x = keys & MASK32
    z = (keys >> xp.uint64(32)) & MASK32
    anti_i = (x & ibit) != xp.uint64(0)
    anti_j = (x & jbit) != xp.uint64(0)
    anti = anti_i != anti_j

    commute_keys, commute_coeffs = keys[~anti], coeffs[~anti]

    anti_keys = keys[anti]
    partner_keys = anti_keys ^ m_hi  # flip both z-bits (high 32 bits)
    union_keys = xp.union1d(anti_keys, partner_keys)

    old_self = _lookup(keys, coeffs, union_keys, xp)
    partner_of_union = union_keys ^ m_hi
    old_partner = _lookup(keys, coeffs, partner_of_union, xp)

    ux = union_keys & MASK32
    uz = (union_keys >> xp.uint64(32)) & MASK32
    u_anti_i = (ux & ibit) != xp.uint64(0)
    k_anti_bit = xp.where(u_anti_i, ibit, jbit)
    sign = xp.where((uz & k_anti_bit) != xp.uint64(0), -1.0, 1.0)  # X->+1, Y->-1

    new_anti_vals = cos_t * old_self + sign * sin_t * old_partner

    out_keys = xp.concatenate([commute_keys, union_keys])
    out_vals = xp.concatenate([commute_coeffs, new_anti_vals])

    keep = xp.abs(out_vals) > thresh
    out_keys, out_vals = out_keys[keep], out_vals[keep]
    order = xp.argsort(out_keys)
    return out_keys[order], out_vals[order]


def propagate_forward_sorted(keys, coeffs, gate_sequence, delta: float = 0.0, xp=np,
                             stats: Optional["object"] = None):
    """Same reverse-order walk as propagation.propagate_forward (hard rule 3)."""
    thresh = max(delta, 1e-15)
    n_before = None
    for gate in reversed(gate_sequence):
        if stats is not None:
            n_before = float(xp.sum(coeffs ** 2)) if keys.shape[0] else 0.0
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            keys, coeffs = apply_rx_sorted(keys, coeffs, q, theta, thresh, xp)
        elif gate[0] == 'rzz':
            _, qi, qj, theta, _ = gate
            keys, coeffs = apply_rzz_sorted(keys, coeffs, qi, qj, theta, thresh, xp)
        else:
            continue
        if stats is not None:
            # Norm is exactly preserved by an exact rotation; whatever norm
            # was lost this gate was discarded by the threshold.
            n_after = float(xp.sum(coeffs ** 2)) if keys.shape[0] else 0.0
            stats.sum_sq += max(0.0, n_before - n_after)
            stats.n_gates += 1
    return keys, coeffs


def self_check(seed: int = 0, n: int = 4, n_gates: int = 60, xp=np) -> None:
    """Compare directly against the string-dict oracle, term-for-term."""
    from .propagation import apply_rx_forward, apply_rzz_forward
    from .propagation_packed import label_to_xz, xz_to_label

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

    x0, z0 = label_to_xz(label0)
    keys, coeffs = to_sorted_arrays({(x0, z0): 1.0}, xp)
    keys, coeffs = propagate_forward_sorted(keys, coeffs, gates, 1e-12, xp)
    result = from_sorted_arrays(keys, coeffs, xp)
    result_labels = {xz_to_label(x, z, n): c for (x, z), c in result.items()}

    keys_diff = set(spo_str) ^ set(result_labels)
    max_dev = max(
        (abs(spo_str.get(k, 0.0) - result_labels.get(k, 0.0))
         for k in set(spo_str) | set(result_labels)), default=0.0)
    backend = 'numpy' if xp is np else getattr(xp, '__name__', str(xp))
    print(f"  [{backend}] n={n}, {n_gates} gates: string={len(spo_str)} terms, "
          f"sorted={len(result_labels)} terms, "
          f"symmetric-diff={len(keys_diff)}, max coeff dev={max_dev:.3e}")
    assert not keys_diff, f"sorted engine produced different terms: {keys_diff}"
    assert max_dev < 1e-9, "sorted engine coefficients diverge from string engine"


if __name__ == '__main__':
    print("propagation_sorted self-check vs the string-dict oracle:")
    for seed in range(5):
        self_check(seed=seed, n=4, n_gates=80)
    for seed in range(3):
        self_check(seed=seed, n=6, n_gates=150)
    print("ALL SORTED-ARRAY EQUIVALENCE CHECKS PASSED")
