"""Sorted-array Pauli propagation: the vectorised (and GPU-portable) engine.

propagation_numba.py's numba.typed.Dict is faster than a Python dict (JIT
compiles the per-term Python loop) but is still fundamentally a hashmap
walked one term at a time. docs/issues/03-engine-performance.md is explicit
that GPU only becomes meaningful after this step: representing an SPO as two
*sorted, parallel arrays* (keys, coeffs) and applying a gate as whole-array
numpy operations (union, searchsorted, boolean masking) instead of a
per-term loop -- a data-parallel kernel with no Python-level loop at all.

The array API used here (sort, searchsorted, where, boolean indexing) is
close enough between numpy and cupy that this same code runs on the GPU by
passing `xp=cupy` to any function in this module -- there is no separate GPU
module, and TEST 20 in scripts/00_validate_small.py drives exactly this code
with xp=cupy. That is the apples-to-apples comparison the CPU-vs-GPU
question needs: same algorithm, same code, different array backend.

The 2026-09-03 answer to that question is *no* for this kernel; the reasons
and the numbers are in docs/issues/03-engine-performance.md. Two things to
know before re-running the benchmark: pass `stats=None`, because the
truncation accounting does `float(xp.sum(...))` twice per gate and each is a
device-to-host sync that serialises the whole pipeline; and note that a gate
here is ~35 whole-array calls, so cupy's per-call dispatch overhead is
multiplied by 35 * n_gates before any arithmetic happens.

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
    """{(x,z): coeff} -> (sorted keys array, aligned coeffs array).

    Sorted by the *packed key*, not by the (x, z) tuple. Those two orders are
    different -- the key puts z in the high 32 bits, so it orders by z first,
    while tuple order goes by x first -- and getting it wrong returns an array
    that is not sorted at all, which silently breaks every searchsorted in
    this module. The only in-tree caller used to be self_check() with a single
    term, where any order is sorted, so nothing caught it.
    """
    if not packed:
        return xp.zeros(0, dtype=xp.uint64), xp.zeros(0, dtype=xp.float64)
    items = sorted(
        ((np.uint64(x) | (np.uint64(z) << np.uint64(32)), c)
         for (x, z), c in packed.items()),
        key=lambda kc: int(kc[0]))
    keys = xp.asarray([k for k, _ in items], dtype=xp.uint64)
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


def _union_sorted(a, b, xp=np):
    """xp.union1d(a, b) for inputs that are each already unique.

    Not a micro-optimisation: numpy 2.5's np.unique -- which union1d calls --
    is ~70x slower than the sort-and-mask it is meant to be (2.4M uint64:
    1696 ms vs 24 ms, measured). union1d is called once per gate here, and it
    dominated the whole engine: one RX gate over 1.2M terms cost 855 ms with
    it and 175 ms without. The result is bit-identical, not approximate.
    """
    s = xp.sort(xp.concatenate((a, b)))
    if s.size == 0:
        return s
    keep = xp.ones(s.size, dtype=bool)
    keep[1:] = s[1:] != s[:-1]
    return s[keep]


def _argsort(a, xp=np):
    """Stable sort where the backend has one: the array being sorted is two
    already-sorted runs concatenated, which timsort merges in O(N) instead of
    re-sorting. cupy's argsort is stable by default and rejects `kind`."""
    return xp.argsort(a, kind='stable') if xp is np else xp.argsort(a)


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
    union_keys = _union_sorted(anti_keys, partner_keys, xp)

    old_self = _lookup(keys, coeffs, union_keys, xp)
    partner_of_union = union_keys ^ kbit
    old_partner = _lookup(keys, coeffs, partner_of_union, xp)
    sign = xp.where((union_keys & kbit) != xp.uint64(0), 1.0, -1.0)
    new_anti_vals = cos_t * old_self + sign * sin_t * old_partner

    out_keys = xp.concatenate([commute_keys, union_keys])
    out_vals = xp.concatenate([commute_coeffs, new_anti_vals])

    keep = xp.abs(out_vals) > thresh
    out_keys, out_vals = out_keys[keep], out_vals[keep]
    order = _argsort(out_keys, xp)
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
    union_keys = _union_sorted(anti_keys, partner_keys, xp)

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
    order = _argsort(out_keys, xp)
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


# ===========================================================================
# Backward pass
# ===========================================================================
#
# propagate_backward is what BP-PPS training actually spends its time in --
# measured at the production L=3 angles with delta=1e-6, one gradient
# evaluation over the 32 observables was 40.1 s forward against 84.1 s
# backward -- and until now it existed only in propagation.py's string-dict
# form. Porting the forward pass alone therefore capped the achievable speedup
# at about 1.5x; this is the other two thirds.
#
# The backward pass carries a second array beside the coefficients: the
# co-state lam_P = dL/da_P, which BP-PPS's Eq. 20 rotates by the same U(-theta)
# as the coefficients. The key domain is shared, which is what lets both live
# on one sorted key array: propagate_backward starts from adjoint = seed with
# seed a subset of the evolved SPO, and every step writes a lam entry only
# where the corresponding coefficient survived truncation, so
# adjoint.keys() is a subset of coeffs.keys() for the whole walk. The string
# engine's union over `coeffs.keys() | adjoint.keys()` is therefore just
# coeffs.keys(), and one (keys, a, lam) triple loses nothing.
#
# Why the gradient carries a factor of one half. propagation.py accumulates
#
#     grad += sign(P) * (lam_P * a_R - lam_R * a_P)
#
# once per {P, R} pair, which it enforces with a `processed` set -- a
# sequential construct with no array form. It does not need one: R differs
# from P only at the flipped bit, so sign(R) = -sign(P), and the bracket also
# changes sign under the swap, leaving the product invariant. Summing the same
# expression over every key in the union counts each pair exactly twice, so
# half the union sum is the same number with no pairing logic at all. This is
# the same argument the module docstring makes for the forward update rule.


def _backward_gate_sorted(keys, a, lam, flip, anti, sign_of,
                          cos_t, sin_t, thresh, xp):
    """One inverse gate step shared by RX and RZZ.

    Args:
        keys: Sorted uint64 Pauli keys.
        a: Coefficients, parallel to ``keys``.
        lam: Co-state dL/da, parallel to ``keys``.
        flip: uint64 mask turning a key into its partner under this gate.
        anti: Boolean mask, True where the key anticommutes with the generator.
        sign_of: Callable mapping an array of keys to their +-1 signs.
        cos_t, sin_t: cos/sin of the gate angle.
        thresh: Truncation threshold, applied to the coefficient magnitude.
        xp: numpy or cupy.

    Returns:
        (keys, a, lam, gradient, discarded_sum_sq, n_discarded)
    """
    if keys.shape[0] == 0:
        return keys, a, lam, 0.0, 0.0, 0

    commute = ~anti
    ck, ca, cl = keys[commute], a[commute], lam[commute]

    anti_keys = keys[anti]
    if anti_keys.shape[0] == 0:
        union_keys = anti_keys
        new_a = a[anti]
        new_l = lam[anti]
        gradient = 0.0
    else:
        union_keys = _union_sorted(anti_keys, anti_keys ^ flip, xp)
        partner_keys = union_keys ^ flip

        a_self = _lookup(keys, a, union_keys, xp)
        a_part = _lookup(keys, a, partner_keys, xp)
        l_self = _lookup(keys, lam, union_keys, xp)
        l_part = _lookup(keys, lam, partner_keys, xp)
        sign = sign_of(union_keys)

        # Eq. 21, on the post-gate values, before the inverse rotation.
        # Half the union sum -- see the note above.
        gradient = 0.5 * float(xp.sum(sign * (l_self * a_part - l_part * a_self)))

        # Eqs. 11 / 20: the inverse rotation U(-theta), applied per key. Doing
        # it independently for a key and its partner reproduces propagation.py's
        # paired (new_P, new_R) update because sign(R) = -sign(P).
        new_a = cos_t * a_self - sign * sin_t * a_part
        new_l = cos_t * l_self - sign * sin_t * l_part

    out_keys = xp.concatenate([ck, union_keys])
    out_a = xp.concatenate([ca, new_a])
    out_l = xp.concatenate([cl, new_l])

    # Joint truncation: keyed on the coefficient, dropping its co-state with
    # it, exactly as propagation._backward_rx does.
    keep = xp.abs(out_a) > thresh
    dropped = out_a[~keep]
    disc_sq = float(xp.sum(dropped * dropped)) if dropped.shape[0] else 0.0
    n_disc = int(dropped.shape[0])

    out_keys, out_a, out_l = out_keys[keep], out_a[keep], out_l[keep]
    order = _argsort(out_keys, xp)
    return (out_keys[order], out_a[order], out_l[order],
            gradient, disc_sq, n_disc)


def backward_rx_sorted(keys, a, lam, qubit: int, theta: float, thresh: float,
                       xp=np):
    """Inverse RX step. See :func:`_backward_gate_sorted`."""
    kbit = xp.uint64(1) << xp.uint64(qubit)
    z = (keys >> xp.uint64(32)) & MASK32
    anti = (z & kbit) != xp.uint64(0)
    # P[k] == 'Y' (x-bit set) -> +1, P[k] == 'Z' -> -1, matching
    # propagation._backward_rx's sign.
    sign_of = lambda k: xp.where((k & kbit) != xp.uint64(0), 1.0, -1.0)
    return _backward_gate_sorted(keys, a, lam, kbit, anti, sign_of,
                                 float(np.cos(theta)), float(np.sin(theta)),
                                 thresh, xp)


def backward_rzz_sorted(keys, a, lam, qi: int, qj: int, theta: float,
                        thresh: float, xp=np):
    """Inverse RZZ step. See :func:`_backward_gate_sorted`."""
    ibit = xp.uint64(1) << xp.uint64(qi)
    jbit = xp.uint64(1) << xp.uint64(qj)
    m_hi = (ibit | jbit) << xp.uint64(32)
    x = keys & MASK32
    anti = ((x & ibit) != xp.uint64(0)) != ((x & jbit) != xp.uint64(0))

    def sign_of(k):
        kx = k & MASK32
        kz = (k >> xp.uint64(32)) & MASK32
        anti_bit = xp.where((kx & ibit) != xp.uint64(0), ibit, jbit)
        # X at the anticommuting site -> +1, Y -> -1.
        return xp.where((kz & anti_bit) != xp.uint64(0), -1.0, 1.0)

    return _backward_gate_sorted(keys, a, lam, m_hi, anti, sign_of,
                                 float(np.cos(theta)), float(np.sin(theta)),
                                 thresh, xp)


def propagate_backward_sorted(keys, a, lam, gate_sequence, n_params: int,
                              delta: float = 0.0, xp=np,
                              stats: Optional["object"] = None):
    """Gradients for every trainable parameter, on the sorted-array engine.

    Walks the gates in **circuit order**, the opposite of
    :func:`propagate_forward_sorted` -- hard rule 3 in CLAUDE.md, and a real
    bug once (commit e9c3b50).

    Args:
        keys, a: Evolved SPO from the forward pass, sorted.
        lam: Gradient seed dL/da on the same key domain (zero where the seed
            has no entry).
        gate_sequence: The same sequence the forward pass used, circuit order.
        n_params: Length of the gradient vector.
        delta: Truncation threshold for the backward reconstruction.
        xp: numpy or cupy.
        stats: Optional TruncationStats to accumulate discarded weight into.

    Returns:
        numpy array of shape (n_params,).
    """
    thresh = max(delta, 1e-15)
    gradients = np.zeros(n_params)

    for gate in gate_sequence:
        if gate[0] == 'rx':
            _, q, theta, pidx = gate
            keys, a, lam, grad, dsq, ndisc = backward_rx_sorted(
                keys, a, lam, q, theta, thresh, xp)
        elif gate[0] == 'rzz':
            _, qi, qj, theta, pidx = gate
            keys, a, lam, grad, dsq, ndisc = backward_rzz_sorted(
                keys, a, lam, qi, qj, theta, thresh, xp)
        else:
            continue

        if pidx >= 0:
            gradients[pidx] += grad
        if stats is not None:
            stats.sum_sq += dsq
            stats.n_discarded += ndisc
            stats.n_gates += 1

    return gradients
