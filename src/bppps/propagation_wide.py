"""Multi-word Pauli propagation: the same algebra above 32 qubits.

propagation_sorted.py packs a Pauli into one uint64 -- x in the low 32 bits,
z in the high 32 -- which is injective only to 32 qubits. 5x5 fits; 7x7 (49)
and 10x10 (100) do not, and that packing is what stands between this project
and T2. This module carries the same SPO as a **(N, 2W) uint64 matrix**: W
words of x mask followed by W words of z mask, W = ceil(n/64). Nothing else
about the physics changes.

Why this is a separate module rather than a wider dtype in the existing one:
the 64-bit kernel finds a key's old coefficient with `searchsorted`, and
searchsorted has no multi-column form. numpy has structured dtypes that would
sort lexicographically, but cupy cannot sort them, and the GPU is exactly
where the large-n workloads need to run. So the lookup has to go.

It can. Three facts, each checked against the 64-bit engine over random gates
before this module was written (docs/issues/03-engine-performance.md):

  A.  union & keys == anti_keys.
      The union is {anti keys} u {their partners}. A partner differs from its
      key only in the coordinate the gate does *not* test -- RX flips x and
      tests z, RZZ flips z and tests x -- so a partner anticommutes too. Any
      union element that is in `keys` at all is therefore in anti_keys.
      old_self is thus nonzero exactly on the anti_keys, whose coefficients we
      already hold; no lookup needed.

  B.  The union is closed under the partner flip, since (A u A^f)^f = A^f u A.

  C.  old_partner == old_self[argsort(union ^ flip)].
      With U the sorted union and F = U^f, argsort(F) is the permutation o
      with F[o] = U, i.e. U[o[k]] = partner(U[k]). Evaluating old_self at that
      permutation is exactly old_partner.

What remains is sort, argsort, XOR and equality -- all of which extend to a
matrix key by LSD radix (one stable argsort per column, least significant
first) and a row-wise `==`, on numpy and cupy alike.

The cost is roughly 2W stable argsorts per sort where the packed engine needs
one, so this is the slower engine at any n it shares with propagation_sorted.
Use it when the packed key cannot represent the lattice, not for speed:
pick_engine() below encodes that rule so callers do not have to.
"""

from typing import List, Optional

import numpy as np

from .propagation_packed import label_to_xz

#: Bits per word. The key is W words of x followed by W words of z.
WORD = 64

#: n above which even this representation needs revisiting. There is no
#: algorithmic limit here -- W just grows -- but nothing above 10x10 has been
#: measured, and a (N, 2W) matrix costs 8*2W bytes per term.
MAX_QUBITS_TESTED = 128


def n_words(n_qubits: int) -> int:
    """Words per mask. 49 qubits -> 1, 100 qubits -> 2."""
    return (n_qubits + WORD - 1) // WORD


def pick_kernel(n_qubits: int) -> str:
    """'packed' or 'wide' -- which array representation this lattice needs.

    The packed key (propagation_sorted) is the faster of the two and holds 32
    qubits; above that it is not injective and this module takes over. Callers
    should use this rather than choosing by hand, which is how a silent
    collision gets shipped.
    """
    from .propagation_sorted import MAX_QUBITS
    if n_qubits > MAX_QUBITS_TESTED:
        raise ValueError(
            f"{n_qubits} qubits exceeds the largest width tested here "
            f"({MAX_QUBITS_TESTED}); the representation generalises but "
            f"nothing above that has been checked against the oracle.")
    return 'packed' if n_qubits <= MAX_QUBITS else 'wide'


def _argsort(a, xp=np):
    """Stable argsort. cupy's is stable by default and rejects `kind`."""
    return xp.argsort(a, kind='stable') if xp is np else xp.argsort(a)


def lexsort_rows(keys, xp=np):
    """Row order sorting the key matrix lexicographically, column 0 first.

    LSD radix: stable-sort by the least significant column, then the next,
    ending on column 0. Because every pass is stable, the composition is
    stable overall -- which the dedup below relies on to keep the anti-half
    entry of a duplicated key rather than its zero-valued partner-half twin.
    """
    order = xp.arange(keys.shape[0])
    for col in range(keys.shape[1] - 1, -1, -1):
        order = order[_argsort(keys[order, col], xp)]
    return order


def _dedup_first(keys, vals, xp=np):
    """Unique rows of an already-sorted key matrix, keeping the first value."""
    if keys.shape[0] == 0:
        return keys, vals
    same = (keys[1:] == keys[:-1]).all(axis=1)
    first = xp.ones(keys.shape[0], dtype=bool)
    first[1:] = ~same
    return keys[first], vals[first]


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _row_from_xz(x: int, z: int, w: int) -> np.ndarray:
    row = np.empty(2 * w, dtype=np.uint64)
    for k in range(w):
        row[k] = np.uint64((x >> (WORD * k)) & 0xFFFFFFFFFFFFFFFF)
        row[w + k] = np.uint64((z >> (WORD * k)) & 0xFFFFFFFFFFFFFFFF)
    return row


def to_wide_arrays(spo: dict, n_qubits: int, xp=np):
    """{label: coeff} -> (sorted (N, 2W) key matrix, aligned coefficients)."""
    w = n_words(n_qubits)
    if not spo:
        return (xp.zeros((0, 2 * w), dtype=xp.uint64),
                xp.zeros(0, dtype=xp.float64))
    keys = np.empty((len(spo), 2 * w), dtype=np.uint64)
    vals = np.empty(len(spo), dtype=np.float64)
    for i, (label, c) in enumerate(spo.items()):
        x, z = label_to_xz(label)
        keys[i] = _row_from_xz(x, z, w)
        vals[i] = c
    order = lexsort_rows(keys, np)
    return xp.asarray(keys[order]), xp.asarray(vals[order])


def from_wide_arrays(keys, coeffs, n_qubits: int, xp=np) -> dict:
    """(keys, coeffs) -> {label: coeff}, vectorised over terms."""
    keys_h = keys.get() if xp is not np else np.asarray(keys)
    vals_h = coeffs.get() if xp is not np else np.asarray(coeffs)
    if keys_h.shape[0] == 0:
        return {}
    w = keys_h.shape[1] // 2
    pos = np.arange(n_qubits)
    col, off = pos // WORD, (pos % WORD).astype(np.uint64)
    xb = (keys_h[:, col] >> off) & np.uint64(1)
    zb = (keys_h[:, w + col] >> off) & np.uint64(1)
    code = (xb + 2 * zb).astype(np.intp)        # 0=I, 1=X, 2=Z, 3=Y
    table = np.frombuffer(b'IXZY', dtype=np.uint8)
    chars = np.ascontiguousarray(table[code])
    labels = chars.view(f'S{n_qubits}').ravel().astype(str).tolist()
    return dict(zip(labels, vals_h.tolist()))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _bit(keys, col: int, off: int, xp=np):
    """Boolean: is bit `off` of column `col` set, per row?"""
    return (keys[:, col] >> xp.uint64(off)) & xp.uint64(1) != xp.uint64(0)


def _flip_vector(cols, n_cols: int, xp=np):
    """XOR mask flipping the given (column, bit-offset) pairs."""
    f = np.zeros(n_cols, dtype=np.uint64)
    for col, off in cols:
        f[col] |= np.uint64(1) << np.uint64(off)
    return xp.asarray(f)


def _apply_gate(keys, coeffs, anti, flip, sign_on_union, cos_t, sin_t,
                thresh, xp):
    """One gate, shared by RX and RZZ.

    `sign_on_union` maps the union key matrix to its +-1 signs; everything
    else is the gate-independent machinery of facts A, B and C.
    """
    if keys.shape[0] == 0:
        return keys, coeffs

    commute = ~anti
    ck, cv = keys[commute], coeffs[commute]
    A, Av = keys[anti], coeffs[anti]
    if A.shape[0] == 0:
        return keys, coeffs

    # Fact A: the union's overlap with `keys` is exactly A, so stacking A
    # (carrying its coefficients) under A^flip (carrying zeros) and keeping
    # the first of each duplicate run *is* old_self on the union -- no lookup.
    cat = xp.concatenate([A, A ^ flip[None, :]])
    catv = xp.concatenate([Av, xp.zeros(A.shape[0], dtype=Av.dtype)])
    order = lexsort_rows(cat, xp)
    U, old_self = _dedup_first(cat[order], catv[order], xp)

    # Fact C: old_partner is old_self permuted by the flip's own sort order.
    old_partner = old_self[lexsort_rows(U ^ flip[None, :], xp)]

    new_vals = cos_t * old_self + sign_on_union(U) * sin_t * old_partner

    out_keys = xp.concatenate([ck, U])
    out_vals = xp.concatenate([cv, new_vals])
    keep = xp.abs(out_vals) > thresh
    out_keys, out_vals = out_keys[keep], out_vals[keep]
    order = lexsort_rows(out_keys, xp)
    return out_keys[order], out_vals[order]


def apply_rx_wide(keys, coeffs, qubit: int, theta: float, thresh: float,
                  n_qubits: int, xp=np):
    """exp(i theta X_q) in the Heisenberg picture. Anti by z, flip x."""
    w = keys.shape[1] // 2
    col, off = qubit // WORD, qubit % WORD
    anti = _bit(keys, w + col, off, xp)                 # z bit set
    flip = _flip_vector([(col, off)], 2 * w, xp)        # flip the x bit
    # sign is the union key's own x bit, exactly as in propagation_sorted
    def sign_on_union(U):
        return xp.where(_bit(U, col, off, xp), 1.0, -1.0)
    return _apply_gate(keys, coeffs, anti, flip, sign_on_union,
                       float(np.cos(theta)), float(np.sin(theta)), thresh, xp)


def apply_rzz_wide(keys, coeffs, qi: int, qj: int, theta: float, thresh: float,
                   n_qubits: int, xp=np):
    """exp(i theta Z_i Z_j) in the Heisenberg picture. Anti by x parity, flip z."""
    w = keys.shape[1] // 2
    ci, oi = qi // WORD, qi % WORD
    cj, oj = qj // WORD, qj % WORD
    xi, xj = _bit(keys, ci, oi, xp), _bit(keys, cj, oj, xp)
    anti = xi != xj
    flip = _flip_vector([(w + ci, oi), (w + cj, oj)], 2 * w, xp)

    def sign_on_union(U):
        # Exactly one of the two x bits is set on the union (flipping z cannot
        # change x). Take the z bit of *that* qubit: X -> +1, Y -> -1.
        use_i = _bit(U, ci, oi, xp)
        z_sel = xp.where(use_i, _bit(U, w + ci, oi, xp), _bit(U, w + cj, oj, xp))
        return xp.where(z_sel, -1.0, 1.0)

    return _apply_gate(keys, coeffs, anti, flip, sign_on_union,
                       float(np.cos(theta)), float(np.sin(theta)), thresh, xp)


def propagate_forward_wide(keys, coeffs, gate_sequence, n_qubits: int,
                           delta: float = 0.0, xp=np,
                           stats: Optional["object"] = None):
    """Same reverse-order walk as propagation.propagate_forward (hard rule 3).

    The truncation accounting stays on the device and carries each gate's
    output norm into the next gate, for the reasons given in
    propagation_sorted.propagate_forward_sorted.
    """
    thresh = max(delta, 1e-15)
    track = stats is not None
    if track:
        lost = xp.zeros((), dtype=xp.float64)
        n_gates = 0
        n_cur = (xp.sum(coeffs ** 2) if keys.shape[0]
                 else xp.zeros((), dtype=xp.float64))
    for gate in reversed(gate_sequence):
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            keys, coeffs = apply_rx_wide(keys, coeffs, q, theta, thresh,
                                         n_qubits, xp)
        elif gate[0] == 'rzz':
            _, qi, qj, theta, _ = gate
            keys, coeffs = apply_rzz_wide(keys, coeffs, qi, qj, theta, thresh,
                                          n_qubits, xp)
        else:
            continue
        if track:
            n_after = (xp.sum(coeffs ** 2) if keys.shape[0]
                       else xp.zeros((), dtype=xp.float64))
            lost = lost + xp.maximum(n_cur - n_after, 0.0)
            n_cur = n_after
            n_gates += 1
    if track:
        stats.sum_sq += float(lost)
        stats.n_gates += n_gates
    return keys, coeffs


def self_check(seed: int = 0, n: int = 4, n_gates: int = 60, xp=np,
               qubits: Optional[List[int]] = None) -> None:
    """Term for term against the string-dict oracle, at any n it can reach.

    The oracle has no key-width limit, so this is a real check above 32 qubits
    too -- which is the whole point of the module and the one thing the
    packed engine's tests could never cover.

    `qubits` restricts the random circuit to a chosen support. Without it, a
    circuit on 100 qubits spreads so thin that the SPO stays at a couple of
    terms and proves nothing; pointing it at a window straddling qubit 32 (the
    packed key's wall) or qubit 64 (this module's word boundary) is what
    actually exercises the representation.
    """
    from .propagation import apply_rx_forward, apply_rzz_forward

    rng = np.random.default_rng(seed)
    pool = list(range(n)) if qubits is None else list(qubits)
    gates = []
    for _ in range(n_gates):
        if rng.random() < 0.5:
            q = int(rng.choice(pool))
            gates.append(('rx', q, float(rng.uniform(-1, 1)), -1))
        else:
            qi, qj = (int(v) for v in rng.choice(pool, size=2, replace=False))
            gates.append(('rzz', qi, qj, float(rng.uniform(-1, 1)), -1))

    seed_q = pool[0]
    label0 = ''.join('X' if k == seed_q else 'I' for k in range(n))
    spo = {label0: 1.0}
    for gate in reversed(gates):
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            spo = apply_rx_forward(spo, q, theta, 1e-12, None)
        else:
            _, qi, qj, theta, _ = gate
            spo = apply_rzz_forward(spo, qi, qj, theta, 1e-12, None)

    keys, coeffs = to_wide_arrays({label0: 1.0}, n, xp)
    keys, coeffs = propagate_forward_wide(keys, coeffs, gates, n, 1e-12, xp)
    got = from_wide_arrays(keys, coeffs, n, xp)

    diff = set(spo) ^ set(got)
    max_dev = max((abs(spo.get(k, 0.0) - got.get(k, 0.0))
                   for k in set(spo) | set(got)), default=0.0)
    backend = 'numpy' if xp is np else getattr(xp, '__name__', str(xp))
    span = f"q{min(pool)}-{max(pool)}"
    print(f"  [{backend}] n={n:3d} ({n_words(n)} word/mask, {span:>9}), "
          f"{n_gates} gates: string={len(spo)} terms, wide={len(got)} terms, "
          f"symmetric-diff={len(diff)}, max coeff dev={max_dev:.3e}")
    assert not diff, f"wide engine produced different terms ({len(diff)})"
    assert max_dev < 1e-9, "wide engine coefficients diverge from the oracle"


if __name__ == '__main__':
    print("propagation_wide self-check vs the string-dict oracle:")
    for seed in range(3):
        self_check(seed=seed, n=4, n_gates=80)
    for seed in range(2):
        self_check(seed=seed, n=6, n_gates=150)

    # Above the packed key's 32-qubit wall, on a support that straddles it.
    # The windows are deliberately narrow: what has to be exercised is the
    # high bit positions and the word boundary, not the term count, and a
    # 12-qubit window saturates toward 4^12 terms of 100-character labels.
    for seed in range(3):
        self_check(seed=seed, n=49, n_gates=90, qubits=range(28, 36))
    # 7x7's far corner: entirely above 32, still one word.
    self_check(seed=0, n=49, n_gates=90, qubits=range(41, 49))
    # 10x10 needs two words per mask; straddle the 64-bit boundary, and end
    # with a support entirely in the high word.
    for seed in range(3):
        self_check(seed=seed, n=100, n_gates=90, qubits=range(60, 68))
    self_check(seed=0, n=100, n_gates=90, qubits=range(92, 100))
    print("ALL WIDE-KEY EQUIVALENCE CHECKS PASSED")
