"""Operator Stabilizer Entropy (OSE) regularization.

BP-PPS Sec. IV A, Eq. (31): for a *normalized* operator

    O_hat = sum_P a_P P    with    sum_P |a_P|^2 = 1

the operator stabilizer Renyi-alpha entropy is the participation entropy of
the probability distribution p_P = |a_P|^2 over Pauli strings,

    M[alpha](O_hat) = 1/(1-alpha) * log( sum_P p_P^alpha )
    M[1]  (O_hat) = - sum_P p_P log p_P                (Shannon limit)
    M[0]  (O_hat) = log(rank)

It measures how spread out the Pauli decomposition is, and therefore controls
the truncation error in the same way entanglement entropy controls the bond
dimension of an MPS. It is also a magic monotone, so it doubles as a proxy for
the non-Clifford resources the circuit generates.

Normalization matters
---------------------
The SPOs this module is handed are *not* normalized:

  * the Hamiltonian used for ground-state preparation has
    ||H||_rHS = sqrt(sum J_ij^2 + n h^2)  (e.g. sqrt(40) for the 4x4 model);
  * a propagated single-site observable starts at norm 1, but truncation
    strictly decreases the norm as the circuit deepens.

So p_P must be computed as |a_P|^2 / ||O||^2, and the gradient must carry the
chain rule through that normalization. Both are done here. As a check, the
resulting gradient is orthogonal to the coefficient vector,
sum_P a_P dM/da_P = 0, since M depends only on the direction of a.

Regularized objective (Eq. 32):   L_total = L + lambda_OSE * M[alpha]
"""

import numpy as np
from typing import Dict, Tuple

SPO = Dict[str, float]

# Below this squared weight a term contributes nothing measurable to the
# entropy and would only inject log(0) noise.
_EPS = 1e-30


def operator_norm_sq(coeffs: SPO) -> float:
    """Squared rescaled Hilbert-Schmidt norm, ||O||^2 = sum_P |a_P|^2.

    Pauli strings are orthonormal under the rescaled Hilbert-Schmidt inner
    product <A,B>_rHS = 2^-n Tr[A^dag B], so the rHS norm of an SPO is just
    the l2 norm of its coefficient vector.
    """
    return float(sum(a * a for a in coeffs.values()))


def _distribution(coeffs: SPO) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (labels-order coefficient array, probabilities, norm^2)."""
    values = np.fromiter(coeffs.values(), dtype=float, count=len(coeffs))
    norm_sq = float(np.dot(values, values))
    if norm_sq <= _EPS:
        return values, np.zeros_like(values), 0.0
    return values, (values * values) / norm_sq, norm_sq


def compute_ose(coeffs: SPO, alpha: float = 1.0) -> float:
    """Operator stabilizer Renyi-alpha entropy of an SPO (Eq. 31).

    The operator is normalized internally, so the caller may pass an SPO of
    any norm.

    Args:
        coeffs: SPO coefficient dictionary.
        alpha: Renyi index. 1.0 uses the Shannon limit.

    Returns:
        M[alpha] >= 0, in nats.
    """
    if not coeffs:
        return 0.0

    _, p, norm_sq = _distribution(coeffs)
    if norm_sq == 0.0:
        return 0.0

    mask = p > _EPS
    if not np.any(mask):
        return 0.0

    if alpha == 1.0:
        return float(-np.sum(p[mask] * np.log(p[mask])))
    return float(np.log(np.sum(p[mask] ** alpha)) / (1.0 - alpha))


def ose_gradient_seed(coeffs: SPO, lambda_ose: float,
                      alpha: float = 1.0) -> Dict[str, float]:
    """Gradient of ``lambda_ose * M[alpha]`` with respect to each coefficient.

    With N = ||O||^2 and p_P = a_P^2 / N, differentiating Eq. (31) through the
    normalization gives, for the Shannon case,

        dM[1]/da_P = -(2 a_P / N) * ( log p_P + M[1] )

    and for general alpha, with S = sum_Q p_Q^alpha,

        dM[a]/da_P = (2 a_P alpha) / ((1-alpha) N S) * ( p_P^(alpha-1) - S )

    Both are orthogonal to a, as required for a scale-invariant functional.

    Args:
        coeffs: SPO coefficient dictionary (any norm).
        lambda_ose: Regularization strength; 0 returns an empty seed.
        alpha: Renyi index.

    Returns:
        Dict mapping Pauli labels to lambda_ose * dM/da_P.
    """
    if lambda_ose == 0.0 or not coeffs:
        return {}

    labels = list(coeffs.keys())
    values, p, norm_sq = _distribution(coeffs)
    if norm_sq == 0.0:
        return {}

    safe_p = np.where(p > _EPS, p, _EPS)

    if alpha == 1.0:
        entropy = float(-np.sum(p[p > _EPS] * np.log(p[p > _EPS])))
        grad = -(2.0 * values / norm_sq) * (np.log(safe_p) + entropy)
    else:
        S = float(np.sum(safe_p ** alpha))
        grad = ((2.0 * values * alpha) / ((1.0 - alpha) * norm_sq * S)
                * (safe_p ** (alpha - 1.0) - S))

    grad *= lambda_ose
    return {label: float(g) for label, g in zip(labels, grad) if g != 0.0}
