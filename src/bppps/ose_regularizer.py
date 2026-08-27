"""Operator Stabilizer Entropy (OSE) regularization.

OSE controls the complexity (sparsity) of the evolved Pauli operator.
Uses Shannon entropy (alpha=1) as specified by user decision.

For SPO O = Σ_P a_P P with ||O||² = Σ_P a_P² = 1 (unit observable):
    M^[1](O) = -Σ_P a_P² log(a_P²)

Gradient w.r.t. coefficient a_P:
    ∂M/∂a_P = -2a_P (1 + log(a_P²)) = -2a_P (1 + 2 log|a_P|)
"""

import numpy as np
from typing import Dict

SPO = Dict[str, float]

# Small epsilon to avoid log(0)
_EPS = 1e-30


def compute_ose(coeffs: SPO, alpha: float = 1.0) -> float:
    """Compute Operator Stabilizer Entropy of an SPO.

    Args:
        coeffs: SPO coefficient dictionary.
        alpha: Rényi parameter (1 = Shannon entropy).

    Returns:
        OSE value (non-negative).
    """
    values = np.array(list(coeffs.values()))
    p = values ** 2  # a_P² (already normalized since ||O||=1)

    if alpha == 1.0:
        # Shannon entropy: -Σ p log(p)
        mask = p > _EPS
        entropy = -np.sum(p[mask] * np.log(p[mask]))
        return float(entropy)
    else:
        # Rényi entropy: log(Σ p^α) / (1 - α)
        mask = p > _EPS
        return float(np.log(np.sum(p[mask] ** alpha)) / (1 - alpha))


def ose_gradient_seed(coeffs: SPO, lambda_ose: float,
                      alpha: float = 1.0) -> Dict[str, float]:
    """Compute OSE gradient contribution to the gradient seed.

    For α=1 (Shannon entropy):
        ∂M/∂a_P = -2 a_P (1 + log(a_P²))

    The total seed contribution is: λ_OSE · ∂M/∂a_P

    Args:
        coeffs: SPO coefficient dictionary.
        lambda_ose: OSE regularization strength.
        alpha: Rényi parameter.

    Returns:
        Dict mapping Pauli labels to OSE gradient contributions.
    """
    seed = {}

    if lambda_ose == 0.0:
        return seed

    for P, a_P in coeffs.items():
        if abs(a_P) < _EPS:
            continue

        if alpha == 1.0:
            # ∂M/∂a_P = -2 a_P (1 + log(a_P²))
            log_ap2 = np.log(a_P ** 2 + _EPS)
            grad_m = -2.0 * a_P * (1.0 + log_ap2)
        else:
            # For general α, gradient is more complex
            # Not used (user fixed α=1)
            raise NotImplementedError(f"OSE gradient for alpha={alpha}")

        seed[P] = lambda_ose * grad_m

    return seed
