"""Pauli propagation engine for BP-PPS.

Implements forward and backward propagation of Sparse Pauli Operators (SPOs)
through quantum gates in the Heisenberg picture.

Physics summary:
    Gate U_σ(θ) = exp(-iθσ/2) conjugates observable O as:
        O → U†_σ(θ) O U_σ(θ)

    For Pauli P commuting with σ: coefficient unchanged.
    For anti-commuting pair (P, R) with σ·P = (s·i)·R where s ∈ {+1,-1}:
        a_P' = cos(θ)·a_P + s·sin(θ)·a_R      (Eq. 8)
        a_R' = -s·sin(θ)·a_P + cos(θ)·a_R

    Backward gradient (Eq. 21):
        ∂L/∂θ_t = Σ_{(P,R)} s·(λ_P·ã_R - λ_R·ã_P)

    where ã are coefficients, λ are adjoint variables, and the sum
    is over all anti-commuting pairs under gate σ.

Gate ordering convention:
    A gate sequence is always stored in *circuit order*: sequence[0] is the
    gate applied first to the state, matching HVA.build_circuit() and
    TrotterCircuit.build_circuit(). The circuit unitary is therefore

        U = g_{T-1} · ... · g_1 · g_0        (matrix product)

    The observable lives at the circuit *output*, so Heisenberg propagation
    pushes it back toward the input: it conjugates by the last gate first,

        U† O U = g_0† ... g_{T-1}† O g_{T-1} ... g_0

    Hence propagate_forward() walks the sequence in reverse, and
    propagate_backward() — which undoes those steps — walks it forwards.
    Feeding the sequence to propagate_forward() in list order instead would
    silently evolve the observable under U† rather than U.
"""

import math

import numpy as np
from typing import Dict, List, Optional, Tuple

# Type alias: Sparse Pauli Operator = {pauli_label_string: float_coefficient}
SPO = Dict[str, float]


class TruncationStats:
    """Tracks the l2 weight thrown away by the coefficient truncation.

    Implements the empirical observable-error estimate of BP-PPS Appendix B.
    The weight discarded at gate s is

        eps_s = sqrt( sum_{P in D_s} |a_P^[s]|^2 )              (Eq. B3)

    and, because residuals created at different gates are not aligned after
    further propagation, they are accumulated in quadrature rather than
    coherently:

        eps_emp = sqrt( sum_s eps_s^2 )                          (Eq. B16)

    which reduces to one running sum of squares over every discarded
    coefficient. This is *not* a rigorous bound -- the paper drops both the
    2^(n/2) Cauchy-Schwarz factor and the worst-case alignment assumption --
    but Fig. 8 of the paper shows it upper-bounds the observed energy error in
    practice, which is what makes it useful as a run-time diagnostic.
    """

    __slots__ = ('sum_sq', 'n_discarded', 'n_gates')

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum_sq = 0.0
        self.n_discarded = 0
        self.n_gates = 0

    def discard(self, coeff: float) -> None:
        """Record one dropped coefficient."""
        self.sum_sq += coeff * coeff
        self.n_discarded += 1

    @property
    def error_estimate(self) -> float:
        """eps_emp of Eq. (B16)."""
        return math.sqrt(self.sum_sq)

    def merge(self, other: "TruncationStats") -> None:
        """Absorb another accumulator (e.g. one observable of L_{X,Z})."""
        self.sum_sq += other.sum_sq
        self.n_discarded += other.n_discarded
        self.n_gates += other.n_gates

    def __repr__(self) -> str:
        return (f"TruncationStats(eps_emp={self.error_estimate:.3e}, "
                f"discarded={self.n_discarded}, gates={self.n_gates})")


# ============================================================================
# Specialized gate applications (RX and RZZ)
# ============================================================================

def apply_rx_forward(coeffs: SPO, qubit: int, theta: float,
                     delta: float = 0.0,
                     stats: Optional[TruncationStats] = None) -> SPO:
    """Apply RX(θ) gate on qubit in Heisenberg picture (forward).

    Generator σ = X on qubit k.
    Anti-commuting iff P[k] ∈ {Y, Z}.
    Conjugate swap: Y ↔ Z at position k.

    Sign:
        P[k]=Y → X·Y = iZ → s = +1
        P[k]=Z → X·Z = -iY → s = -1

    Args:
        stats: Optional TruncationStats accumulating the discarded l2 weight.
    """
    new_coeffs = {}
    processed = set()
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    thresh = max(delta, 1e-15)
    k = qubit

    for P, a_P in coeffs.items():
        if P in processed:
            continue

        c = P[k]
        if c == 'I' or c == 'X':
            # Commuting: coefficient unchanged, but still subject to the
            # threshold (BP-PPS App. C: "coefficients below the threshold are
            # discarded" applies to the whole updated array, not only to the
            # rotated pairs).
            if abs(a_P) > thresh:
                new_coeffs[P] = a_P
            elif stats is not None:
                stats.discard(a_P)
        else:
            # Anti-commuting: Y ↔ Z swap at position k
            if c == 'Y':
                R = P[:k] + 'Z' + P[k+1:]
                sign = 1
            else:  # c == 'Z'
                R = P[:k] + 'Y' + P[k+1:]
                sign = -1

            a_R = coeffs.get(R, 0.0)

            new_a_P = cos_t * a_P + sign * sin_t * a_R
            new_a_R = -sign * sin_t * a_P + cos_t * a_R

            if abs(new_a_P) > thresh:
                new_coeffs[P] = new_a_P
            elif stats is not None:
                stats.discard(new_a_P)
            if abs(new_a_R) > thresh:
                new_coeffs[R] = new_a_R
            elif stats is not None:
                stats.discard(new_a_R)

            processed.add(P)
            processed.add(R)

    if stats is not None:
        stats.n_gates += 1
    return new_coeffs


def apply_rzz_forward(coeffs: SPO, qi: int, qj: int, theta: float,
                      delta: float = 0.0,
                      stats: Optional[TruncationStats] = None) -> SPO:
    """Apply RZZ(θ) gate on qubits (qi, qj) in Heisenberg picture (forward).

    Generator σ = Z_qi ⊗ Z_qj.
    Anti-commuting iff exactly one of P[qi], P[qj] ∈ {X, Y}.

    At anti-commuting position: X ↔ Y swap.
    At commuting position (of qi/qj pair): I ↔ Z swap.

    Sign:
        X at anti-commuting position → s = +1
        Y at anti-commuting position → s = -1

    Args:
        stats: Optional TruncationStats accumulating the discarded l2 weight.
    """
    new_coeffs = {}
    processed = set()
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    thresh = max(delta, 1e-15)

    for P, a_P in coeffs.items():
        if P in processed:
            continue

        ci = P[qi]
        cj = P[qj]
        anti_i = ci == 'X' or ci == 'Y'
        anti_j = cj == 'X' or cj == 'Y'

        if anti_i == anti_j:
            # Both commute or both anti-commute → overall commutes
            if abs(a_P) > thresh:
                new_coeffs[P] = a_P
            elif stats is not None:
                stats.discard(a_P)
        else:
            # Exactly one anti-commutes → overall anti-commutes
            R_chars = list(P)

            if anti_i:
                # Position qi: X↔Y swap (anti-commuting with Z)
                R_chars[qi] = 'Y' if ci == 'X' else 'X'
                sign = 1 if ci == 'X' else -1
                # Position qj: I↔Z swap (commuting Z·I=Z, Z·Z=I)
                R_chars[qj] = 'Z' if cj == 'I' else 'I'
            else:
                # Position qj: X↔Y swap
                R_chars[qj] = 'Y' if cj == 'X' else 'X'
                sign = 1 if cj == 'X' else -1
                # Position qi: I↔Z swap
                R_chars[qi] = 'Z' if ci == 'I' else 'I'

            R = ''.join(R_chars)
            a_R = coeffs.get(R, 0.0)

            new_a_P = cos_t * a_P + sign * sin_t * a_R
            new_a_R = -sign * sin_t * a_P + cos_t * a_R

            if abs(new_a_P) > thresh:
                new_coeffs[P] = new_a_P
            elif stats is not None:
                stats.discard(new_a_P)
            if abs(new_a_R) > thresh:
                new_coeffs[R] = new_a_R
            elif stats is not None:
                stats.discard(new_a_R)

            processed.add(P)
            processed.add(R)

    if stats is not None:
        stats.n_gates += 1
    return new_coeffs


# ============================================================================
# Backward gate steps (gradient + inverse propagation, fused)
# ============================================================================
#
# BP-PPS Sec. II B 2 stores the derivative *jointly* with the SPO and applies
# one truncation rule to the whole tuple:
#
#   "We use the basic truncation rule based on the magnitude of the
#    coefficients |a_Pi|. If the magnitude is smaller than the threshold
#    |a_Pi| < δ, we discard the tuple (P_i, a_Pi, ∂L/∂a_Pi) including both
#    the coefficient and the derivative."
#
# Two consequences, both intended:
#   * the adjoint is never truncated on its own scale (which is arbitrary --
#     it is 2(a_P - ã_P) for L_{X,Z} but 1 for the energy);
#   * backward propagation is automatically restricted to Pauli strings that
#     carry coefficient support, which is what keeps the backward pass the
#     same cost as the forward pass.
#
# Doing the gradient and the two inverse rotations in a single sweep also
# replaces the three passes the previous implementation needed.

def _backward_rx(coeffs: SPO, adjoint: SPO, qubit: int, theta: float,
                 delta: float = 0.0,
                 stats: Optional[TruncationStats] = None
                 ) -> Tuple[SPO, SPO, float]:
    """Backward step for an RX gate.

    Accumulates ∂L/∂θ (Eq. 21) from the post-gate values, then rotates both
    the coefficients and the adjoints by U(-θ) (Eqs. 11 and 20) and applies
    the joint truncation rule.
    """
    gradient = 0.0
    new_coeffs: SPO = {}
    new_adjoint: SPO = {}
    processed = set()
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    thresh = max(delta, 1e-15)
    k = qubit

    for P in (coeffs.keys() | adjoint.keys()):
        if P in processed:
            continue

        c = P[k]
        if c == 'I' or c == 'X':
            # Commuting: no gradient contribution, no rotation.
            a_P = coeffs.get(P, 0.0)
            if abs(a_P) > thresh:
                new_coeffs[P] = a_P
                lam_P = adjoint.get(P, 0.0)
                if lam_P != 0.0:
                    new_adjoint[P] = lam_P
            elif stats is not None:
                stats.discard(a_P)
            continue

        if c == 'Y':
            R = P[:k] + 'Z' + P[k+1:]
            sign = 1
        else:  # c == 'Z'
            R = P[:k] + 'Y' + P[k+1:]
            sign = -1

        a_P = coeffs.get(P, 0.0)
        a_R = coeffs.get(R, 0.0)
        lam_P = adjoint.get(P, 0.0)
        lam_R = adjoint.get(R, 0.0)

        # --- Eq. 21, evaluated at step t (post-gate) ---
        gradient += sign * (lam_P * a_R - lam_R * a_P)

        # --- Eqs. 11 / 20: inverse rotation U(-θ) of both arrays ---
        new_a_P = cos_t * a_P - sign * sin_t * a_R
        new_a_R = sign * sin_t * a_P + cos_t * a_R
        new_l_P = cos_t * lam_P - sign * sin_t * lam_R
        new_l_R = sign * sin_t * lam_P + cos_t * lam_R

        # --- joint truncation, keyed on the coefficient magnitude ---
        if abs(new_a_P) > thresh:
            new_coeffs[P] = new_a_P
            if new_l_P != 0.0:
                new_adjoint[P] = new_l_P
        elif stats is not None:
            stats.discard(new_a_P)
        if abs(new_a_R) > thresh:
            new_coeffs[R] = new_a_R
            if new_l_R != 0.0:
                new_adjoint[R] = new_l_R
        elif stats is not None:
            stats.discard(new_a_R)

        processed.add(P)
        processed.add(R)

    if stats is not None:
        stats.n_gates += 1
    return new_coeffs, new_adjoint, gradient


def _backward_rzz(coeffs: SPO, adjoint: SPO, qi: int, qj: int,
                  theta: float, delta: float = 0.0,
                  stats: Optional[TruncationStats] = None
                  ) -> Tuple[SPO, SPO, float]:
    """Backward step for an RZZ gate. See :func:`_backward_rx`."""
    gradient = 0.0
    new_coeffs: SPO = {}
    new_adjoint: SPO = {}
    processed = set()
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    thresh = max(delta, 1e-15)

    for P in (coeffs.keys() | adjoint.keys()):
        if P in processed:
            continue

        ci = P[qi]
        cj = P[qj]
        anti_i = ci == 'X' or ci == 'Y'
        anti_j = cj == 'X' or cj == 'Y'

        if anti_i == anti_j:
            a_P = coeffs.get(P, 0.0)
            if abs(a_P) > thresh:
                new_coeffs[P] = a_P
                lam_P = adjoint.get(P, 0.0)
                if lam_P != 0.0:
                    new_adjoint[P] = lam_P
            elif stats is not None:
                stats.discard(a_P)
            continue

        R_chars = list(P)
        if anti_i:
            R_chars[qi] = 'Y' if ci == 'X' else 'X'
            sign = 1 if ci == 'X' else -1
            R_chars[qj] = 'Z' if cj == 'I' else 'I'
        else:
            R_chars[qj] = 'Y' if cj == 'X' else 'X'
            sign = 1 if cj == 'X' else -1
            R_chars[qi] = 'Z' if ci == 'I' else 'I'
        R = ''.join(R_chars)

        a_P = coeffs.get(P, 0.0)
        a_R = coeffs.get(R, 0.0)
        lam_P = adjoint.get(P, 0.0)
        lam_R = adjoint.get(R, 0.0)

        gradient += sign * (lam_P * a_R - lam_R * a_P)

        new_a_P = cos_t * a_P - sign * sin_t * a_R
        new_a_R = sign * sin_t * a_P + cos_t * a_R
        new_l_P = cos_t * lam_P - sign * sin_t * lam_R
        new_l_R = sign * sin_t * lam_P + cos_t * lam_R

        if abs(new_a_P) > thresh:
            new_coeffs[P] = new_a_P
            if new_l_P != 0.0:
                new_adjoint[P] = new_l_P
        elif stats is not None:
            stats.discard(new_a_P)
        if abs(new_a_R) > thresh:
            new_coeffs[R] = new_a_R
            if new_l_R != 0.0:
                new_adjoint[R] = new_l_R
        elif stats is not None:
            stats.discard(new_a_R)

        processed.add(P)
        processed.add(R)

    if stats is not None:
        stats.n_gates += 1
    return new_coeffs, new_adjoint, gradient


# ============================================================================
# Gate sequence builders
# ============================================================================

# Gate tuple formats:
#   RX:  ('rx',  qubit,    theta, param_index)
#   RZZ: ('rzz', qi, qj,  theta, param_index)
# param_index = -1 for non-trainable (Trotter) gates

def build_hva_gate_sequence(num_qubits: int, bonds: list,
                            substep_bonds: dict, n_layers: int,
                            params: np.ndarray) -> list:
    """Build gate sequence for HVA circuit.

    Gate order per layer: RX on all qubits, then RZZ in 4 square lattice substeps.
    This matches the HVA.build_circuit() method.

    Args:
        num_qubits: Number of qubits.
        bonds: List of (i, j) bond tuples.
        substep_bonds: Dict {1..4: [(bond_idx, i, j), ...]} for square lattice coloring.
        n_layers: Number of HVA layers.
        params: Parameter array of shape (n_params,).

    Returns:
        List of gate tuples.
    """
    sequence = []
    num_bonds = len(bonds)
    params_per_layer = num_qubits + num_bonds

    for layer in range(n_layers):
        offset = layer * params_per_layer

        # RX gates on all qubits
        for q in range(num_qubits):
            pidx = offset + q
            sequence.append(('rx', q, float(params[pidx]), pidx))

        # RZZ gates in 4 square lattice substeps (same order as HVA.build_circuit)
        for step in range(1, 5):
            for bond_idx, i, j in substep_bonds[step]:
                pidx = offset + num_qubits + bond_idx
                sequence.append(('rzz', i, j, float(params[pidx]), pidx))

    return sequence


def _append_s2_step(sequence: list, num_qubits: int, substep_bonds: dict,
                    J: np.ndarray, h: float, tau: float) -> None:
    """Append one 2nd-order Suzuki-Trotter step S2(τ) to the sequence.

    S2(τ) = exp(-iH_X τ/2) · exp(-iH_ZZ τ) · exp(-iH_X τ/2)

    This is a helper used by both order=2 and order=4 Trotter builders.
    """
    # exp(-iH_X τ/2): RX(-2h·τ/2) = RX(-h·τ)
    for q in range(num_qubits):
        theta = -h * tau
        sequence.append(('rx', q, theta, -1))
    # exp(-iH_ZZ τ): RZZ(-2J_ij·τ)
    for s in range(1, 5):
        for bond_idx, i, j in substep_bonds[s]:
            theta = -2.0 * J[bond_idx] * tau
            sequence.append(('rzz', i, j, theta, -1))
    # exp(-iH_X τ/2): RX(-h·τ)
    for q in range(num_qubits):
        theta = -h * tau
        sequence.append(('rx', q, theta, -1))


def build_trotter_gate_sequence(num_qubits: int, substep_bonds: dict,
                                J: np.ndarray, h: float,
                                dt: float, n_steps: int,
                                order: int = 2) -> list:
    """Build gate sequence for Trotter circuit (fixed, non-trainable).

    Matches TrotterCircuit.build_circuit() gate ordering.
    All param_index = -1.

    For Hamiltonian H = -Σ J_ij Z_i Z_j - h Σ X_i:
        exp(-iH dt) ≈ product of RX(-2h·dt) and RZZ(-2J_ij·dt)

    Supported orders:
        1: Lie-Trotter (1st order)
        2: 2nd-order Suzuki-Trotter
        4: 4th-order Suzuki-Trotter
           S4(dt) = S2(p·dt)² · S2((1-4p)·dt) · S2(p·dt)²
           where p = 1 / (4 - 4^(1/3))
    """
    sequence = []

    for _ in range(n_steps):
        if order == 1:
            # 1st order: exp(-iH_ZZ dt) · exp(-iH_X dt)
            for s in range(1, 5):
                for bond_idx, i, j in substep_bonds[s]:
                    theta = -2.0 * J[bond_idx] * dt
                    sequence.append(('rzz', i, j, theta, -1))
            for q in range(num_qubits):
                theta = -2.0 * h * dt
                sequence.append(('rx', q, theta, -1))

        elif order == 2:
            _append_s2_step(sequence, num_qubits, substep_bonds, J, h, dt)

        elif order == 4:
            # 4th-order Suzuki-Trotter:
            # S4(dt) = S2(p·dt) · S2(p·dt) · S2((1-4p)·dt) · S2(p·dt) · S2(p·dt)
            p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
            tau_outer = p * dt
            tau_inner = (1.0 - 4.0 * p) * dt
            _append_s2_step(sequence, num_qubits, substep_bonds, J, h, tau_outer)
            _append_s2_step(sequence, num_qubits, substep_bonds, J, h, tau_outer)
            _append_s2_step(sequence, num_qubits, substep_bonds, J, h, tau_inner)
            _append_s2_step(sequence, num_qubits, substep_bonds, J, h, tau_outer)
            _append_s2_step(sequence, num_qubits, substep_bonds, J, h, tau_outer)

        else:
            raise ValueError(f"Unsupported Trotter order: {order}. Use 1, 2, or 4.")

    return sequence


# ============================================================================
# Full forward / backward passes
# ============================================================================

def propagate_forward(init_spo: SPO, gate_sequence: list,
                      delta: float = 0.0,
                      stats: Optional[TruncationStats] = None) -> SPO:
    """Propagate an SPO through a gate sequence (Heisenberg picture, forward).

    Computes U† O U for the circuit unitary U built from `gate_sequence`.
    The observable starts at the circuit output, so the gates are conjugated
    last-to-first (see "Gate ordering convention" in the module docstring).

    Args:
        init_spo: Initial SPO (e.g., {X_0_label: 1.0}).
        gate_sequence: List of gate tuples from build_*_gate_sequence,
            in circuit order (first gate applied first to the state).
        delta: Truncation threshold. Terms with |a_P| < delta are discarded.
        stats: Optional TruncationStats accumulating the discarded l2 weight,
            i.e. the BP-PPS Appendix B observable-error estimate.

    Returns:
        Evolved SPO after all gates.
    """
    spo = dict(init_spo)
    for gate in reversed(gate_sequence):
        if gate[0] == 'rx':
            _, q, theta, _ = gate
            spo = apply_rx_forward(spo, q, theta, delta, stats)
        elif gate[0] == 'rzz':
            _, qi, qj, theta, _ = gate
            spo = apply_rzz_forward(spo, qi, qj, theta, delta, stats)
    return spo


def propagate_backward(final_spo: SPO, seed: SPO,
                       gate_sequence: list, n_params: int,
                       delta: float = 0.0,
                       stats: Optional[TruncationStats] = None) -> np.ndarray:
    """Backward pass: compute gradients for all trainable parameters.

    Undoes the forward sweep, so it visits the gates in the opposite order:
    forward went last-to-first, backward goes first-to-last. At each gate:
    1. Compute gradient contribution (Eq. 21) — must happen *before* the
       inverse rotation, since Eq. 21 uses the post-gate coefficients and
       adjoints at step t.
    2. Apply inverse gate to reconstruct coefficients (Eq. 20)
    3. Apply inverse gate to propagate adjoint backward

    Args:
        final_spo: Evolved SPO from forward pass.
        seed: Gradient seed {pauli_label: ∂L/∂a_P^[T]}.
        gate_sequence: Same gate sequence used in forward pass, in circuit order.
        n_params: Total number of trainable parameters.
        delta: Truncation threshold for backward reconstruction. Applied to
            the (Pauli, coefficient, adjoint) tuple as a unit, keyed on the
            coefficient magnitude.
        stats: Optional TruncationStats accumulating the discarded l2 weight.

    Returns:
        Gradient array of shape (n_params,).
    """
    gradients = np.zeros(n_params)

    coeffs = dict(final_spo)
    adjoint = dict(seed)

    for gate in gate_sequence:
        if gate[0] == 'rx':
            _, q, theta, pidx = gate
            coeffs, adjoint, grad = _backward_rx(
                coeffs, adjoint, q, theta, delta, stats
            )
        elif gate[0] == 'rzz':
            _, qi, qj, theta, pidx = gate
            coeffs, adjoint, grad = _backward_rzz(
                coeffs, adjoint, qi, qj, theta, delta, stats
            )
        else:
            continue

        if pidx >= 0:
            gradients[pidx] += grad

    return gradients
