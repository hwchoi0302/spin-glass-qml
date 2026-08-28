"""
    bppps_engine.jl — Sparse Pauli dynamics + BP-PPS backward pass

Direct implementation of "Backpropagating Pauli Propagation"
(arXiv:2607.15184), Eqs. 20-21. This is a port of `src/bppps/propagation.py`
and must stay numerically identical to it; `scripts/00_validate_small.py`
tests 10-12 are the reference the Python side is held to.

No external dependencies — in particular this file does not use
PauliPropagation.jl, so it always runs.

Key properties
    - Memory: O(N_P) — only the current SPO and its adjoint are stored;
      intermediates are reconstructed with inverse gates rather than cached.
    - Standard reverse-mode AD would need O(n_param · N_P).

SPO representation: Dict{String, Float64}
    Key   = N-character Pauli label ("IXYZ..."), label[k] acts on qubit k
            (1-based).
    Value = real coefficient a_P.

Gate tuple: (type::Symbol, params..., param_index::Int)
    :rx  → (:rx, qubit, theta, param_idx)
    :rzz → (:rzz, qi, qj, theta, param_idx)
    param_idx = -1 for non-trainable gates.

Gate ordering convention
    A gate sequence is stored in *circuit order*: sequence[1] is applied first
    to the state, so the circuit unitary is U = g_T · … · g_1. The observable
    lives at the circuit output, so Heisenberg propagation conjugates by the
    last gate first,

        U† O U = g_1† … g_T† O g_T … g_1

    Hence `propagate_forward` walks the sequence in reverse and
    `propagate_backward` — which undoes those steps — walks it forwards.
    Consuming the sequence the other way round silently evolves the observable
    under U† instead of U.
"""

# Type alias
const SPO = Dict{String, Float64}

# ============================================================================
# Truncation error tracking (BP-PPS Appendix B)
# ============================================================================

"""Accumulates the l2 weight discarded by coefficient truncation.

The weight dropped at gate s is εₛ = sqrt(Σ_{P∈Dₛ} |a_P|²) (Eq. B3). Residuals
created at different gates are not aligned after further propagation, so they
are combined in quadrature (Eq. B16):

    ε_emp = sqrt( Σₛ εₛ² )

which is one running sum of squares over every discarded coefficient. This is
an empirical estimate, not a rigorous bound, but Fig. 8 of the paper shows it
upper-bounds the observed error in practice.
"""
mutable struct TruncationStats
    sum_sq::Float64
    n_discarded::Int
    n_gates::Int
end
TruncationStats() = TruncationStats(0.0, 0, 0)

@inline function discard!(stats::TruncationStats, coeff::Float64)
    stats.sum_sq += coeff * coeff
    stats.n_discarded += 1
    return nothing
end
@inline discard!(::Nothing, ::Float64) = nothing

@inline count_gate!(stats::TruncationStats) = (stats.n_gates += 1; nothing)
@inline count_gate!(::Nothing) = nothing

error_estimate(stats::TruncationStats) = sqrt(stats.sum_sq)

function merge!(dest::TruncationStats, src::TruncationStats)
    dest.sum_sq += src.sum_sq
    dest.n_discarded += src.n_discarded
    dest.n_gates += src.n_gates
    return dest
end

const MaybeStats = Union{TruncationStats, Nothing}

# ============================================================================
# Forward gate applications (Heisenberg picture)
# ============================================================================

"""Apply RX(θ) on qubit k (1-based).

Generator σ = X_k; anti-commuting iff P[k] ∈ {Y, Z}.
Conjugate pair: Y ↔ Z at position k, with σ·P = s·i·R and
s = +1 for P[k]=Y (X·Y = iZ), s = -1 for P[k]=Z (X·Z = -iY).
"""
function apply_rx_forward!(result::SPO, coeffs::SPO, k::Int, theta::Float64,
                           delta::Float64, stats::MaybeStats=nothing)
    empty!(result)
    processed = Set{String}()
    ct, st = cos(theta), sin(theta)
    thresh = max(delta, 1e-15)

    for (P, a_P) in coeffs
        P in processed && continue

        c = P[k]
        if c == 'I' || c == 'X'
            # Commuting: unchanged, but still subject to the threshold.
            if abs(a_P) > thresh
                result[P] = a_P
            else
                discard!(stats, a_P)
            end
        else
            R_chars = collect(P)
            if c == 'Y'
                R_chars[k] = 'Z'
                s = 1.0
            else  # 'Z'
                R_chars[k] = 'Y'
                s = -1.0
            end
            R = String(R_chars)
            a_R = get(coeffs, R, 0.0)

            new_a_P = ct * a_P + s * st * a_R
            new_a_R = -s * st * a_P + ct * a_R

            abs(new_a_P) > thresh ? (result[P] = new_a_P) : discard!(stats, new_a_P)
            abs(new_a_R) > thresh ? (result[R] = new_a_R) : discard!(stats, new_a_R)

            push!(processed, P)
            push!(processed, R)
        end
    end
    count_gate!(stats)
    return result
end

"""Apply RZZ(θ) on qubits (qi, qj) (1-based).

Generator σ = Z_qi ⊗ Z_qj; anti-commuting iff exactly one of P[qi], P[qj] is
in {X, Y}. At the anti-commuting site X ↔ Y, at the other site I ↔ Z.
"""
function apply_rzz_forward!(result::SPO, coeffs::SPO, qi::Int, qj::Int,
                            theta::Float64, delta::Float64,
                            stats::MaybeStats=nothing)
    empty!(result)
    processed = Set{String}()
    ct, st = cos(theta), sin(theta)
    thresh = max(delta, 1e-15)

    for (P, a_P) in coeffs
        P in processed && continue

        ci, cj = P[qi], P[qj]
        anti_i = ci == 'X' || ci == 'Y'
        anti_j = cj == 'X' || cj == 'Y'

        if anti_i == anti_j
            if abs(a_P) > thresh
                result[P] = a_P
            else
                discard!(stats, a_P)
            end
        else
            R_chars = collect(P)
            if anti_i
                R_chars[qi] = ci == 'X' ? 'Y' : 'X'
                s = ci == 'X' ? 1.0 : -1.0
                R_chars[qj] = cj == 'I' ? 'Z' : 'I'
            else
                R_chars[qj] = cj == 'X' ? 'Y' : 'X'
                s = cj == 'X' ? 1.0 : -1.0
                R_chars[qi] = ci == 'I' ? 'Z' : 'I'
            end
            R = String(R_chars)
            a_R = get(coeffs, R, 0.0)

            new_a_P = ct * a_P + s * st * a_R
            new_a_R = -s * st * a_P + ct * a_R

            abs(new_a_P) > thresh ? (result[P] = new_a_P) : discard!(stats, new_a_P)
            abs(new_a_R) > thresh ? (result[R] = new_a_R) : discard!(stats, new_a_R)

            push!(processed, P)
            push!(processed, R)
        end
    end
    count_gate!(stats)
    return result
end

# Convenience wrappers (allocating versions)
function apply_rx_forward(coeffs::SPO, k::Int, theta::Float64,
                          delta::Float64=0.0, stats::MaybeStats=nothing)
    apply_rx_forward!(SPO(), coeffs, k, theta, delta, stats)
end

function apply_rzz_forward(coeffs::SPO, qi::Int, qj::Int, theta::Float64,
                           delta::Float64=0.0, stats::MaybeStats=nothing)
    apply_rzz_forward!(SPO(), coeffs, qi, qj, theta, delta, stats)
end

# ============================================================================
# Gate sequence builders
# ============================================================================

"""HVA gate sequence in circuit order: RX on every qubit, then RZZ in
4 conflict-free substeps. Matches `bppps.propagation.build_hva_gate_sequence`.
"""
function build_hva_gate_tuples(n_qubits::Int, bonds, substeps, n_layers::Int,
                               params::Vector{Float64})
    sequence = Tuple[]
    n_bonds = length(bonds)
    params_per_layer = n_qubits + n_bonds

    for layer in 1:n_layers
        offset = (layer - 1) * params_per_layer

        for q in 1:n_qubits
            pidx = offset + q
            push!(sequence, (:rx, q, params[pidx], pidx))
        end

        for step in 1:4, bond_idx in substeps[step]
            i, j = bonds[bond_idx]
            pidx = offset + n_qubits + bond_idx
            push!(sequence, (:rzz, i, j, params[pidx], pidx))
        end
    end
    return sequence
end

# ============================================================================
# Forward propagation
# ============================================================================

"""Propagate an SPO through a circuit-order gate sequence.

Computes U† O U; the gates are conjugated last-to-first.
"""
function propagate_forward(init_spo::SPO, gate_sequence;
                           delta::Float64=0.0, stats::MaybeStats=nothing)
    spo = copy(init_spo)
    buf = SPO()

    for gate in Iterators.reverse(gate_sequence)
        if gate[1] == :rx
            _, q, theta, _ = gate
            apply_rx_forward!(buf, spo, q, theta, delta, stats)
            spo, buf = buf, spo
        elseif gate[1] == :rzz
            _, qi, qj, theta, _ = gate
            apply_rzz_forward!(buf, spo, qi, qj, theta, delta, stats)
            spo, buf = buf, spo
        end
    end
    return spo
end

# ============================================================================
# Backward propagation (Eqs. 20-21), gradient fused with the inverse update
# ============================================================================
#
# BP-PPS Sec. II B 2 stores the derivative jointly with the SPO and applies
# one truncation rule to the whole tuple:
#
#   "If the magnitude is smaller than the threshold |ã_Pi| < δ, we discard the
#    tuple (P̃i, ã_P̃i, ∂L/∂ã_Pi) including both the coefficient and the
#    derivative."
#
# So the adjoint is never truncated on its own (arbitrary) scale, and backward
# propagation is automatically restricted to Pauli strings carrying
# coefficient support — which is what keeps the backward pass the same cost as
# the forward pass.

"""Backward step for an RX gate: Eq. 21 gradient, then the Eq. 11/20 inverse
rotation of coefficients and adjoints, then the joint truncation.

The gradient must be taken *before* the inverse rotation, since Eq. 21 uses
the post-gate values at step t.
"""
function _backward_rx(coeffs::SPO, adjoint::SPO, k::Int, theta::Float64,
                      delta::Float64, stats::MaybeStats=nothing)
    grad = 0.0
    new_coeffs = SPO()
    new_adjoint = SPO()
    processed = Set{String}()
    ct, st = cos(theta), sin(theta)
    thresh = max(delta, 1e-15)

    for P in union(keys(coeffs), keys(adjoint))
        P in processed && continue

        c = P[k]
        if c == 'I' || c == 'X'
            a_P = get(coeffs, P, 0.0)
            if abs(a_P) > thresh
                new_coeffs[P] = a_P
                λ_P = get(adjoint, P, 0.0)
                λ_P != 0.0 && (new_adjoint[P] = λ_P)
            else
                discard!(stats, a_P)
            end
            continue
        end

        R_chars = collect(P)
        if c == 'Y'
            R_chars[k] = 'Z'
            s = 1.0
        else
            R_chars[k] = 'Y'
            s = -1.0
        end
        R = String(R_chars)

        a_P = get(coeffs, P, 0.0)
        a_R = get(coeffs, R, 0.0)
        λ_P = get(adjoint, P, 0.0)
        λ_R = get(adjoint, R, 0.0)

        grad += s * (λ_P * a_R - λ_R * a_P)

        new_a_P = ct * a_P - s * st * a_R
        new_a_R = s * st * a_P + ct * a_R
        new_λ_P = ct * λ_P - s * st * λ_R
        new_λ_R = s * st * λ_P + ct * λ_R

        if abs(new_a_P) > thresh
            new_coeffs[P] = new_a_P
            new_λ_P != 0.0 && (new_adjoint[P] = new_λ_P)
        else
            discard!(stats, new_a_P)
        end
        if abs(new_a_R) > thresh
            new_coeffs[R] = new_a_R
            new_λ_R != 0.0 && (new_adjoint[R] = new_λ_R)
        else
            discard!(stats, new_a_R)
        end

        push!(processed, P)
        push!(processed, R)
    end
    count_gate!(stats)
    return new_coeffs, new_adjoint, grad
end

"""Backward step for an RZZ gate. See `_backward_rx`."""
function _backward_rzz(coeffs::SPO, adjoint::SPO, qi::Int, qj::Int,
                       theta::Float64, delta::Float64,
                       stats::MaybeStats=nothing)
    grad = 0.0
    new_coeffs = SPO()
    new_adjoint = SPO()
    processed = Set{String}()
    ct, st = cos(theta), sin(theta)
    thresh = max(delta, 1e-15)

    for P in union(keys(coeffs), keys(adjoint))
        P in processed && continue

        ci, cj = P[qi], P[qj]
        anti_i = ci == 'X' || ci == 'Y'
        anti_j = cj == 'X' || cj == 'Y'

        if anti_i == anti_j
            a_P = get(coeffs, P, 0.0)
            if abs(a_P) > thresh
                new_coeffs[P] = a_P
                λ_P = get(adjoint, P, 0.0)
                λ_P != 0.0 && (new_adjoint[P] = λ_P)
            else
                discard!(stats, a_P)
            end
            continue
        end

        R_chars = collect(P)
        if anti_i
            R_chars[qi] = ci == 'X' ? 'Y' : 'X'
            s = ci == 'X' ? 1.0 : -1.0
            R_chars[qj] = cj == 'I' ? 'Z' : 'I'
        else
            R_chars[qj] = cj == 'X' ? 'Y' : 'X'
            s = cj == 'X' ? 1.0 : -1.0
            R_chars[qi] = ci == 'I' ? 'Z' : 'I'
        end
        R = String(R_chars)

        a_P = get(coeffs, P, 0.0)
        a_R = get(coeffs, R, 0.0)
        λ_P = get(adjoint, P, 0.0)
        λ_R = get(adjoint, R, 0.0)

        grad += s * (λ_P * a_R - λ_R * a_P)

        new_a_P = ct * a_P - s * st * a_R
        new_a_R = s * st * a_P + ct * a_R
        new_λ_P = ct * λ_P - s * st * λ_R
        new_λ_R = s * st * λ_P + ct * λ_R

        if abs(new_a_P) > thresh
            new_coeffs[P] = new_a_P
            new_λ_P != 0.0 && (new_adjoint[P] = new_λ_P)
        else
            discard!(stats, new_a_P)
        end
        if abs(new_a_R) > thresh
            new_coeffs[R] = new_a_R
            new_λ_R != 0.0 && (new_adjoint[R] = new_λ_R)
        else
            discard!(stats, new_a_R)
        end

        push!(processed, P)
        push!(processed, R)
    end
    count_gate!(stats)
    return new_coeffs, new_adjoint, grad
end

"""BP-PPS backward pass: gradients for every trainable parameter.

Undoes the forward sweep, so it visits the gates in the opposite order:
forward went last-to-first, backward goes first-to-last.
Memory: O(N_P) — only the current coefficients and adjoints are held.
"""
function propagate_backward(final_spo::SPO, seed::SPO, gate_sequence,
                            n_params::Int; delta::Float64=0.0,
                            stats::MaybeStats=nothing)
    gradients = zeros(n_params)
    coeffs = copy(final_spo)
    adjoint = copy(seed)

    for gate in gate_sequence
        if gate[1] == :rx
            _, q, theta, pidx = gate
            coeffs, adjoint, grad = _backward_rx(coeffs, adjoint, q, theta,
                                                 delta, stats)
            pidx > 0 && (gradients[pidx] += grad)
        elseif gate[1] == :rzz
            _, qi, qj, theta, pidx = gate
            coeffs, adjoint, grad = _backward_rzz(coeffs, adjoint, qi, qj,
                                                  theta, delta, stats)
            pidx > 0 && (gradients[pidx] += grad)
        end
    end
    return gradients
end

# ============================================================================
# Utility
# ============================================================================

"""True if a Pauli label contains only I and Z (contributes to ⟨0|·|0⟩)."""
is_iz_only(label::String) = all(c -> c == 'I' || c == 'Z', label)

"""Observable label, e.g. `make_obs_label(4, 'X', 2)` → "IXII"."""
function make_obs_label(n_qubits::Int, pauli::Char, qubit::Int)
    chars = fill('I', n_qubits)
    chars[qubit] = pauli
    return String(chars)
end

"""Squared rescaled Hilbert-Schmidt norm, ‖O‖² = Σ_P |a_P|²."""
operator_norm_sq(spo::SPO) = sum(a * a for a in values(spo); init=0.0)
