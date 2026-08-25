"""
    bppps_engine.jl — BP-PPS Backward Pass Engine (Eq. 20-21)

Direct implementation of the Backpropagating Pauli Propagation algorithm
from Rudolph et al. (2025), NOT Zygote automatic differentiation.

Key differences from standard AD:
    - Memory: O(N_P) — only stores current SPO + adjoint
    - Standard AD: O(D × N_P) — stores ALL intermediate SPOs
    - BP-PPS reconstructs intermediates on-the-fly via inverse gates

SPO representation: Dict{String, Float64}
    Key = N-character Pauli label ("IXYZ..."), 1-based qubit indexing
    Value = real coefficient a_P

Gate tuple: (type::Symbol, params..., param_index::Int)
    :rx  → (:rx, qubit, theta, param_idx)
    :rzz → (:rzz, qi, qj, theta, param_idx)
    param_idx = -1 for non-trainable gates
"""

# Type alias
const SPO = Dict{String, Float64}

# ============================================================================
# Specialized forward gate applications (Heisenberg picture)
# ============================================================================

"""Apply RX(θ) on qubit k (1-based) to SPO.

Generator σ = X_k.
Anti-commuting iff P[k] ∈ {Y, Z}.
Sign: P[k]=Y → s=+1, P[k]=Z → s=-1.
"""
function apply_rx_forward!(result::SPO, coeffs::SPO, k::Int, theta::Float64, delta::Float64)
    empty!(result)
    processed = Set{String}()
    ct, st = cos(theta), sin(theta)

    for (P, a_P) in coeffs
        P in processed && continue

        c = P[k]
        if c == 'I' || c == 'X'
            # Commuting
            result[P] = a_P
        else
            # Anti-commuting: Y ↔ Z swap at position k
            R_chars = collect(P)
            if c == 'Y'
                R_chars[k] = 'Z'
                s = 1.0
            else  # Z
                R_chars[k] = 'Y'
                s = -1.0
            end
            R = String(R_chars)
            a_R = get(coeffs, R, 0.0)

            new_a_P = ct * a_P + s * st * a_R
            new_a_R = -s * st * a_P + ct * a_R

            thresh = max(delta, 1e-15)
            abs(new_a_P) > thresh && (result[P] = new_a_P)
            abs(new_a_R) > thresh && (result[R] = new_a_R)

            push!(processed, P)
            push!(processed, R)
        end
    end
    return result
end

"""Apply RZZ(θ) on qubits (qi, qj) (1-based) to SPO.

Generator σ = Z_qi ⊗ Z_qj.
Anti-commuting iff exactly one of P[qi], P[qj] ∈ {X, Y}.
"""
function apply_rzz_forward!(result::SPO, coeffs::SPO, qi::Int, qj::Int,
                             theta::Float64, delta::Float64)
    empty!(result)
    processed = Set{String}()
    ct, st = cos(theta), sin(theta)

    for (P, a_P) in coeffs
        P in processed && continue

        ci, cj = P[qi], P[qj]
        anti_i = ci == 'X' || ci == 'Y'
        anti_j = cj == 'X' || cj == 'Y'

        if anti_i == anti_j
            # Both commute or both anti-commute → overall commutes
            result[P] = a_P
        else
            R_chars = collect(P)
            if anti_i
                # Position qi: X↔Y swap
                R_chars[qi] = ci == 'X' ? 'Y' : 'X'
                s = ci == 'X' ? 1.0 : -1.0
                # Position qj: I↔Z swap
                R_chars[qj] = cj == 'I' ? 'Z' : 'I'
            else
                # Position qj: X↔Y swap
                R_chars[qj] = cj == 'X' ? 'Y' : 'X'
                s = cj == 'X' ? 1.0 : -1.0
                # Position qi: I↔Z swap
                R_chars[qi] = ci == 'I' ? 'Z' : 'I'
            end
            R = String(R_chars)
            a_R = get(coeffs, R, 0.0)

            new_a_P = ct * a_P + s * st * a_R
            new_a_R = -s * st * a_P + ct * a_R

            thresh = max(delta, 1e-15)
            abs(new_a_P) > thresh && (result[P] = new_a_P)
            abs(new_a_R) > thresh && (result[R] = new_a_R)

            push!(processed, P)
            push!(processed, R)
        end
    end
    return result
end

# Convenience wrappers (allocating versions)
function apply_rx_forward(coeffs::SPO, k::Int, theta::Float64, delta::Float64=0.0)
    result = SPO()
    apply_rx_forward!(result, coeffs, k, theta, delta)
    return result
end

function apply_rzz_forward(coeffs::SPO, qi::Int, qj::Int, theta::Float64, delta::Float64=0.0)
    result = SPO()
    apply_rzz_forward!(result, coeffs, qi, qj, theta, delta)
    return result
end

# ============================================================================
# Gate sequence builders (for training — returns tuples with param indices)
# ============================================================================

"""Build HVA gate sequence as Vector of (type, ..., param_index) tuples.

Matches the Python propagation.py `build_hva_gate_sequence` exactly.
"""
function build_hva_gate_tuples(n_qubits::Int, bonds, substeps, n_layers::Int,
                                params::Vector{Float64})
    sequence = []
    n_bonds = length(bonds)
    params_per_layer = n_qubits + n_bonds

    for layer in 1:n_layers
        offset = (layer - 1) * params_per_layer

        # RX on all qubits
        for q in 1:n_qubits
            pidx = offset + q
            push!(sequence, (:rx, q, params[pidx], pidx))
        end

        # RZZ in 4 square lattice substeps
        for step in 1:4
            for bond_idx in substeps[step]
                i, j = bonds[bond_idx]
                pidx = offset + n_qubits + bond_idx
                push!(sequence, (:rzz, i, j, params[pidx], pidx))
            end
        end
    end
    return sequence
end

# ============================================================================
# Forward propagation
# ============================================================================

"""Propagate SPO through gate sequence (forward pass)."""
function propagate_forward(init_spo::SPO, gate_sequence; delta::Float64=0.0)
    spo = copy(init_spo)
    buf = SPO()  # reusable buffer

    for gate in gate_sequence
        if gate[1] == :rx
            _, q, theta, _ = gate
            apply_rx_forward!(buf, spo, q, theta, delta)
            spo, buf = buf, spo  # swap
        elseif gate[1] == :rzz
            _, qi, qj, theta, _ = gate
            apply_rzz_forward!(buf, spo, qi, qj, theta, delta)
            spo, buf = buf, spo
        end
    end
    return spo
end

# ============================================================================
# Backward propagation (Eq. 20-21 from BP-PPS paper)
# ============================================================================

"""BP-PPS backward pass: compute gradients for all trainable parameters.

Algorithm (for each gate t, processed in reverse order):
1. Gradient (Eq. 21):
   ∂L/∂θ_t = Σ_{anti-commuting (P,R)} s · (λ_P · ã_R − λ_R · ã_P)
   
2. Inverse rotation (Eq. 20): apply gate with −θ to BOTH
   - coefficients ã (reconstruct previous step)
   - adjoint λ (propagate gradient seed backward)

Memory: O(N_P) — only current coefficients + adjoint are stored.
"""
function propagate_backward(final_spo::SPO, seed::SPO, gate_sequence,
                             n_params::Int; delta::Float64=0.0)
    gradients = zeros(n_params)

    coeffs = copy(final_spo)
    adjoint = copy(seed)
    buf_c = SPO()
    buf_a = SPO()

    for gate in Iterators.reverse(gate_sequence)
        if gate[1] == :rx
            _, q, theta, pidx = gate

            # --- Eq. 21: gradient contribution ---
            if pidx > 0
                grad = _rx_gradient(coeffs, adjoint, q)
                gradients[pidx] += grad
            end

            # --- Eq. 20: inverse gate (θ → -θ) ---
            apply_rx_forward!(buf_c, coeffs, q, -theta, delta)
            coeffs, buf_c = buf_c, coeffs
            apply_rx_forward!(buf_a, adjoint, q, -theta, delta)
            adjoint, buf_a = buf_a, adjoint

        elseif gate[1] == :rzz
            _, qi, qj, theta, pidx = gate

            if pidx > 0
                grad = _rzz_gradient(coeffs, adjoint, qi, qj)
                gradients[pidx] += grad
            end

            apply_rzz_forward!(buf_c, coeffs, qi, qj, -theta, delta)
            coeffs, buf_c = buf_c, coeffs
            apply_rzz_forward!(buf_a, adjoint, qi, qj, -theta, delta)
            adjoint, buf_a = buf_a, adjoint
        end
    end

    return gradients
end

"""Compute gradient contribution for RX gate at qubit k (Eq. 21)."""
function _rx_gradient(coeffs::SPO, adjoint::SPO, k::Int)
    grad = 0.0
    processed = Set{String}()

    all_keys = union(keys(coeffs), keys(adjoint))

    for P in all_keys
        P in processed && continue
        c = P[k]
        (c == 'I' || c == 'X') && continue  # commuting

        R_chars = collect(P)
        if c == 'Y'
            R_chars[k] = 'Z'
            s = 1.0
        else  # Z
            R_chars[k] = 'Y'
            s = -1.0
        end
        R = String(R_chars)

        a_P = get(coeffs, P, 0.0)
        a_R = get(coeffs, R, 0.0)
        λ_P = get(adjoint, P, 0.0)
        λ_R = get(adjoint, R, 0.0)

        grad += s * (λ_P * a_R - λ_R * a_P)

        push!(processed, P)
        push!(processed, R)
    end
    return grad
end

"""Compute gradient contribution for RZZ gate at (qi, qj) (Eq. 21)."""
function _rzz_gradient(coeffs::SPO, adjoint::SPO, qi::Int, qj::Int)
    grad = 0.0
    processed = Set{String}()

    all_keys = union(keys(coeffs), keys(adjoint))

    for P in all_keys
        P in processed && continue

        ci, cj = P[qi], P[qj]
        anti_i = ci == 'X' || ci == 'Y'
        anti_j = cj == 'X' || cj == 'Y'
        anti_i == anti_j && continue  # commuting

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

        push!(processed, P)
        push!(processed, R)
    end
    return grad
end

# ============================================================================
# Utility
# ============================================================================

"""Check if a Pauli label contains only I and Z."""
is_iz_only(label::String) = all(c -> c == 'I' || c == 'Z', label)

"""Create observable label: e.g., make_obs_label(4, 'X', 2) → "IXII" """
function make_obs_label(n_qubits::Int, pauli::Char, qubit::Int)
    chars = fill('I', n_qubits)
    chars[qubit] = pauli
    return String(chars)
end
