"""
    hamiltonian.jl — 2D Spin Glass Hamiltonian builder for PauliPropagation.jl

Builds the Hamiltonian as a PauliSum and the bond structure for a square lattice.
Also generates HVA and Trotter gate sequences as PauliRotation arrays.

Hamiltonian: H = -Σ_{⟨i,j⟩} J_{ij} Z_i Z_j  -  h Σ_i X_i
"""

using PauliPropagation
using Random

# ============================================================================
# Lattice helpers
# ============================================================================

"""Qubit index from grid position (x,y) on Lx×Ly lattice, 1-based."""
qubit_index(x::Int, y::Int, Lx::Int) = (y - 1) * Lx + x

"""Build nearest-neighbor bonds for Lx×Ly open-boundary square lattice."""
function build_bonds(Lx::Int, Ly::Int)
    bonds = Tuple{Int,Int}[]
    # Horizontal
    for y in 1:Ly, x in 1:(Lx-1)
        push!(bonds, (qubit_index(x, y, Lx), qubit_index(x+1, y, Lx)))
    end
    # Vertical
    for y in 1:(Ly-1), x in 1:Lx
        push!(bonds, (qubit_index(x, y, Lx), qubit_index(x, y+1, Lx)))
    end
    return bonds
end

"""Classify bonds into 4 parallel substeps (square lattice coloring)."""
function classify_bonds(bonds, Lx::Int, Ly::Int)
    substeps = [Int[] for _ in 1:4]
    for (idx, (i, j)) in enumerate(bonds)
        xi = (i - 1) % Lx + 1
        yi = (i - 1) ÷ Lx + 1
        xj = (j - 1) % Lx + 1
        yj = (j - 1) ÷ Lx + 1

        if yi == yj  # Horizontal
            if min(xi, xj) % 2 == 1  # even in 0-based = odd in 1-based
                push!(substeps[1], idx)
            else
                push!(substeps[2], idx)
            end
        elseif xi == xj  # Vertical
            if min(yi, yj) % 2 == 1
                push!(substeps[3], idx)
            else
                push!(substeps[4], idx)
            end
        else
            push!(substeps[1], idx)
        end
    end
    return substeps
end

# ============================================================================
# Coupling generators
# ============================================================================

"""EA bimodal: J_ij ∈ {+1, -1} uniformly at random."""
function ea_bimodal(n_bonds::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    return [rand(rng, [-1, 1]) for _ in 1:n_bonds]
end

"""Gaussian: J_ij ~ N(0, 1)."""
function gaussian_couplings(n_bonds::Int; seed::Int=42)
    rng = MersenneTwister(seed)
    return randn(rng, n_bonds)
end

# ============================================================================
# Hamiltonian as PauliSum
# ============================================================================

"""Build Hamiltonian H = -Σ J_{ij} Z_i Z_j - h Σ X_i as a PauliSum."""
function build_hamiltonian(n_qubits::Int, bonds, J::Vector{Float64}, h::Float64)
    terms = PauliSum(n_qubits)
    # ZZ interactions
    for (idx, (i, j)) in enumerate(bonds)
        terms += -J[idx] * PauliString(n_qubits, [:Z, :Z], [i, j])
    end
    # Transverse field
    for q in 1:n_qubits
        terms += -h * PauliString(n_qubits, [:X], [q])
    end
    return terms
end

# ============================================================================
# Gate sequence builders
# ============================================================================

"""Build HVA gate sequence as Vector{PauliRotation}.

Layer structure: RX on all qubits → RZZ in 4 parallel substeps.
Each gate is a PauliRotation(θ, generator).

Returns: (gates, n_params)
"""
function build_hva_gates(n_qubits::Int, bonds, substeps, n_layers::Int, params)
    gates = []
    idx = 1
    for layer in 1:n_layers
        # RX on all qubits
        for q in 1:n_qubits
            gen = PauliString(n_qubits, [:X], [q])
            push!(gates, PauliRotation(params[idx], gen))
            idx += 1
        end
        # RZZ in 4 parallel substeps
        for step in 1:4
            for bond_idx in substeps[step]
                i, j = bonds[bond_idx]
                gen = PauliString(n_qubits, [:Z, :Z], [i, j])
                push!(gates, PauliRotation(params[idx], gen))
                idx += 1
            end
        end
    end
    return gates
end

"""Number of HVA parameters per layer."""
hva_params_per_layer(n_qubits::Int, n_bonds::Int) = n_qubits + n_bonds

"""Build Trotter gate sequence (fixed angles, non-trainable).

2nd-order Suzuki-Trotter: exp(-iH_X dt/2) · exp(-iH_ZZ dt) · exp(-iH_X dt/2)
Each exp(-iαP) = PauliRotation(2α, P) since PauliRotation(θ,P) = exp(-iθP/2).

For H_X term -h X_i:  exp(-i(-h X_i)dt) = exp(ihX_i dt)
    → PauliRotation(-2h·dt, X_i)

For H_ZZ term -J_ij Z_iZ_j: exp(-i(-J_ij Z_iZ_j)dt) = exp(iJ_ij Z_iZ_j dt)
    → PauliRotation(-2J_ij·dt, Z_iZ_j)
"""
function build_trotter_gates(
    n_qubits::Int, bonds, substeps,
    J::Vector{Float64}, h::Float64,
    dt::Float64, n_steps::Int;
    order::Int=2
)
    gates = []
    for _ in 1:n_steps
        if order == 1
            # H_ZZ part
            for step in 1:4
                for bond_idx in substeps[step]
                    i, j = bonds[bond_idx]
                    gen = PauliString(n_qubits, [:Z, :Z], [i, j])
                    push!(gates, PauliRotation(-2.0 * J[bond_idx] * dt, gen))
                end
            end
            # H_X part
            for q in 1:n_qubits
                gen = PauliString(n_qubits, [:X], [q])
                push!(gates, PauliRotation(-2.0 * h * dt, gen))
            end

        elseif order == 2
            # H_X half step
            for q in 1:n_qubits
                gen = PauliString(n_qubits, [:X], [q])
                push!(gates, PauliRotation(-h * dt, gen))
            end
            # H_ZZ full step
            for step in 1:4
                for bond_idx in substeps[step]
                    i, j = bonds[bond_idx]
                    gen = PauliString(n_qubits, [:Z, :Z], [i, j])
                    push!(gates, PauliRotation(-2.0 * J[bond_idx] * dt, gen))
                end
            end
            # H_X half step
            for q in 1:n_qubits
                gen = PauliString(n_qubits, [:X], [q])
                push!(gates, PauliRotation(-h * dt, gen))
            end
        end
    end
    return gates
end
