"""
    hamiltonian.jl — 2D spin glass model definition for the Julia pipeline

Hamiltonian: H = -Σ_{⟨i,j⟩} J_{ij} Z_i Z_j  -  h Σ_i X_i

Model provenance
----------------
The couplings are NOT generated here. `scripts/00_build_model.py` writes
`results/<Lx>x<Ly>/model_config.json`, and `load_model_config` reads it back.
Both halves of the pipeline must agree on

  * the bond ordering, since J[k] is attached to bonds[k] by index, and
  * the actual random draw, since Julia's MersenneTwister and NumPy's PCG64
    produce different sequences from the same seed.

`build_bonds` is kept only so that a config file can be validated against the
ordering it claims; it is deliberately identical to
`SpinGlass2D._build_bonds` on the Python side (horizontal bonds first, in
row-major order, then vertical bonds in *column-major* order).

Gate sequences
--------------
Two representations are produced:

  * `build_*_gate_tuples`  → the dependency-free tuple format consumed by
    `bppps_engine.jl`. This is the reference path and always works.
  * `build_*_pp_circuit`   → `(gates, thetas)` for PauliPropagation.jl, the
    optional fast path for target generation. Run
    `julia --project=julia julia/scripts/check_env.jl` once to confirm the
    installed PauliPropagation version matches this call signature before
    starting a long run.
"""

using JSON
using Random

# ============================================================================
# Lattice helpers
# ============================================================================

"""Qubit index from grid position (x,y) on an Lx×Ly lattice, 1-based."""
qubit_index(x::Int, y::Int, Lx::Int) = (y - 1) * Lx + x

"""Nearest-neighbour bonds for an Lx×Ly open-boundary square lattice.

Ordering must match Python's `SpinGlass2D._build_bonds`:
horizontal bonds row by row, then vertical bonds **column by column**.
"""
function build_bonds(Lx::Int, Ly::Int)
    bonds = Tuple{Int,Int}[]
    # Horizontal: for y, for x
    for y in 1:Ly, x in 1:(Lx - 1)
        push!(bonds, (qubit_index(x, y, Lx), qubit_index(x + 1, y, Lx)))
    end
    # Vertical: for x, for y   (x outer — this is the ordering Python uses)
    for x in 1:Lx, y in 1:(Ly - 1)
        push!(bonds, (qubit_index(x, y, Lx), qubit_index(x, y + 1, Lx)))
    end
    return bonds
end

"""Colour the bonds into 4 conflict-free substeps (square-lattice colouring).

Mirrors `hamiltonians.classify_substep_bonds`. Qubit indices here are 1-based,
so the parity tests are shifted by one relative to the Python version.
"""
function classify_bonds(bonds, Lx::Int, Ly::Int)
    substeps = [Int[] for _ in 1:4]
    for (idx, (i, j)) in enumerate(bonds)
        xi = (i - 1) % Lx + 1
        yi = (i - 1) ÷ Lx + 1
        xj = (j - 1) % Lx + 1
        yj = (j - 1) ÷ Lx + 1

        if yi == yj                       # horizontal
            push!(substeps[min(xi, xj) % 2 == 1 ? 1 : 2], idx)
        elseif xi == xj                   # vertical
            push!(substeps[min(yi, yj) % 2 == 1 ? 3 : 4], idx)
        else
            push!(substeps[1], idx)
        end
    end
    return substeps
end

# ============================================================================
# Model I/O — model_config.json is the contract with the Python pipeline
# ============================================================================

"""Model description loaded from `model_config.json`."""
struct SpinGlassModel
    Lx::Int
    Ly::Int
    n_qubits::Int
    h::Float64
    bonds::Vector{Tuple{Int,Int}}
    J::Vector{Float64}
    substeps::Vector{Vector{Int}}
end

"""Load the model written by `scripts/00_build_model.py`.

Python stores 0-based qubit indices; they are converted to 1-based here.
The stored bond list is validated against `build_bonds` so that a mismatch
fails immediately instead of silently attaching couplings to the wrong bonds.
"""
function load_model_config(path::String)
    isfile(path) || error(
        "Missing $(path). Run `python scripts/00_build_model.py` first — " *
        "the Julia side must not generate its own couplings."
    )
    cfg = JSON.parsefile(path)

    Lx = Int(cfg["Lx"])
    Ly = Int(cfg["Ly"])
    n_qubits = Int(cfg["num_qubits"])
    h = Float64(cfg["h"])

    # 0-based (Python) → 1-based (Julia)
    bonds = [(Int(b[1]) + 1, Int(b[2]) + 1) for b in cfg["bonds"]]
    J = Float64[Float64(x) for x in cfg["J"]]

    expected = build_bonds(Lx, Ly)
    bonds == expected || error(
        "Bond ordering in $(path) does not match build_bonds($(Lx),$(Ly)).\n" *
        "  config[1:5] = $(bonds[1:min(5,end)])\n" *
        "  julia [1:5] = $(expected[1:min(5,end)])\n" *
        "Both pipelines must use the same ordering; regenerate the config."
    )
    length(J) == length(bonds) || error(
        "J has $(length(J)) entries but there are $(length(bonds)) bonds."
    )

    substeps = if haskey(cfg, "substep_bonds")
        # Stored as {"1": [[bond_idx, i, j], ...]}; keep the bond indices and
        # shift them to 1-based.
        [Int[Int(entry[1]) + 1 for entry in cfg["substep_bonds"][string(s)]]
         for s in 1:4]
    else
        classify_bonds(bonds, Lx, Ly)
    end

    return SpinGlassModel(Lx, Ly, n_qubits, h, bonds, J, substeps)
end

"""Build the Hamiltonian as an SPO in the `bppps_engine.jl` dictionary format.

H = -Σ J_ij Z_i Z_j - h Σ X_i, which is exactly the operator whose expectation
value the ground-state mode minimises.
"""
function build_hamiltonian_spo(model::SpinGlassModel)
    spo = Dict{String,Float64}()
    for (idx, (i, j)) in enumerate(model.bonds)
        chars = fill('I', model.n_qubits)
        chars[i] = 'Z'
        chars[j] = 'Z'
        key = String(chars)
        spo[key] = get(spo, key, 0.0) - model.J[idx]
    end
    for q in 1:model.n_qubits
        chars = fill('I', model.n_qubits)
        chars[q] = 'X'
        key = String(chars)
        spo[key] = get(spo, key, 0.0) - model.h
    end
    return spo
end

# ============================================================================
# Gate sequences — dependency-free tuple format (bppps_engine.jl)
# ============================================================================

"""Number of HVA parameters per layer."""
hva_params_per_layer(n_qubits::Int, n_bonds::Int) = n_qubits + n_bonds

"""Append one 2nd-order Suzuki-Trotter step S₂(τ) in tuple format.

S₂(τ) = exp(-iH_X τ/2) · exp(-iH_ZZ τ) · exp(-iH_X τ/2), realised with
RX(-h·τ) and RZZ(-2J·τ) because H_X = -h ΣX and H_ZZ = -Σ J ZZ.
"""
function _append_s2_tuples!(seq, model::SpinGlassModel, tau::Float64)
    for q in 1:model.n_qubits
        push!(seq, (:rx, q, -model.h * tau, -1))
    end
    for step in 1:4, bond_idx in model.substeps[step]
        i, j = model.bonds[bond_idx]
        push!(seq, (:rzz, i, j, -2.0 * model.J[bond_idx] * tau, -1))
    end
    for q in 1:model.n_qubits
        push!(seq, (:rx, q, -model.h * tau, -1))
    end
end

"""Trotter gate sequence in tuple format, in circuit order.

Orders 1, 2 and 4 are supported; 4th order uses
S₄(dt) = S₂(p·dt)² · S₂((1-4p)·dt) · S₂(p·dt)² with p = 1/(4 - 4^{1/3}).
"""
function build_trotter_gate_tuples(model::SpinGlassModel, dt::Float64,
                                    n_steps::Int; order::Int=4)
    seq = Tuple[]
    for _ in 1:n_steps
        if order == 1
            for step in 1:4, bond_idx in model.substeps[step]
                i, j = model.bonds[bond_idx]
                push!(seq, (:rzz, i, j, -2.0 * model.J[bond_idx] * dt, -1))
            end
            for q in 1:model.n_qubits
                push!(seq, (:rx, q, -2.0 * model.h * dt, -1))
            end
        elseif order == 2
            _append_s2_tuples!(seq, model, dt)
        elseif order == 4
            p = 1.0 / (4.0 - 4.0^(1.0 / 3.0))
            _append_s2_tuples!(seq, model, p * dt)
            _append_s2_tuples!(seq, model, p * dt)
            _append_s2_tuples!(seq, model, (1.0 - 4.0 * p) * dt)
            _append_s2_tuples!(seq, model, p * dt)
            _append_s2_tuples!(seq, model, p * dt)
        else
            error("Unsupported Trotter order: $(order). Use 1, 2, or 4.")
        end
    end
    return seq
end

"""HVA parameters reproducing a 1st-order Trotter circuit of the same depth.

Mirrors `bppps.warm_start.trotter_warm_start`; see that docstring for the
derivation. BP-PPS Sec. III B reports random initialisation only converges for
2-3 layers, so this is the default initialiser.
"""
function trotter_warm_start(model::SpinGlassModel, delta_t::Float64,
                            n_layers::Int)
    n_bonds = length(model.bonds)
    per_layer = hva_params_per_layer(model.n_qubits, n_bonds)
    params = zeros(n_layers * per_layer)
    dt = delta_t / n_layers

    for layer in 1:n_layers
        offset = (layer - 1) * per_layer
        params[offset + 1 : offset + model.n_qubits] .= -2.0 * model.h * dt
        for b in 1:n_bonds
            params[offset + model.n_qubits + b] = -2.0 * model.J[b] * dt
        end
    end
    return params
end

# ============================================================================
# Gate sequences — PauliPropagation.jl (optional fast path)
# ============================================================================
#
# PauliPropagation.jl separates the circuit topology from the angles:
#
#     circuit = [PauliRotation([:X], [1]), PauliRotation([:Z,:Z], [1,2]), ...]
#     thetas  = [θ₁, θ₂, ...]
#     psum    = propagate(circuit, observable, thetas; min_abs_coeff=δ)
#
# The angle is *not* a constructor argument. Passing one, as an earlier version
# of this file did, does not match any published constructor, and calling
# `propagate` without `thetas` fails for parametrised gates. Its rotation
# convention, exp(-iθP/2), is the same one used throughout this project.

"""Build the target Trotter circuit for PauliPropagation.jl.

Returns `(circuit, thetas)` — see the note above on why the angles are
separate. Angles match `build_trotter_gate_tuples` exactly.
"""
function build_trotter_pp_circuit(model::SpinGlassModel, dt::Float64,
                                   n_steps::Int; order::Int=4)
    tuples = build_trotter_gate_tuples(model, dt, n_steps; order=order)
    circuit = Any[]
    thetas = Float64[]
    for g in tuples
        if g[1] == :rx
            push!(circuit, PauliRotation([:X], [g[2]]))
            push!(thetas, g[3])
        else  # :rzz
            push!(circuit, PauliRotation([:Z, :Z], [g[2], g[3]]))
            push!(thetas, g[4])
        end
    end
    return circuit, thetas
end
