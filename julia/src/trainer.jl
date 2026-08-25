"""
    trainer.jl — BP-PPS Training using PauliPropagation.jl + Zygote AD

Supports:
1. Time-Evolution Compression: L_{X,Z} = Σ_G Σ_P (a_P - ã_P)²
2. Ground State Preparation:   L_E   = ⟨0|U†HU|0⟩

Gradients computed automatically via Zygote.jl through PauliPropagation's
differentiable `propagate` and `expectation_value` functions.
"""

using PauliPropagation
using Zygote
using Optim
using LinearAlgebra
using JSON
using Random

include("hamiltonian.jl")

# ============================================================================
# Target SPO generation
# ============================================================================

"""
Generate target SPOs by propagating local observables through fine Trotter.

For each X_i and Z_i, propagate through Trotter(dt_fine, n_steps) to get
the evolved Pauli decomposition: Ũ†X_iŨ = Σ_P ã_P P.

Returns Dict("X_1" => PauliSum, "Z_1" => PauliSum, ...).
"""
function generate_targets(
    n_qubits::Int, bonds, substeps,
    J::Vector{Float64}, h::Float64,
    delta_t::Float64;
    dt_fine::Float64=0.01,
    order::Int=2,
    cutoff::Float64=1e-6,
    observables::String="XZ",
    verbose::Bool=true
)
    n_steps = round(Int, delta_t / dt_fine)
    trotter_gates = build_trotter_gates(
        n_qubits, bonds, substeps, J, h, dt_fine, n_steps; order=order
    )

    verbose && println("  Trotter: dt=$(dt_fine), steps=$(n_steps), ",
                       "gates=$(length(trotter_gates))")

    targets = Dict{String, Any}()

    obs_list = []
    if occursin("X", observables)
        append!(obs_list, [("X", q) for q in 1:n_qubits])
    end
    if occursin("Z", observables)
        append!(obs_list, [("Z", q) for q in 1:n_qubits])
    end

    for (idx, (pauli, q)) in enumerate(obs_list)
        key = "$(pauli)_$(q)"
        sym = pauli == "X" ? :X : :Z

        # Create single-site observable
        init_obs = PauliSum(n_qubits)
        init_obs += 1.0 * PauliString(n_qubits, [sym], [q])

        # Propagate through Trotter circuit
        evolved = propagate(trotter_gates, init_obs; min_abs_coeff=cutoff)
        targets[key] = evolved

        if verbose && idx % max(1, length(obs_list) ÷ 10) == 0
            println("    [$(idx)/$(length(obs_list))] $(key): ",
                    "$(length(evolved)) terms")
        end
    end

    if verbose
        total_terms = sum(length(v) for v in values(targets))
        println("  Target generation complete: $(length(targets)) observables, ",
                "$(total_terms) total terms")
    end

    return targets
end

# ============================================================================
# Cost functions (differentiable via Zygote)
# ============================================================================

"""
Time-evolution compression loss.

L = Σ_G Σ_P (a_P^(circuit) - ã_P^(target))²

where a_P are coefficients of propagated observable through HVA,
and ã_P are target coefficients from Trotter.
"""
function time_evolution_loss(
    params,
    n_qubits::Int, bonds, substeps, n_layers::Int,
    targets::Dict;
    cutoff::Float64=1e-4
)
    gates = build_hva_gates(n_qubits, bonds, substeps, n_layers, params)

    total_loss = 0.0

    for (key, target_spo) in targets
        pauli_str, q_str = split(key, "_")
        q = parse(Int, q_str)
        sym = pauli_str == "X" ? :X : :Z

        # Initial observable
        init_obs = PauliSum(n_qubits)
        init_obs += 1.0 * PauliString(n_qubits, [sym], [q])

        # Forward propagation through HVA
        evolved = propagate(gates, init_obs; min_abs_coeff=cutoff)

        # SPO coefficient comparison loss
        # L_G = Σ_P (a_P - ã_P)²
        total_loss += spo_mse_loss(evolved, target_spo)
    end

    return total_loss
end

"""MSE loss between two PauliSums."""
function spo_mse_loss(evolved, target)
    loss = 0.0
    # Terms in evolved
    for (ps, coeff) in evolved
        target_coeff = get(target, ps, 0.0)
        loss += (coeff - target_coeff)^2
    end
    # Terms in target but not in evolved
    for (ps, coeff) in target
        if !haskey(evolved, ps)
            loss += coeff^2
        end
    end
    return loss
end

"""
Ground state energy loss.

L = ⟨0|U†HU|0⟩ = Σ_{P ∈ {I,Z}^n} a_P

where H is propagated through U(θ) and we sum coefficients
of I/Z-only Pauli strings (which have ⟨0|P|0⟩ = 1).
"""
function ground_state_loss(
    params,
    n_qubits::Int, bonds, substeps, n_layers::Int,
    hamiltonian::PauliSum;
    cutoff::Float64=1e-4
)
    gates = build_hva_gates(n_qubits, bonds, substeps, n_layers, params)
    evolved = propagate(gates, hamiltonian; min_abs_coeff=cutoff)
    return expectation_value(evolved)
end

# ============================================================================
# Training loop
# ============================================================================

"""
Train HVA parameters using L-BFGS-B with Zygote gradients.

Args:
    mode: "time_evolution" or "ground_state"
    params_init: initial parameter vector
    n_epochs: max optimization iterations
    targets: Dict of target SPOs (for time_evolution mode)
    hamiltonian: PauliSum (for ground_state mode)
    
Returns: (optimized_params, loss_history)
"""
function train_bppps(;
    n_qubits::Int,
    bonds,
    substeps,
    n_layers::Int,
    mode::String="time_evolution",
    params_init=nothing,
    n_epochs::Int=100,
    cutoff::Float64=1e-4,
    targets=nothing,
    hamiltonian=nothing,
    verbose::Bool=true,
    save_path::String=""
)
    n_params = n_layers * hva_params_per_layer(n_qubits, length(bonds))

    if params_init === nothing
        rng = MersenneTwister(42)
        params_init = 0.1 * randn(rng, n_params)
    end

    # Define cost function based on mode
    if mode == "time_evolution"
        @assert targets !== nothing "targets required for time_evolution mode"
        cost_fn = θ -> time_evolution_loss(
            θ, n_qubits, bonds, substeps, n_layers, targets; cutoff=cutoff
        )
    elseif mode == "ground_state"
        @assert hamiltonian !== nothing "hamiltonian required for ground_state mode"
        cost_fn = θ -> ground_state_loss(
            θ, n_qubits, bonds, substeps, n_layers, hamiltonian; cutoff=cutoff
        )
    else
        error("Unknown mode: $(mode)")
    end

    # Gradient function via Zygote
    grad_fn = θ -> Zygote.gradient(cost_fn, θ)[1]

    loss_history = Float64[]

    # Callback to record history
    callback = Optim.OptimizationState -> begin
        push!(loss_history, Optim.OptimizationState.value)
        if verbose && length(loss_history) % max(1, n_epochs ÷ 10) == 0
            iter = length(loss_history)
            println("  Epoch $(iter)/$(n_epochs): loss=$(loss_history[end])")
        end
        return false  # don't halt
    end

    verbose && println("Starting $(mode) training: $(n_params) params, ",
                       "$(n_epochs) iterations")

    # Run optimization
    result = optimize(
        cost_fn,
        grad_fn,
        params_init,
        LBFGS(),
        Optim.Options(
            iterations=n_epochs,
            show_trace=verbose,
            store_trace=true,
        )
    )

    opt_params = Optim.minimizer(result)
    final_loss = Optim.minimum(result)

    verbose && println("Training complete: final loss=$(final_loss)")

    # Save results
    if save_path != ""
        results = Dict(
            "optimized_params" => collect(opt_params),
            "final_loss" => final_loss,
            "n_qubits" => n_qubits,
            "n_layers" => n_layers,
            "mode" => mode,
        )
        open(save_path, "w") do f
            JSON.print(f, results, 2)
        end
        verbose && println("Results saved to $(save_path)")
    end

    return opt_params, loss_history
end

# ============================================================================
# Utility: save/load targets
# ============================================================================

"""Save target SPOs to JSON for interoperability with Python."""
function save_targets_json(targets::Dict, filepath::String, n_qubits::Int)
    serialized = Dict{String, Any}()
    for (key, spo) in targets
        terms = Dict{String, Float64}()
        for (ps, coeff) in spo
            # Convert PauliString to string label
            label = pauli_to_string(ps, n_qubits)
            terms[label] = coeff
        end
        serialized[key] = terms
    end
    open(filepath, "w") do f
        JSON.print(f, serialized, 2)
    end
    println("Targets saved to $(filepath): $(length(targets)) observables")
end

"""Convert PauliString to string label ('IXYZ...')."""
function pauli_to_string(ps::PauliString, n_qubits::Int)
    chars = fill('I', n_qubits)
    for (q, p) in ps
        if p == :X
            chars[q] = 'X'
        elseif p == :Y
            chars[q] = 'Y'
        elseif p == :Z
            chars[q] = 'Z'
        end
    end
    return String(chars)
end
