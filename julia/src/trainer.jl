"""
    trainer.jl — BP-PPS Training Loop

Uses the proper BP-PPS backward pass (Eq. 20-21), NOT Zygote AD.
- Target generation: PauliPropagation.jl (high-performance forward propagation)
- HVA training: bppps_engine.jl (custom backward pass, O(N_P) memory)

Target generation parameters (matching paper):
    - 4th-order Suzuki-Trotter
    - dt = 0.001
    - cutoff = 1e-8

Loss function (Eq. 30):
    L_{X,Z} = Σ_i ||X_i(t) - X̃_i(t)||²_rHS + ||Z_i(t) - Z̃_i(t)||²_rHS
            = Σ_i Σ_P (a_P - ã_P)²

    The rHS norm makes Pauli strings orthonormal, so ||O-Õ||²_rHS = Σ_P (a_P - ã_P)².
"""

using PauliPropagation
using LinearAlgebra
using JSON
using Random
using Printf

include("hamiltonian.jl")
include("bppps_engine.jl")

# ============================================================================
# Target SPO generation (using PauliPropagation.jl for performance)
# ============================================================================

"""
Generate target SPOs via PauliPropagation.jl.

Uses 4th-order Suzuki-Trotter with dt=0.001 (matching the paper).
The result is converted to our Dict{String, Float64} format for training.
"""
function generate_targets(
    n_qubits::Int, bonds, substeps,
    J::Vector{Float64}, h::Float64,
    delta_t::Float64;
    dt_fine::Float64=0.001,    # Paper: dt = 0.001
    order::Int=4,              # Paper: 4th-order Suzuki-Trotter
    cutoff::Float64=1e-8,      # Paper: δ = 1e-8
    observables::String="XZ",
    verbose::Bool=true
)
    n_steps = round(Int, delta_t / dt_fine)

    verbose && println("  Trotter: order=$(order), dt=$(dt_fine), steps=$(n_steps)")

    # Build Trotter gate sequence using PauliPropagation.jl types
    trotter_gates = build_trotter_gates(
        n_qubits, bonds, substeps, J, h, dt_fine, n_steps; order=order
    )
    verbose && println("  Total gates: $(length(trotter_gates))")

    targets = Dict{String, SPO}()

    # Build observable list
    obs_list = Tuple{String, Int}[]
    if occursin("X", observables)
        append!(obs_list, [("X", q) for q in 1:n_qubits])
    end
    if occursin("Z", observables)
        append!(obs_list, [("Z", q) for q in 1:n_qubits])
    end

    t_start = time()
    for (idx, (pauli, q)) in enumerate(obs_list)
        key = "$(pauli)_$(q)"
        sym = pauli == "X" ? :X : :Z

        # Create observable using PauliPropagation.jl
        init_obs = PauliSum(n_qubits)
        init_obs += 1.0 * PauliString(n_qubits, [sym], [q])

        # Forward propagation (using PauliPropagation.jl — optimized)
        evolved = propagate(trotter_gates, init_obs; min_abs_coeff=cutoff)

        # Convert PauliSum → Dict{String, Float64}
        target_spo = paulisum_to_spo(evolved, n_qubits)
        targets[key] = target_spo

        if verbose && idx % max(1, length(obs_list) ÷ 5) == 0
            elapsed = time() - t_start
            println("    [$(idx)/$(length(obs_list))] $(key): ",
                    "$(length(target_spo)) terms, elapsed=$(round(elapsed, digits=1))s")
        end
    end

    if verbose
        total_terms = sum(length(v) for v in values(targets))
        elapsed = time() - t_start
        println("  Target generation complete: $(length(targets)) observables, ",
                "$(total_terms) total terms, $(round(elapsed, digits=1))s")
    end

    return targets
end

"""Convert PauliPropagation.jl PauliSum to Dict{String, Float64} SPO."""
function paulisum_to_spo(psum, n_qubits::Int)
    spo = SPO()
    for (ps, coeff) in psum
        label = pauli_string_to_label(ps, n_qubits)
        spo[label] = Float64(real(coeff))
    end
    return spo
end

"""Convert PauliString to string label."""
function pauli_string_to_label(ps, n_qubits::Int)
    chars = fill('I', n_qubits)
    # PauliString iteration gives (qubit_index, pauli_symbol) pairs
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

# ============================================================================
# Training step (using BP-PPS backward pass)
# ============================================================================

"""
One training step for time-evolution compression.

Loss: L_{X,Z} = Σ_G Σ_P (a_P - ã_P)²  (rHS norm squared)
Seed: ∂L/∂a_P = 2(a_P - ã_P)

Uses bppps_engine.jl for forward + backward.
"""
function time_evolution_step(params::Vector{Float64},
                              n_qubits::Int, bonds, substeps,
                              n_layers::Int, targets::Dict{String, SPO};
                              delta::Float64=1e-4)
    n_params = length(params)
    gate_seq = build_hva_gate_tuples(n_qubits, bonds, substeps, n_layers, params)

    total_loss = 0.0
    total_grad = zeros(n_params)

    for (obs_key, target_spo) in targets
        parts = split(obs_key, "_")
        pauli_char = first(parts[1])
        q = parse(Int, parts[2])

        # Initial observable
        init_label = make_obs_label(n_qubits, pauli_char, q)
        init_spo = SPO(init_label => 1.0)

        # Forward pass
        evolved = propagate_forward(init_spo, gate_seq; delta=delta)

        # Compute loss: Σ_P (a_P - ã_P)²
        # Compute seed: ∂L/∂a_P = 2(a_P - ã_P)
        all_paulis = union(keys(evolved), keys(target_spo))
        loss_g = 0.0
        seed = SPO()

        for P in all_paulis
            a_P = get(evolved, P, 0.0)
            a_target = get(target_spo, P, 0.0)
            diff = a_P - a_target
            loss_g += diff^2
            abs(diff) > 1e-15 && (seed[P] = 2.0 * diff)
        end

        total_loss += loss_g

        # Backward pass (Eq. 20-21)
        grad_g = propagate_backward(evolved, seed, gate_seq, n_params; delta=delta)
        total_grad .+= grad_g
    end

    return total_loss, total_grad
end

"""One training step for ground state preparation.

Loss: E(θ) = ⟨0|U†HU|0⟩ = Σ_{P∈{I,Z}^n} a_P
Seed: ∂L/∂a_P = 1 if P ∈ {I,Z}^n, else 0
"""
function ground_state_step(params::Vector{Float64},
                            n_qubits::Int, bonds, substeps,
                            n_layers::Int, ham_spo::SPO;
                            delta::Float64=1e-4)
    n_params = length(params)
    gate_seq = build_hva_gate_tuples(n_qubits, bonds, substeps, n_layers, params)

    # Forward: propagate H through HVA
    evolved = propagate_forward(ham_spo, gate_seq; delta=delta)

    # Energy = Σ_{P∈{I,Z}^n} a_P
    energy = sum(a for (P, a) in evolved if is_iz_only(P); init=0.0)

    # Seed: 1 for I,Z-only strings
    seed = SPO(P => 1.0 for (P, _) in evolved if is_iz_only(P))

    # Backward
    grad = propagate_backward(evolved, seed, gate_seq, n_params; delta=delta)

    return energy, grad
end

# ============================================================================
# Adam optimizer + training loop
# ============================================================================

"""Train with Adam optimizer.

Returns (optimized_params, loss_history).
"""
function train_adam(;
    n_qubits::Int,
    bonds,
    substeps,
    n_layers::Int,
    mode::String="time_evolution",
    targets::Union{Dict{String,SPO}, Nothing}=nothing,
    ham_spo::Union{SPO, Nothing}=nothing,
    n_epochs::Int=200,
    lr::Float64=0.01,
    delta::Float64=1e-4,
    params_init::Union{Vector{Float64}, Nothing}=nothing,
    verbose::Bool=true,
    save_path::String=""
)
    n_params = n_layers * hva_params_per_layer(n_qubits, length(bonds))

    if params_init === nothing
        rng = MersenneTwister(42)
        params_init = 0.1 * randn(rng, n_params)
    end
    params = copy(params_init)

    # Adam state
    m = zeros(n_params)
    v = zeros(n_params)
    β1, β2, ε = 0.9, 0.999, 1e-8

    losses = Float64[]
    t_start = time()

    for epoch in 1:n_epochs
        if mode == "time_evolution"
            loss, grad = time_evolution_step(
                params, n_qubits, bonds, substeps, n_layers, targets; delta=delta
            )
        elseif mode == "ground_state"
            loss, grad = ground_state_step(
                params, n_qubits, bonds, substeps, n_layers, ham_spo; delta=delta
            )
        else
            error("Unknown mode: $(mode)")
        end

        push!(losses, loss)

        # Adam update
        m .= β1 .* m .+ (1 - β1) .* grad
        v .= β2 .* v .+ (1 - β2) .* grad.^2
        m_hat = m ./ (1 - β1^epoch)
        v_hat = v ./ (1 - β2^epoch)
        params .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ε)

        if verbose && (epoch % max(1, n_epochs ÷ 10) == 0 || epoch == 1)
            elapsed = time() - t_start
            @printf("  Epoch %4d/%d: loss=%.6f, |grad|=%.4f, time=%.1fs\n",
                    epoch, n_epochs, loss, norm(grad), elapsed)
        end
    end

    # Save results
    if save_path != ""
        results = Dict(
            "optimized_params" => collect(params),
            "final_loss" => losses[end],
            "loss_history" => losses,
            "n_qubits" => n_qubits,
            "n_layers" => n_layers,
            "mode" => mode,
            "n_epochs" => n_epochs,
            "algorithm" => "BP-PPS (Eq. 20-21)",
        )
        open(save_path, "w") do f
            JSON.print(f, results, 2)
        end
        verbose && println("  Results saved to $(save_path)")
    end

    return params, losses
end

# ============================================================================
# Target I/O
# ============================================================================

"""Save targets to JSON."""
function save_targets(targets::Dict{String,SPO}, filepath::String)
    open(filepath, "w") do f
        JSON.print(f, targets, 2)
    end
    println("  Targets saved to $(filepath)")
end

"""Load targets from JSON."""
function load_targets(filepath::String)::Dict{String,SPO}
    data = JSON.parsefile(filepath)
    targets = Dict{String, SPO}()
    for (key, terms) in data
        targets[key] = SPO(k => Float64(v) for (k, v) in terms)
    end
    return targets
end
