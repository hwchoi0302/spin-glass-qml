"""
    trainer.jl — BP-PPS training loop (Julia)

Mirrors `src/bppps/trainer.py`. Gradients come from the backward pass of
`bppps_engine.jl` (Eqs. 20-21), never from automatic differentiation.

Loss functions
    Time-evolution compression (Eq. 30):
        L_{X,Z} = Σ_G ‖G(t) - G̃(t)‖²_rHS = Σ_G Σ_P (a_P - ã_P)²
      Pauli strings are orthonormal under the rescaled Hilbert-Schmidt inner
      product, so the operator distance is the l2 distance of the coefficients.

    Ground-state preparation (Sec. III A):
        E(θ) = ⟨0|U(θ)† H U(θ)|0⟩ = Σ_{P ∈ {I,Z}ⁿ} a_P
      The operator propagated forward is H itself — in the Heisenberg picture
      the circuit is moved onto the observable rather than onto the state.

Optimisation follows the paper's recipe: Adam first, then L-BFGS to converge,
which is worthwhile because BP-PPS supplies exact analytic gradients.

Every propagation is instrumented with the Appendix B truncation-error
estimate, and the threshold can tighten itself when that estimate grows large
compared to the quantity being optimised.

Target generation uses `bppps_engine.jl` by default (no external
dependencies). Set `use_pauli_propagation=true` to take the faster
PauliPropagation.jl path; run `julia/scripts/check_env.jl` first to confirm
the installed version matches the call signature.
"""

using JSON
using LinearAlgebra
using Optim
using Printf
using Random

include("hamiltonian.jl")
include("bppps_engine.jl")

# PauliPropagation.jl is an optional accelerator for target generation only.
# Loading it must not be able to break the rest of the pipeline, so the import
# is guarded: if the package is missing or fails to precompile, the built-in
# engine is used instead.
const HAS_PAULI_PROPAGATION = try
    @eval using PauliPropagation
    true
catch err
    @warn "PauliPropagation.jl unavailable; falling back to bppps_engine.jl \
for target generation (slower but dependency-free)." exception = err
    false
end

# ============================================================================
# Target SPO generation
# ============================================================================

"""Generate the target SPOs G̃ = V† G V for every local X and Z observable.

Uses a 4th-order Suzuki-Trotter approximation of V = exp(-iH·Δt) with a fine
step and a tight threshold — the paper's values are dt = 0.001 and δ = 1e-8,
which makes the target numerically exact rather than a hardware circuit.
"""
function generate_targets(model::SpinGlassModel, delta_t::Float64;
                          dt_fine::Float64=0.001,
                          order::Int=4,
                          cutoff::Float64=1e-8,
                          observables::String="XZ",
                          use_pauli_propagation::Bool=false,
                          verbose::Bool=true)
    n_steps = round(Int, delta_t / dt_fine)
    gate_tuples = build_trotter_gate_tuples(model, dt_fine, n_steps; order=order)

    if use_pauli_propagation && !HAS_PAULI_PROPAGATION
        @warn "use_pauli_propagation=true but the package is unavailable; \
using bppps_engine.jl instead."
        use_pauli_propagation = false
    end

    if verbose
        println("  Trotter: order=$(order), dt=$(dt_fine), steps=$(n_steps)")
        println("  Gates:   $(length(gate_tuples))")
        println("  Backend: ", use_pauli_propagation ?
                "PauliPropagation.jl" : "bppps_engine.jl")
    end

    obs_list = Tuple{Char,Int}[]
    occursin("X", observables) && append!(obs_list, [('X', q) for q in 1:model.n_qubits])
    occursin("Z", observables) && append!(obs_list, [('Z', q) for q in 1:model.n_qubits])

    targets = Dict{String, SPO}()
    stats = TruncationStats()
    t_start = time()

    pp_circuit, pp_thetas = if use_pauli_propagation
        build_trotter_pp_circuit(model, dt_fine, n_steps; order=order)
    else
        (nothing, nothing)
    end

    for (idx, (pauli, q)) in enumerate(obs_list)
        key = "$(pauli)_$(q)"

        if use_pauli_propagation
            sym = pauli == 'X' ? :X : :Z
            obs = PauliString(model.n_qubits, sym, q)
            evolved = propagate(pp_circuit, obs, pp_thetas; min_abs_coeff=cutoff)
            targets[key] = paulisum_to_spo(evolved, model.n_qubits)
        else
            init_spo = SPO(make_obs_label(model.n_qubits, pauli, q) => 1.0)
            targets[key] = propagate_forward(init_spo, gate_tuples;
                                             delta=cutoff, stats=stats)
        end

        if verbose && idx % max(1, length(obs_list) ÷ 5) == 0
            @printf("    [%d/%d] %s: %d terms, elapsed=%.1fs\n",
                    idx, length(obs_list), key, length(targets[key]),
                    time() - t_start)
        end
    end

    if verbose
        total = sum(length(v) for v in values(targets))
        @printf("  Complete: %d observables, %d terms, %.1fs\n",
                length(targets), total, time() - t_start)
        if !use_pauli_propagation
            @printf("  Truncation error estimate (Eq. B16): %.3e\n",
                    error_estimate(stats))
        end
    end

    return targets, stats
end

"""Convert a PauliPropagation.jl PauliSum to the Dict{String,Float64} format.

Iterating a PauliSum yields (packed Pauli integer, coefficient); `getpauli`
extracts the single-site Pauli at a given index and `inttosymbol` names it.
Validated by `julia/scripts/check_env.jl`.
"""
function paulisum_to_spo(psum, n_qubits::Int)
    spo = SPO()
    for (pstr, coeff) in psum
        chars = fill('I', n_qubits)
        for q in 1:n_qubits
            sym = inttosymbol(getpauli(pstr, q))
            sym == :I || (chars[q] = first(String(sym)))
        end
        spo[String(chars)] = Float64(real(coeff))
    end
    return spo
end

# ============================================================================
# Training steps
# ============================================================================

"""Mutable optimisation state shared by the two loss modes."""
mutable struct TrainerState
    model::SpinGlassModel
    n_layers::Int
    n_params::Int
    mode::String
    targets::Union{Dict{String,SPO}, Nothing}
    ham_spo::Union{SPO, Nothing}
    delta::Float64
    min_delta::Float64
    adaptive_delta::Bool
    delta_factor::Float64
    error_ratio::Float64
    patience::Int
    last_error_estimate::Float64
    delta_history::Vector{Tuple{Int,Float64}}
end

function TrainerState(model::SpinGlassModel, n_layers::Int, mode::String;
                      targets=nothing, ham_spo=nothing,
                      delta::Float64=1e-3, min_delta::Float64=1e-5,
                      adaptive_delta::Bool=true, delta_factor::Float64=0.1,
                      error_ratio::Float64=0.1, patience::Int=10)
    n_params = n_layers * hva_params_per_layer(model.n_qubits,
                                               length(model.bonds))
    TrainerState(model, n_layers, n_params, mode, targets, ham_spo,
                 delta, min_delta, adaptive_delta, delta_factor,
                 error_ratio, patience, 0.0, Tuple{Int,Float64}[])
end

"""One training step for time-evolution compression.

The loss counts the full union of the evolved and target supports — a Pauli
present only in the target is a genuine part of the error. The gradient seed
is restricted to the evolved support, following BP-PPS Sec. III A ("we perform
optimization by propagating only the derivative with non-zero coefficient
support"); the joint truncation rule in the backward pass would prune anything
else at the first gate anyway.
"""
function time_evolution_step(state::TrainerState, params::Vector{Float64})
    gate_seq = build_hva_gate_tuples(state.model.n_qubits, state.model.bonds,
                                     state.model.substeps, state.n_layers,
                                     params)
    stats = TruncationStats()
    total_loss = 0.0
    total_grad = zeros(state.n_params)

    for (obs_key, target_spo) in state.targets
        parts = split(obs_key, "_")
        pauli = first(parts[1])
        q = parse(Int, parts[2])

        init_spo = SPO(make_obs_label(state.model.n_qubits, pauli, q) => 1.0)
        evolved = propagate_forward(init_spo, gate_seq;
                                    delta=state.delta, stats=stats)

        loss_g = 0.0
        seed = SPO()
        for (P, a_P) in evolved
            diff = a_P - get(target_spo, P, 0.0)
            loss_g += diff * diff
            diff != 0.0 && (seed[P] = 2.0 * diff)
        end
        for (P, a_target) in target_spo
            haskey(evolved, P) || (loss_g += a_target * a_target)
        end
        total_loss += loss_g

        total_grad .+= propagate_backward(evolved, seed, gate_seq,
                                          state.n_params;
                                          delta=state.delta, stats=stats)
    end

    state.last_error_estimate = error_estimate(stats)
    return total_loss, total_grad
end

"""One training step for ground-state preparation.

Seed (Eq. 24): ∂L/∂a_P = 1 on the I/Z strings. That vector is dense over 2ⁿ
strings, so — as the paper does — it is restricted to the strings the
propagated operator actually carries.
"""
function ground_state_step(state::TrainerState, params::Vector{Float64})
    gate_seq = build_hva_gate_tuples(state.model.n_qubits, state.model.bonds,
                                     state.model.substeps, state.n_layers,
                                     params)
    stats = TruncationStats()

    evolved = propagate_forward(state.ham_spo, gate_seq;
                                delta=state.delta, stats=stats)
    energy = sum(a for (P, a) in evolved if is_iz_only(P); init=0.0)
    seed = SPO(P => 1.0 for (P, _) in evolved if is_iz_only(P))

    grad = propagate_backward(evolved, seed, gate_seq, state.n_params;
                              delta=state.delta, stats=stats)

    state.last_error_estimate = error_estimate(stats)
    return energy, grad
end

"""Dispatch one training step by mode."""
function train_step(state::TrainerState, params::Vector{Float64})
    if state.mode == "time_evolution"
        return time_evolution_step(state, params)
    elseif state.mode == "ground_state"
        return ground_state_step(state, params)
    end
    error("Unknown mode: $(state.mode)")
end

"""Coefficient-space quantity the tracked truncation error is compared to.

ε_emp is an l2 norm of discarded coefficient weight, so it must be compared
against something in the same units: |E| for the (linear) energy, sqrt(L) for
the (quadratic) L_{X,Z} cost.
"""
function error_scale(state::TrainerState, loss::Float64)
    state.mode == "ground_state" ? abs(loss) : sqrt(max(loss, 0.0))
end

"""Tighten δ when the tracked error starts to dominate the residual."""
function maybe_tighten_delta!(state::TrainerState, epoch::Int, loss::Float64,
                              last_change::Int)
    (!state.adaptive_delta || state.delta <= state.min_delta) && return false, last_change
    epoch - last_change < state.patience && return false, last_change
    state.last_error_estimate <= state.error_ratio * error_scale(state, loss) &&
        return false, last_change

    state.delta = max(state.min_delta, state.delta * state.delta_factor)
    push!(state.delta_history, (epoch, state.delta))
    return true, epoch
end

# ============================================================================
# Stage 1: Adam
# ============================================================================

function train_adam!(state::TrainerState, params::Vector{Float64};
                     n_epochs::Int=200, lr::Float64=0.01,
                     β1::Float64=0.9, β2::Float64=0.999, ε::Float64=1e-8,
                     verbose::Bool=true)
    m = zeros(state.n_params)
    v = zeros(state.n_params)
    losses = Float64[]
    t_start = time()
    last_change = 0

    for epoch in 1:n_epochs
        loss, grad = train_step(state, params)
        push!(losses, loss)

        m .= β1 .* m .+ (1 - β1) .* grad
        v .= β2 .* v .+ (1 - β2) .* grad .^ 2
        params .-= lr .* (m ./ (1 - β1^epoch)) ./ (sqrt.(v ./ (1 - β2^epoch)) .+ ε)

        tightened, last_change = maybe_tighten_delta!(state, epoch, loss, last_change)
        if tightened && verbose
            @printf("    [epoch %d] truncation error %.2e exceeded threshold; delta -> %.1e\n",
                    epoch, state.last_error_estimate, state.delta)
        end

        if verbose && (epoch % max(1, n_epochs ÷ 10) == 0 || epoch == 1)
            @printf("  Epoch %4d/%d: loss=%.6f, |grad|=%.4f, eps_trunc=%.2e, delta=%.1e, time=%.1fs\n",
                    epoch, n_epochs, loss, norm(grad),
                    state.last_error_estimate, state.delta, time() - t_start)
        end
    end
    return params, losses
end

# ============================================================================
# Stage 2: L-BFGS
# ============================================================================

"""Quasi-Newton refinement on the exact analytic gradient.

δ is held fixed here: L-BFGS builds a curvature model across iterations and a
changing threshold would make the objective non-stationary.
"""
function refine_lbfgs!(state::TrainerState, params::Vector{Float64};
                       max_iter::Int=200, tolerance_grad::Float64=1e-6,
                       verbose::Bool=true)
    history = Float64[]
    was_adaptive = state.adaptive_delta
    state.adaptive_delta = false

    function fg!(F, G, x)
        loss, grad = train_step(state, collect(x))
        G === nothing || (G .= grad)
        push!(history, loss)
        return loss
    end

    t_start = time()
    result = Optim.optimize(Optim.only_fg!(fg!), copy(params), Optim.LBFGS(),
                            Optim.Options(iterations=max_iter,
                                          g_tol=tolerance_grad))
    state.adaptive_delta = was_adaptive
    best = Optim.minimizer(result)

    if verbose
        @printf("  L-BFGS: %d iterations, loss %.6f -> %.6f, eps_trunc=%.2e, time=%.1fs\n",
                Optim.iterations(result), history[1], Optim.minimum(result),
                state.last_error_estimate, time() - t_start)
        println("    converged: ", Optim.converged(result))
    end

    return best, history
end

# ============================================================================
# Full two-stage optimisation
# ============================================================================

"""Run the configured Adam → L-BFGS schedule and optionally save the record."""
function optimize!(state::TrainerState, params_init::Vector{Float64};
                   n_epochs::Int=200, lr::Float64=0.01,
                   lbfgs_enabled::Bool=true, lbfgs_max_iter::Int=200,
                   lbfgs_tol::Float64=1e-6,
                   verbose::Bool=true, save_path::String="")
    t_start = time()
    params = copy(params_init)
    params, adam_losses = train_adam!(state, params;
                                      n_epochs=n_epochs, lr=lr, verbose=verbose)

    lbfgs_losses = Float64[]
    if lbfgs_enabled
        verbose && println("\n  --- Stage 2: L-BFGS ---")
        params, lbfgs_losses = refine_lbfgs!(state, params;
                                             max_iter=lbfgs_max_iter,
                                             tolerance_grad=lbfgs_tol,
                                             verbose=verbose)
    end

    final_loss = isempty(lbfgs_losses) ? adam_losses[end] : lbfgs_losses[end]
    record = Dict(
        # "params" is the key the Python side reads; keep both for
        # compatibility with older result files.
        "params" => collect(params),
        "optimized_params" => collect(params),
        "adam_losses" => adam_losses,
        "lbfgs_losses" => lbfgs_losses,
        "losses" => vcat(adam_losses, lbfgs_losses),
        "final_loss" => final_loss,
        "n_qubits" => state.model.n_qubits,
        "n_layers" => state.n_layers,
        "n_params" => state.n_params,
        "mode" => state.mode,
        "final_delta" => state.delta,
        "delta_history" => [[e, d] for (e, d) in state.delta_history],
        "truncation_error_estimate" => state.last_error_estimate,
        "training_time_s" => time() - t_start,
        "algorithm" => "BP-PPS (Eqs. 20-21)",
    )

    if save_path != ""
        open(save_path, "w") do f
            JSON.print(f, record, 2)
        end
        verbose && println("  Results saved to $(save_path)")
    end

    return params, record
end

# ============================================================================
# Target I/O
# ============================================================================

function save_targets(targets::Dict{String,SPO}, filepath::String)
    open(filepath, "w") do f
        JSON.print(f, targets, 2)
    end
    println("  Targets saved to $(filepath)")
end

function load_targets(filepath::String)::Dict{String,SPO}
    data = JSON.parsefile(filepath)
    targets = Dict{String, SPO}()
    for (key, terms) in data
        targets[key] = SPO(k => Float64(v) for (k, v) in terms)
    end
    return targets
end
