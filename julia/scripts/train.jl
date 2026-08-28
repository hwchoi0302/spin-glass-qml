"""
    train.jl — BP-PPS training driver (Julia)

Reads the model and every setting from the results directory produced by
`python scripts/00_build_model.py`; nothing is hardcoded here and the
couplings are never regenerated on the Julia side.

Usage:
    python scripts/00_build_model.py                     # once, from Python
    julia --project=julia julia/scripts/train.jl
    julia --project=julia julia/scripts/train.jl results/10x10
"""

using JSON
using Printf

include(joinpath(@__DIR__, "..", "src", "trainer.jl"))

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function main(args)
    results_dir = if isempty(args)
        cfg_probe = joinpath(PROJECT_ROOT, "results")
        isdir(cfg_probe) || error("No results/ directory; run scripts/00_build_model.py first.")
        # Default to the directory named by the run config that exists.
        candidates = filter(d -> isfile(joinpath(cfg_probe, d, "run_config.json")),
                            readdir(cfg_probe))
        length(candidates) == 1 || error(
            "Found $(length(candidates)) result directories $(candidates); " *
            "pass one explicitly, e.g. `julia julia/scripts/train.jl results/4x4`."
        )
        joinpath(cfg_probe, candidates[1])
    else
        isabspath(args[1]) ? args[1] : joinpath(PROJECT_ROOT, args[1])
    end

    run_cfg = JSON.parsefile(joinpath(results_dir, "run_config.json"))
    model = load_model_config(joinpath(results_dir, "model_config.json"))

    ansatz = run_cfg["ansatz"]
    target = run_cfg["target"]
    opt = run_cfg["optimizer"]
    trunc = run_cfg["truncation"]

    n_layers = Int(ansatz["n_layers"])
    delta_t = Float64(target["delta_t"])

    println("=" ^ 64)
    println("BP-PPS training (Julia)  —  backward pass Eqs. 20-21")
    println("=" ^ 64)
    @printf("  results  : %s\n", results_dir)
    @printf("  model    : %dx%d (%d qubits), %d bonds, h=%.3g\n",
            model.Lx, model.Ly, model.n_qubits, length(model.bonds), model.h)
    @printf("  ansatz   : HVA %d layers, %d parameters\n", n_layers,
            n_layers * hva_params_per_layer(model.n_qubits, length(model.bonds)))
    @printf("  target   : S%d Trotter dt=%g, delta_t=%g, cutoff=%g\n",
            Int(target["trotter_order"]), target["dt"], delta_t, target["cutoff"])

    # --- Step 1: target SPOs ------------------------------------------------
    targets_path = joinpath(results_dir, "targets_dt$(delta_t).json")
    targets = if isfile(targets_path)
        println("\n[1] Loading cached targets: $(targets_path)")
        load_targets(targets_path)
    else
        println("\n[1] Generating target SPOs")
        t, _ = generate_targets(model, delta_t;
                                dt_fine=Float64(target["dt"]),
                                order=Int(target["trotter_order"]),
                                cutoff=Float64(target["cutoff"]),
                                observables=String(target["observables"]),
                                use_pauli_propagation=HAS_PAULI_PROPAGATION)
        save_targets(t, targets_path)
        t
    end

    make_state(mode, kwargs...) = TrainerState(
        model, n_layers, mode;
        delta=Float64(trunc["initial_delta"]),
        min_delta=Float64(trunc["min_delta"]),
        adaptive_delta=Bool(trunc["adaptive"]),
        delta_factor=Float64(trunc["factor"]),
        error_ratio=Float64(trunc["error_ratio"]),
        patience=Int(trunc["patience"]),
        kwargs...)

    init_cfg = opt["init"]
    params_init = if Bool(init_cfg["trotter_warm_start"])
        println("  init: Trotter warm start (dt=$(delta_t / n_layers))")
        trotter_warm_start(model, delta_t, n_layers)
    else
        scale = Float64(init_cfg["random_scale"])
        println("  init: uniform random ±$(scale)")
        rng = MersenneTwister(Int(init_cfg["seed"]))
        n = n_layers * hva_params_per_layer(model.n_qubits, length(model.bonds))
        scale .* (2.0 .* rand(rng, n) .- 1.0)
    end

    # --- Step 2: time-evolution compression ---------------------------------
    println("\n[2] Time-evolution compression")
    te_state = make_state("time_evolution"; targets=targets)
    _, te_record = optimize!(te_state, params_init;
        n_epochs=Int(opt["stage1"]["epochs"]),
        lr=Float64(opt["stage1"]["learning_rate"]),
        lbfgs_enabled=Bool(opt["stage2"]["enabled"]),
        lbfgs_max_iter=Int(opt["stage2"]["max_iter"]),
        lbfgs_tol=Float64(opt["stage2"]["tolerance_grad"]),
        save_path=joinpath(results_dir, "trained_params.json"))

    # --- Step 3: ground-state preparation -----------------------------------
    println("\n[3] Ground-state preparation")
    gs_state = make_state("ground_state"; ham_spo=build_hamiltonian_spo(model))
    gs_init = Bool(init_cfg["trotter_warm_start"]) ?
        trotter_warm_start(model, delta_t, n_layers) : params_init
    _, gs_record = optimize!(gs_state, gs_init;
        n_epochs=Int(opt["ground_state"]["epochs"]),
        lr=Float64(opt["ground_state"]["learning_rate"]),
        lbfgs_enabled=Bool(opt["stage2"]["enabled"]),
        lbfgs_max_iter=Int(opt["stage2"]["max_iter"]),
        lbfgs_tol=Float64(opt["stage2"]["tolerance_grad"]),
        save_path=joinpath(results_dir, "gs_trained_params.json"))

    println("\n" * "=" ^ 64)
    println("TRAINING COMPLETE")
    println("=" ^ 64)
    @printf("  L_XZ  : %.6f -> %.6f   (eps_trunc %.2e, delta %.1e)\n",
            te_record["losses"][1], te_record["final_loss"],
            te_record["truncation_error_estimate"], te_record["final_delta"])
    @printf("  Energy: %.6f -> %.6f   (eps_trunc %.2e, delta %.1e)\n",
            gs_record["losses"][1], gs_record["final_loss"],
            gs_record["truncation_error_estimate"], gs_record["final_delta"])
    println("  Output: $(results_dir)/")
    println("\n  Next: python scripts/01_build_hw_circuits.py")
end

main(ARGS)
