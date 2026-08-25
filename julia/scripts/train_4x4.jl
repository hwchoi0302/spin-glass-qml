"""
    train_4x4.jl — 4×4 Spin Glass Model Training Script

Step 1: Generate 4×4 EA bimodal spin glass model
Step 2: Generate target SPOs via fine Trotter propagation
Step 3: Train HVA parameters via BP-PPS (Zygote AD)
Step 4: Save trained parameters (JSON) for quantum hardware execution

Usage:
    julia --project=julia julia/scripts/train_4x4.jl
"""

using Printf

include(joinpath(@__DIR__, "..", "src", "trainer.jl"))

# ============================================================================
# Configuration
# ============================================================================
const Lx = 4
const Ly = 4
const N_QUBITS = Lx * Ly  # 16
const H_FIELD = 1.0
const COUPLING_TYPE = "ea_bimodal"
const SEED = 42

# Training parameters
const N_LAYERS = 3          # HVA layers
const DELTA_T = 0.5         # Time chunk for composition
const DT_FINE = 0.01        # Fine Trotter step for target generation
const DT_COARSE = 0.1       # Coarse step for comparison
const TROTTER_ORDER = 2
const CUTOFF_TARGET = 1e-6  # Tight cutoff for targets
const CUTOFF_TRAIN = 1e-4   # Looser cutoff for training
const N_EPOCHS = 200

const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "results", "4x4")

# ============================================================================
# Main
# ============================================================================
function main()
    println("=" ^ 60)
    println("4×4 Spin Glass BP-PPS Training")
    println("=" ^ 60)

    # Create output directory
    mkpath(OUTPUT_DIR)

    # --- Step 1: Build model ---
    println("\n[Step 1] Building 4×4 spin glass model")
    bonds = build_bonds(Lx, Ly)
    J = Float64.(ea_bimodal(length(bonds); seed=SEED))
    substeps = classify_bonds(bonds, Lx, Ly)

    println("  Qubits: $(N_QUBITS), Bonds: $(length(bonds))")
    println("  J range: [$(minimum(J)), $(maximum(J))]")
    for s in 1:4
        println("  Substep $(s): $(length(substeps[s])) bonds")
    end

    # Build Hamiltonian
    hamiltonian = build_hamiltonian(N_QUBITS, bonds, J, H_FIELD)
    println("  Hamiltonian terms: $(length(hamiltonian))")

    # Save model config
    config = Dict(
        "Lx" => Lx, "Ly" => Ly,
        "h" => H_FIELD, "coupling_type" => COUPLING_TYPE,
        "seed" => SEED, "J" => collect(J),
        "bonds" => [[b[1], b[2]] for b in bonds],
        "n_layers" => N_LAYERS, "delta_t" => DELTA_T,
    )
    open(joinpath(OUTPUT_DIR, "model_config.json"), "w") do f
        JSON.print(f, config, 2)
    end

    # --- Step 2: Generate target SPOs ---
    println("\n[Step 2] Generating target SPOs (Δt=$(DELTA_T), dt=$(DT_FINE))")
    t_start = time()
    targets = generate_targets(
        N_QUBITS, bonds, substeps, J, H_FIELD, DELTA_T;
        dt_fine=DT_FINE, order=TROTTER_ORDER,
        cutoff=CUTOFF_TARGET, verbose=true
    )
    t_target = time() - t_start
    println("  Target generation time: $(@sprintf("%.1f", t_target))s")

    # Save targets
    save_targets_json(targets, joinpath(OUTPUT_DIR, "targets_dt$(DELTA_T).json"), N_QUBITS)

    # --- Step 3: Train HVA ---
    println("\n[Step 3] Training HVA parameters ($(N_LAYERS) layers, $(N_EPOCHS) epochs)")
    n_params = N_LAYERS * hva_params_per_layer(N_QUBITS, length(bonds))
    println("  Total parameters: $(n_params)")

    t_start = time()
    opt_params, losses = train_bppps(
        n_qubits=N_QUBITS,
        bonds=bonds,
        substeps=substeps,
        n_layers=N_LAYERS,
        mode="time_evolution",
        n_epochs=N_EPOCHS,
        cutoff=CUTOFF_TRAIN,
        targets=targets,
        verbose=true,
        save_path=joinpath(OUTPUT_DIR, "trained_params.json")
    )
    t_train = time() - t_start
    println("  Training time: $(@sprintf("%.1f", t_train))s")

    # --- Step 4: Ground state training (optional) ---
    println("\n[Step 4] Ground state training")
    gs_params, gs_losses = train_bppps(
        n_qubits=N_QUBITS,
        bonds=bonds,
        substeps=substeps,
        n_layers=N_LAYERS,
        mode="ground_state",
        n_epochs=100,
        cutoff=CUTOFF_TRAIN,
        hamiltonian=hamiltonian,
        verbose=true,
        save_path=joinpath(OUTPUT_DIR, "gs_trained_params.json")
    )

    # --- Summary ---
    println("\n" * "=" ^ 60)
    println("TRAINING COMPLETE")
    println("=" ^ 60)
    println("  Time-evolution final loss: $(losses[end])")
    println("  Ground state final energy: $(gs_losses[end])")
    println("  Output directory: $(OUTPUT_DIR)")
    println("  Files:")
    for f in readdir(OUTPUT_DIR)
        println("    - $(f)")
    end
end

main()
