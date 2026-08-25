"""
    train_4x4.jl — 4×4 Spin Glass BP-PPS Training

Target generation: 4th-order Suzuki-Trotter, dt=0.001, cutoff=1e-8
Training: BP-PPS backward pass (Eq. 20-21), Adam optimizer

Usage:
    julia --project=julia julia/scripts/train_4x4.jl
"""

using Printf

include(joinpath(@__DIR__, "..", "src", "trainer.jl"))

# ============================================================================
# Configuration (matching BP-PPS paper parameters)
# ============================================================================
const Lx = 4
const Ly = 4
const N_QUBITS = Lx * Ly  # 16
const H_FIELD = 1.0
const SEED = 42

# Target generation (paper: 4th order, dt=0.001, cutoff=1e-8)
const DT_FINE = 0.001
const TROTTER_ORDER = 4
const CUTOFF_TARGET = 1e-8

# Training
const N_LAYERS = 3
const DELTA_T = 0.5
const CUTOFF_TRAIN = 1e-4
const N_EPOCHS_TE = 200     # Time-evolution epochs
const N_EPOCHS_GS = 100     # Ground-state epochs
const LR = 0.01

const OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "results", "4x4")

# ============================================================================
function main()
    println("=" ^ 60)
    println("4×4 Spin Glass BP-PPS Training")
    println("  Algorithm: BP-PPS backward pass (Eq. 20-21)")
    println("  Target: 4th-order Suzuki-Trotter, dt=$(DT_FINE)")
    println("=" ^ 60)

    mkpath(OUTPUT_DIR)

    # --- Step 1: Build model ---
    println("\n[Step 1] Building model")
    bonds = build_bonds(Lx, Ly)
    J = Float64.(ea_bimodal(length(bonds); seed=SEED))
    substeps = classify_bonds(bonds, Lx, Ly)
    n_bonds = length(bonds)

    println("  Qubits: $(N_QUBITS), Bonds: $(n_bonds)")
    n_params = N_LAYERS * hva_params_per_layer(N_QUBITS, n_bonds)
    println("  HVA params: $(n_params) ($(N_LAYERS) layers × $(N_QUBITS + n_bonds))")

    # Save model config
    config = Dict(
        "Lx" => Lx, "Ly" => Ly, "h" => H_FIELD, "seed" => SEED,
        "J" => collect(J), "bonds" => [[b[1], b[2]] for b in bonds],
        "n_layers" => N_LAYERS, "delta_t" => DELTA_T,
        "dt_fine" => DT_FINE, "trotter_order" => TROTTER_ORDER,
        "cutoff_target" => CUTOFF_TARGET,
    )
    open(joinpath(OUTPUT_DIR, "model_config.json"), "w") do f
        JSON.print(f, config, 2)
    end

    # --- Step 2: Generate targets ---
    println("\n[Step 2] Generating target SPOs")
    println("  Δt=$(DELTA_T), dt=$(DT_FINE), order=$(TROTTER_ORDER), cutoff=$(CUTOFF_TARGET)")

    targets = generate_targets(
        N_QUBITS, bonds, substeps, J, H_FIELD, DELTA_T;
        dt_fine=DT_FINE, order=TROTTER_ORDER,
        cutoff=CUTOFF_TARGET, verbose=true
    )
    save_targets(targets, joinpath(OUTPUT_DIR, "targets_dt$(DELTA_T).json"))

    # --- Step 3: Time-evolution training ---
    println("\n[Step 3] Time-evolution training ($(N_EPOCHS_TE) epochs)")
    te_params, te_losses = train_adam(
        n_qubits=N_QUBITS, bonds=bonds, substeps=substeps,
        n_layers=N_LAYERS, mode="time_evolution",
        targets=targets, n_epochs=N_EPOCHS_TE, lr=LR,
        delta=CUTOFF_TRAIN, verbose=true,
        save_path=joinpath(OUTPUT_DIR, "trained_params.json")
    )

    # --- Step 4: Ground state training ---
    println("\n[Step 4] Ground state training ($(N_EPOCHS_GS) epochs)")

    # Build Hamiltonian SPO
    ham_spo = SPO()
    for (idx, (i, j)) in enumerate(bonds)
        label = make_obs_label(N_QUBITS, 'I', 1)  # start with all I
        chars = fill('I', N_QUBITS)
        chars[i] = 'Z'; chars[j] = 'Z'
        ham_spo[String(chars)] = get(ham_spo, String(chars), 0.0) - J[idx]
    end
    for q in 1:N_QUBITS
        label = make_obs_label(N_QUBITS, 'X', q)
        ham_spo[label] = get(ham_spo, label, 0.0) - H_FIELD
    end

    gs_params, gs_losses = train_adam(
        n_qubits=N_QUBITS, bonds=bonds, substeps=substeps,
        n_layers=N_LAYERS, mode="ground_state",
        ham_spo=ham_spo, n_epochs=N_EPOCHS_GS, lr=0.05,
        delta=CUTOFF_TRAIN, verbose=true,
        save_path=joinpath(OUTPUT_DIR, "gs_trained_params.json")
    )

    # --- Summary ---
    println("\n" * "=" ^ 60)
    println("TRAINING COMPLETE")
    println("=" ^ 60)
    @printf("  Time-evolution: loss %.6f → %.6f (%.1f%% reduction)\n",
            te_losses[1], te_losses[end],
            (1 - te_losses[end] / te_losses[1]) * 100)
    @printf("  Ground state:   E = %.6f\n", gs_losses[end])
    println("  Output: $(OUTPUT_DIR)/")
end

main()
