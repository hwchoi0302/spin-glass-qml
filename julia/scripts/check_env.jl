"""
    check_env.jl — validate the Julia environment before a long run

Run this once on a new machine. It takes seconds and checks the two things
that can silently waste hours:

  1. the built-in SPD engine reproduces a known-good reference (so the Julia
     and Python halves agree), and
  2. the installed PauliPropagation.jl actually matches the call signature
     used by `build_trotter_pp_circuit` / `paulisum_to_spo`.

Check 2 is the one to watch: PauliPropagation.jl separates topology from
angles — `PauliRotation(symbols, qinds)` plus a `thetas` vector passed to
`propagate` — so a constructor like `PauliRotation(theta, generator)` does not
exist and `propagate(circuit, obs)` fails for parametrised gates. If check 2
fails, target generation still works via the built-in engine; only the fast
path is lost.

Usage:
    julia --project=julia -e 'using Pkg; Pkg.instantiate()'
    julia --project=julia julia/scripts/check_env.jl
"""

using Printf

include(joinpath(@__DIR__, "..", "src", "trainer.jl"))

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

passed = Ref(0)
failed = Ref(0)

function check(name::String, f::Function)
    print("  ", rpad(name, 52))
    try
        f()
        println("OK")
        passed[] += 1
    catch err
        println("FAIL")
        println("      ", sprint(showerror, err))
        failed[] += 1
    end
end

println("=" ^ 64)
println("Julia environment check")
println("=" ^ 64)
@printf("  Julia %s, %d thread(s)\n\n", VERSION, Threads.nthreads())

# --- 1. built-in engine -----------------------------------------------------
println("[1] Built-in SPD engine (bppps_engine.jl)")

check("RX Heisenberg update: X†ZX rotation") do
    spo = apply_rx_forward(SPO("ZI" => 1.0), 1, 0.7, 0.0)
    # U†ZU = cos(θ)Z + sin(θ)Y for U = exp(-iθX/2)
    @assert isapprox(spo["ZI"], cos(0.7); atol=1e-12) "Z coefficient wrong"
    @assert isapprox(spo["YI"], sin(0.7); atol=1e-12) "Y coefficient wrong"
end

check("Gate ordering: last gate conjugated first") do
    seq = [(:rx, 1, 0.7, -1), (:rzz, 1, 2, 0.9, -1)]
    got = propagate_forward(SPO("ZI" => 1.0), seq; delta=0.0)
    # Conjugating by the RZZ first, then the RX, must reproduce this:
    step1 = apply_rzz_forward(SPO("ZI" => 1.0), 1, 2, 0.9, 0.0)
    want = apply_rx_forward(step1, 1, 0.7, 0.0)
    for (k, v) in want
        @assert isapprox(get(got, k, 0.0), v; atol=1e-12) "mismatch at $(k)"
    end
end

check("Gradient (Eq. 21) vs central difference") do
    bonds = build_bonds(2, 2)
    model = SpinGlassModel(2, 2, 4, 1.0, bonds,
                           Float64[1.0, -1.0, -1.0, 1.0],
                           classify_bonds(bonds, 2, 2))
    ham = build_hamiltonian_spo(model)
    n_layers = 2
    n_params = n_layers * hva_params_per_layer(4, length(bonds))
    rng = MersenneTwister(0)
    θ = 2.0 .* rand(rng, n_params) .- 1.0

    function energy_grad(x)
        seq = build_hva_gate_tuples(4, bonds, model.substeps, n_layers, x)
        ev = propagate_forward(ham, seq; delta=0.0)
        E = sum(a for (P, a) in ev if is_iz_only(P); init=0.0)
        seed = SPO(P => 1.0 for (P, _) in ev if is_iz_only(P))
        return E, propagate_backward(ev, seed, seq, length(x); delta=0.0)
    end

    _, g = energy_grad(θ)
    eps = 1e-6
    gfd = similar(g)
    for i in eachindex(θ)
        up = copy(θ); up[i] += eps
        dn = copy(θ); dn[i] -= eps
        gfd[i] = (energy_grad(up)[1] - energy_grad(dn)[1]) / (2eps)
    end
    rel = sqrt(sum((g .- gfd) .^ 2)) / sqrt(sum(gfd .^ 2))
    @assert rel < 1e-6 "relative gradient error $(rel)"
end

check("Truncation error estimate accumulates") do
    bonds = build_bonds(3, 3)
    model = SpinGlassModel(3, 3, 9, 1.0, bonds,
                           Float64[(-1.0)^k for k in 1:length(bonds)],
                           classify_bonds(bonds, 3, 3))
    ham = build_hamiltonian_spo(model)
    seq = build_trotter_gate_tuples(model, 0.05, 6; order=2)
    stats = TruncationStats()
    propagate_forward(ham, seq; delta=1e-3, stats=stats)
    @assert stats.n_gates > 0 "no gates counted"
    @assert error_estimate(stats) > 0 "no truncation recorded at delta=1e-3"
end

# --- 2. model config contract ----------------------------------------------
println("\n[2] Model config contract with the Python pipeline")

check("results/*/model_config.json loads and validates") do
    dirs = filter(isdir, [joinpath(PROJECT_ROOT, "results", d)
                          for d in readdir(joinpath(PROJECT_ROOT, "results"))])
    configs = filter(p -> isfile(joinpath(p, "model_config.json")), dirs)
    @assert !isempty(configs) "no model_config.json found; run scripts/00_build_model.py"
    for dir in configs
        m = load_model_config(joinpath(dir, "model_config.json"))
        @assert length(m.J) == length(m.bonds)
        @assert sum(length(s) for s in m.substeps) == length(m.bonds)
    end
end

# --- 3. PauliPropagation.jl fast path ---------------------------------------
println("\n[3] PauliPropagation.jl fast path (optional)")

if !HAS_PAULI_PROPAGATION
    println("  package not installed — the built-in engine will be used.")
    println("  To install:  julia --project=julia -e 'using Pkg; " *
            "Pkg.add(url=\"https://github.com/MSRudolph/PauliPropagation.jl\")'")
else
    check("PauliRotation(symbols, qinds) constructor") do
        PauliRotation([:X], [1])
        PauliRotation([:Z, :Z], [1, 2])
    end

    check("propagate(circuit, obs, thetas; min_abs_coeff=)") do
        circuit = [PauliRotation([:X], [1]), PauliRotation([:Z, :Z], [1, 2])]
        thetas = [0.7, 0.9]
        obs = PauliString(2, :Z, 1)
        propagate(circuit, obs, thetas; min_abs_coeff=1e-10)
    end

    check("PauliSum -> SPO conversion matches built-in engine") do
        bonds = build_bonds(2, 2)
        model = SpinGlassModel(2, 2, 4, 1.0, bonds,
                               Float64[1.0, -1.0, -1.0, 1.0],
                               classify_bonds(bonds, 2, 2))
        circuit, thetas = build_trotter_pp_circuit(model, 0.05, 4; order=2)
        obs = PauliString(4, :X, 1)
        pp = paulisum_to_spo(propagate(circuit, obs, thetas;
                                       min_abs_coeff=1e-12), 4)

        seq = build_trotter_gate_tuples(model, 0.05, 4; order=2)
        own = propagate_forward(SPO(make_obs_label(4, 'X', 1) => 1.0), seq;
                                delta=1e-12)

        for k in union(keys(pp), keys(own))
            a, b = get(pp, k, 0.0), get(own, k, 0.0)
            @assert isapprox(a, b; atol=1e-8) "$(k): PP=$(a) engine=$(b)"
        end
    end
end

println("\n" * "=" ^ 64)
@printf("  %d passed, %d failed\n", passed[], failed[])
println("=" ^ 64)
failed[] == 0 || exit(1)
