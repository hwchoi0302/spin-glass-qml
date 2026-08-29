# 🌀 spin-glass-qml

> **Simulating 2D Spin Glass Dynamics via Backpropagating Pauli Circuit Compression & Quantum Advantage Verification**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.1%2B-6133ff.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository provides an end-to-end framework to simulate **2D Spin Glass model dynamics and ground state preparation** by compressing deep Trotter circuits into a shallow Hamiltonian Variational Ansatz (HVA) on 2D square lattice processors (e.g., IBM Nighthawk).

### Key Methodologies
1. **Train on Classical, Deploy on Quantum (TCDQ):** Perform parameter optimization classically, and leverage quantum processors exclusively for global entanglement sampling.
2. **Backpropagating Pauli Propagation (BP-PPS):** memory-efficient analytic gradients that reconstruct intermediate operators from circuit reversibility instead of caching them, with barren plateaus avoided by optimising a local operator error rather than a global distribution distance.
3. **Generative Quantum Advantage Verification:** Cross-validating quantum bitstrings against Classical Shadows and demonstrating OTOC decay beyond classical tensor network (PEPS) breakdown thresholds.

---

## 🗂 Project Structure

```text
spin-glass-qml/
├── configs/                      # Every runtime setting lives here
│   ├── model_2d_spinglass.yaml   # lattice size, couplings, transverse field
│   ├── ansatz_ibm.yaml           # HVA layers, target Trotter, hardware Trotter
│   ├── optimizer.yaml            # Adam -> L-BFGS-B, init strategy, truncation
│   └── time_sweep.yaml           # T1-T sweep: times, layers, delta ladder
│
├── src/
│   ├── config.py                 # YAML loader, --set overrides
│   ├── hamiltonians/             # 2D spin glass H, frustration, bond colouring
│   ├── ansatz/                   # HVA (RX + RZZ) and Qiskit Trotter circuits
│   ├── bppps/                    # sparse Pauli dynamics + BP-PPS backward pass
│   │   ├── propagation.py        #   Eqs. 6-8, 20-21, Appendix B error estimate
│   │   ├── trainer.py            #   L_{X,Z} and energy losses, two-stage optimiser
│   │   ├── target_generator.py   #   precision Trotter targets
│   │   ├── warm_start.py         #   Trotter warm start (Sec. III B)
│   │   └── ose_regularizer.py    #   operator stabilizer entropy (Eqs. 31-32)
│   ├── classical_bench/          # exact diagonalisation baseline (<= 22 qubits)
│   ├── hardware/                 # (planned) IBM Runtime deployment, OTOC
│   └── validation/               # (planned) classical shadows cross-validation
│
├── scripts/
│   ├── 00_validate_small.py      # 13 unit tests, incl. gradient vs finite diff
│   ├── 00_build_model.py         # writes results/<Lx>x<Ly>/model_config.json
│   ├── run_pipeline.py           # targets -> baselines -> training -> validation
│   ├── 01_build_hw_circuits.py   # QASM for IBM hardware + depth comparison
│   ├── 02_time_sweep.py          # T1-T: advantage-window search over T
│   ├── 00_verify_targets.py      # spot-check a cached target series
│   ├── plot_results.py           # figures 1-8
│   └── plot_extended.py          # figures 9-11
│
├── julia/                        # optional high-performance mirror
│   ├── src/bppps_engine.jl       # port of propagation.py, dependency-free
│   ├── src/trainer.jl            # port of trainer.py
│   ├── scripts/check_env.jl      # validate the environment before a long run
│   └── scripts/train.jl          # config-driven training driver
│
├── results/<Lx>x<Ly>/            # model_config.json, targets, params, figures
├── documents/                    # research proposal, reference papers
└── docs/
    ├── RUNBOOK.md                # ← start here to run the simulations
    ├── benchmark_plan.md         # the three goals and what each one claims
    └── manual.md                 # pipeline reference
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/hwchoi0302/spin-glass-qml.git
cd spin-glass-qml

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the pipeline

Everything is driven by `configs/*.yaml`; individual values can be overridden
on the command line with `--set section.key=value`.

```bash
# 0. Unit tests (seconds; includes a gradient check against finite differences)
python scripts/00_validate_small.py

# 1. Build the model. This is the ONLY place the couplings J are generated;
#    every later stage, Python or Julia, loads the resulting JSON.
python scripts/00_build_model.py

# 2. Targets -> classical baselines -> BP-PPS training -> validation
python scripts/run_pipeline.py
python scripts/run_pipeline.py --stages 3 4          # resume from training
python scripts/run_pipeline.py --set ansatz.n_layers=5

# 3. Figures
python scripts/plot_results.py
python scripts/plot_extended.py

# 4. QASM circuits for hardware + the depth-compression table
python scripts/01_build_hw_circuits.py --repeats 1 2 3 4 5

# 5. T1-T: sweep the evolution time and look for the advantage window
python scripts/02_time_sweep.py
```

**Running the simulations?** Follow [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — a
self-contained procedure with pass/fail criteria at every step, resume
instructions, and the tables to fill in when reporting results.

**Working on the project?** Open [`docs/issues/`](docs/issues/README.md). Work is
split into one file per topic — scale plan, comparison models, engine
performance, hardware, writing — so a session can pick up a single thread
without carrying the rest. `CLAUDE.md` holds the cross-cutting rules.

Optional Julia path (faster for large targets):

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/scripts/check_env.jl      # run this first
julia --project=julia julia/scripts/train.jl results/4x4
```

Stages 4 and 5 of the research plan — hardware deployment and advantage
verification — are not implemented yet.

---

## 📄 Documentation

For full mathematical derivations, references, and research phases, see [`documents/spin-glass-qml-report.md`](documents/spin-glass-qml-report.md) and [`docs/manual.md`](docs/manual.md).
