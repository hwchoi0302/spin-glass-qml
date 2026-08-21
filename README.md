# 🌀 spin-glass-qml

> **Simulating 2D Spin Glass Dynamics via Backpropagating Pauli Circuit Compression & Quantum Advantage Verification**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.1%2B-6133ff.svg)](https://qiskit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository provides an end-to-end framework to simulate **2D Spin Glass model dynamics and ground state preparation** by compressing deep Trotter circuits into shallow Hardware-Efficient Ansatze (HEA) on 2D square lattice processors (e.g., IBM Nighthawk).

### Key Methodologies
1. **Train on Classical, Deploy on Quantum (TCDQ):** Perform parameter optimization classically, and leverage quantum processors exclusively for global entanglement sampling.
2. **Backpropagating Pauli Propagation (BP-PPS):** Memory-efficient exact analytic gradients without storing state vectors, completely bypassing Barren Plateaus via local operator error tracking ($\mathcal{L}_{X,Z}$).
3. **Generative Quantum Advantage Verification:** Cross-validating quantum bitstrings against Classical Shadows and demonstrating OTOC decay beyond classical tensor network (PEPS) breakdown thresholds.

---

## 🗂 Project Structure

```text
spin-glass-qml/
├── configs/                      # Experiment configuration YAMLs
│   ├── model_2d_spinglass.yaml   # 2D lattice size (Lx, Ly), couplings, transverse field
│   ├── ansatz_ibm.yaml           # HEA depth, layout, 2D topology
│   └── optimizer.yaml            # Optimization hyperparameters (Adam, L-BFGS-B, truncation threshold)
│
├── src/                          # Core Python modules
│   ├── hamiltonians/             # 2D Spin Glass Hamiltonian with random frustration
│   ├── ansatz/                   # SWAP-free HEA & benchmark Trotter circuits
│   ├── bppps/                    # Pauli string manipulation & backpropagation engine
│   ├── classical_bench/          # Exact diagonalization & Tensor Network baselines
│   ├── hardware/                 # Qiskit IBM Runtime deployment & OTOC measurements
│   └── validation/               # Classical Shadows & cross-validation metrics
│
├── scripts/                      # Pipeline execution scripts
│   ├── 01_train_hea.py           # Phase 2: Classical BP-PPS optimization
│   ├── 02_benchmark_classical.py # Phase 3: Classical benchmark comparison
│   ├── 03_deploy_quantum.py      # Phase 4: IBM Quantum hardware execution
│   └── 04_verify_advantage.py    # Phase 5: Sampling verification & advantage declaration
│
├── data/                         # Experiment datasets & checkpoints (gitignored)
│   ├── raw/
│   └── processed/
│
├── documents/                    # Research proposal & reports
│   ├── report.md
│   ├── report.tex
│   └── report.pdf
│
└── tests/                        # Unit tests
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

### 2. Running Pipeline

```bash
# 1. Classical Parameter Optimization (BP-PPS)
python scripts/01_train_hea.py --config configs/model_2d_spinglass.yaml

# 2. Classical Benchmark Comparison
python scripts/02_benchmark_classical.py

# 3. Deploy on IBM Quantum Processor (requires IBM Quantum API token)
python scripts/03_deploy_quantum.py --backend ibm_brisbane

# 4. Verify Quantum Advantage & Plot OTOC
python scripts/04_verify_advantage.py
```

---

## 📄 Documentation

For full mathematical derivations, references, and research phases, see [`documents/report.pdf`](documents/report.pdf) or [`documents/report.md`](documents/report.md).
