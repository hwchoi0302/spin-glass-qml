"""Load trained HVA parameters from Julia and build Qiskit circuits for
quantum hardware execution (IBM Nighthawk).

Usage:
    python scripts/01_build_hw_circuits.py
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
from pathlib import Path

from hamiltonians import SpinGlass2D
from ansatz import HVA, TrotterCircuit

RESULTS_DIR = Path("results/4x4")


def load_trained_params(filepath: str) -> dict:
    """Load trained parameters from Julia output JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def build_hva_circuit_from_params(model: SpinGlass2D, params: np.ndarray,
                                   n_layers: int):
    """Build HVA circuit with trained parameters."""
    hva = HVA(
        num_qubits=model.num_qubits,
        bonds=model.bonds,
        n_layers=n_layers,
        Lx=model.Lx, Ly=model.Ly,
    )
    qc = hva.build_circuit(params)
    qc.measure_all()
    return qc


def build_trotter_circuit(model: SpinGlass2D, t: float, dt: float,
                           order: int = 2):
    """Build Trotter circuit for comparison sampling."""
    pauli_op = model.get_pauli_terms()
    trotter = TrotterCircuit(
        hamiltonian_op=pauli_op,
        num_qubits=model.num_qubits,
    )
    qc = trotter.build_circuit(t, dt, order)
    qc.measure_all()
    return qc


def main():
    print("=" * 60)
    print("Building circuits for quantum hardware")
    print("=" * 60)

    # Load model config
    with open(RESULTS_DIR / "model_config.json") as f:
        config = json.load(f)

    Lx = config["Lx"]
    Ly = config["Ly"]
    h = config["h"]
    seed = config["seed"]

    model = SpinGlass2D(Lx=Lx, Ly=Ly, h=h,
                         coupling_type='ea_bimodal', seed=seed)

    # --- HVA circuit with trained parameters ---
    print("\n[1] Building HVA circuit")
    data = load_trained_params(RESULTS_DIR / "trained_params.json")
    params = np.array(data["optimized_params"])
    n_layers = data["n_layers"]

    qc_hva = build_hva_circuit_from_params(model, params, n_layers)
    print(f"  Qubits: {qc_hva.num_qubits}")
    print(f"  Depth: {qc_hva.depth()}")
    print(f"  Gates: {qc_hva.count_ops()}")

    # --- Trotter circuit for comparison ---
    delta_t = config["delta_t"]
    for dt in [0.1, 0.2]:
        print(f"\n[2] Building Trotter circuit (T={delta_t}, dt={dt})")
        qc_trotter = build_trotter_circuit(model, delta_t, dt, order=2)
        print(f"  Depth: {qc_trotter.depth()}")
        print(f"  Gates: {qc_trotter.count_ops()}")

    # --- Save circuits as QASM ---
    qc_hva.qasm(filename=str(RESULTS_DIR / "hva_circuit.qasm"))
    print(f"\nCircuits saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
