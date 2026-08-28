import numpy as np
from typing import List, Tuple, Optional, Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from hamiltonians.spin_glass_2d import classify_substep_bonds

class HVA:
    """Hamiltonian Variational Ansatz (HVA) for 2D spin glass models.
    
    Attributes:
        num_qubits (int): Number of qubits.
        bonds (List[Tuple[int, int]]): List of bonds (i, j) for 2-qubit interactions.
        n_layers (int): Number of HVA layers.
        Lx (int): Lattice dimension in x (horizontal).
        Ly (int): Lattice dimension in y (vertical).
        J (Optional[Union[List[float], np.ndarray]]): Coupling constants.
        parameter_prefix (str): Prefix for Qiskit parameters.
    """

    def __init__(
        self,
        num_qubits: int,
        bonds: List[Tuple[int, int]],
        n_layers: int,
        Lx: int,
        Ly: int,
        J: Optional[Union[List[float], np.ndarray]] = None,
        parameter_prefix: str = "hva",
    ) -> None:
        """Initialize the HVA circuit builder."""
        self.num_qubits = num_qubits
        self.bonds = bonds
        self.n_layers = n_layers
        self.Lx = Lx
        self.Ly = Ly
        self.J = J
        self.parameter_prefix = parameter_prefix
        
        self._classify_bonds()

    def _classify_bonds(self) -> None:
        """Colour the bonds into 4 conflict-free substeps.

        Delegates to hamiltonians.classify_substep_bonds so that the circuit
        builder, the Pauli-propagation engine and the Julia pipeline all use
        one definition of the square-lattice colouring.
        """
        self.substep_bonds = classify_substep_bonds(self.bonds, self.Lx)

    def count_params(self) -> int:
        """Return total number of parameters."""
        return self.n_layers * (self.num_qubits + len(self.bonds))

    def circuit_depth(self) -> int:
        """Return the circuit depth in terms of layers."""
        return 5 * self.n_layers

    def count_2q_gates(self) -> int:
        """Return the total number of 2-qubit gates."""
        return self.n_layers * len(self.bonds)

    def build_circuit(self, params: Optional[Union[np.ndarray, List[float], ParameterVector]] = None) -> QuantumCircuit:
        """Build the HVA circuit.
        
        Args:
            params: Optional parameters to bind. If None, uses a Qiskit ParameterVector.
            
        Returns:
            QuantumCircuit: The constructed HVA circuit.
        """
        qc = QuantumCircuit(self.num_qubits)
        
        if params is None:
            params = ParameterVector(self.parameter_prefix, self.count_params())
            
        params_per_layer = self.num_qubits + len(self.bonds)
        
        for layer in range(self.n_layers):
            layer_start = layer * params_per_layer
            
            # 1. RX on all qubits
            for i in range(self.num_qubits):
                qc.rx(params[layer_start + i], i)
                
            # 2. RZZ in 4 parallel substeps (square lattice coloring)
            for step in range(1, 5):
                for bond_idx, i, j in self.substep_bonds[step]:
                    param_idx = layer_start + self.num_qubits + bond_idx
                    qc.rzz(params[param_idx], i, j)
                    
        return qc
