import numpy as np
from typing import List, Tuple, Optional, Union
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

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
        """Classify bonds into 4 substeps for brickwork pattern."""
        self.substep_bonds = {1: [], 2: [], 3: [], 4: []}
        
        for idx, (i, j) in enumerate(self.bonds):
            xi, yi = i % self.Lx, i // self.Lx
            xj, yj = j % self.Lx, j // self.Lx
            
            if yi == yj:  # Horizontal
                if min(xi, xj) % 2 == 0:
                    self.substep_bonds[1].append((idx, i, j))
                else:
                    self.substep_bonds[2].append((idx, i, j))
            elif xi == xj:  # Vertical
                if min(yi, yj) % 2 == 0:
                    self.substep_bonds[3].append((idx, i, j))
                else:
                    self.substep_bonds[4].append((idx, i, j))
            else:
                # Fallback if not nearest neighbor on grid
                self.substep_bonds[1].append((idx, i, j))

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
                
            # 2. RZZ in brickwork pattern (4 substeps)
            for step in range(1, 5):
                for bond_idx, i, j in self.substep_bonds[step]:
                    param_idx = layer_start + self.num_qubits + bond_idx
                    qc.rzz(params[param_idx], i, j)
                    
        return qc
