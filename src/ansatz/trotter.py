import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit

class TrotterCircuit:
    """Trotter circuit for time evolution of 2D spin glass models.
    
    Attributes:
        num_qubits (int): Number of qubits.
        bonds (List[Tuple[int, int]]): List of bonds (i, j).
        J (Union[List[float], np.ndarray]): Coupling constants matching bonds order.
        h (float): Transverse field strength.
        Lx (int): Lattice dimension in x.
        Ly (int): Lattice dimension in y.
    """

    def __init__(
        self,
        num_qubits: int,
        bonds: List[Tuple[int, int]],
        J: Union[List[float], np.ndarray],
        h: float,
        Lx: int,
        Ly: int,
    ) -> None:
        """Initialize the Trotter circuit builder."""
        self.num_qubits = num_qubits
        self.bonds = bonds
        self.J = J
        self.h = h
        self.Lx = Lx
        self.Ly = Ly
        
        self._classify_bonds()

    def _classify_bonds(self) -> None:
        """Classify bonds into 4 substeps for brickwork pattern."""
        self.substep_bonds = {1: [], 2: [], 3: [], 4: []}
        
        for idx, (i, j) in enumerate(self.bonds):
            xi, yi = i % self.Lx, i // self.Lx
            xj, yj = j % self.Lx, j // self.Lx
            
            if yi == yj:
                if min(xi, xj) % 2 == 0:
                    self.substep_bonds[1].append((idx, i, j))
                else:
                    self.substep_bonds[2].append((idx, i, j))
            elif xi == xj:
                if min(yi, yj) % 2 == 0:
                    self.substep_bonds[3].append((idx, i, j))
                else:
                    self.substep_bonds[4].append((idx, i, j))
            else:
                self.substep_bonds[1].append((idx, i, j))

    def num_steps(self, t: float, dt: float) -> int:
        """Calculate the number of Trotter steps."""
        return int(round(t / dt))

    def circuit_depth(self, t: float, dt: float, order: int = 2) -> int:
        """Calculate the circuit depth.
        
        Args:
            t: Total time.
            dt: Time step size.
            order: Trotter formula order (1 or 2).
            
        Returns:
            int: Estimated circuit depth.
        """
        steps = self.num_steps(t, dt)
        if order == 1:
            return 5 * steps
        elif order == 2:
            return 6 * steps
        else:
            raise ValueError("Only order 1 and 2 are supported.")

    def count_2q_gates(self, t: float, dt: float) -> int:
        """Calculate the total number of 2-qubit gates."""
        steps = self.num_steps(t, dt)
        return steps * len(self.bonds)

    def build_circuit(self, t: float, dt: float, order: int = 2) -> QuantumCircuit:
        """Build the Trotter circuit for time evolution exp(-iHt).
        
        Args:
            t: Total time.
            dt: Time step size.
            order: Trotter formula order (1 or 2).
            
        Returns:
            QuantumCircuit: The constructed Trotter circuit.
        """
        qc = QuantumCircuit(self.num_qubits)
        steps = self.num_steps(t, dt)
        
        if steps == 0:
            return qc
            
        if order not in (1, 2):
            raise ValueError("Only order 1 and 2 are supported.")
            
        for _ in range(steps):
            if order == 1:
                # 1st order: exp(-iH_ZZ dt) exp(-iH_X dt)
                # exp(-iH_ZZ dt) -> RZZ(-2 * J_ij * dt)
                for step_idx in range(1, 5):
                    for bond_idx, i, j in self.substep_bonds[step_idx]:
                        qc.rzz(-2.0 * self.J[bond_idx] * dt, i, j)
                
                # exp(-iH_X dt) -> RX(-2 * h * dt)
                for i in range(self.num_qubits):
                    qc.rx(-2.0 * self.h * dt, i)
                    
            elif order == 2:
                # 2nd order: exp(-iH_X dt/2) exp(-iH_ZZ dt) exp(-iH_X dt/2)
                # exp(-iH_X dt/2) -> RX(-h * dt)
                for i in range(self.num_qubits):
                    qc.rx(-1.0 * self.h * dt, i)
                    
                # exp(-iH_ZZ dt) -> RZZ(-2 * J_ij * dt)
                for step_idx in range(1, 5):
                    for bond_idx, i, j in self.substep_bonds[step_idx]:
                        qc.rzz(-2.0 * self.J[bond_idx] * dt, i, j)
                        
                # exp(-iH_X dt/2) -> RX(-h * dt)
                for i in range(self.num_qubits):
                    qc.rx(-1.0 * self.h * dt, i)
                    
        return qc
