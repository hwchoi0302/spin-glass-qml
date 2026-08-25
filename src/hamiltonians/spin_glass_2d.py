import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
from qiskit.quantum_info import SparsePauliOp

from .frustration import generate_villain, generate_ea_bimodal, generate_gaussian

class SpinGlass2D:
    """
    A 2D square lattice spin glass model Hamiltonian.
    
    H = - sum J_ij Z_i Z_j - h sum X_i
    """
    def __init__(self, Lx: int, Ly: int, h: float = 1.0, coupling_type: str = 'ea_bimodal', j_magnitude: float = 1.0, seed: int = 42):
        """
        Initialize the SpinGlass2D Hamiltonian.
        
        Args:
            Lx: Number of sites in the x-direction.
            Ly: Number of sites in the y-direction.
            h: Transverse field strength.
            coupling_type: 'villain', 'ea_bimodal', or 'gaussian'.
            j_magnitude: Magnitude of the couplings J_ij.
            seed: Random seed for coupling generation.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.num_qubits = Lx * Ly
        self.h = h
        self.bonds = self._build_bonds()
        
        if coupling_type == 'villain':
            self.J = generate_villain(Lx, Ly, j_magnitude)
        elif coupling_type == 'ea_bimodal':
            self.J = generate_ea_bimodal(Lx, Ly, j_magnitude, seed)
        elif coupling_type == 'gaussian':
            self.J = generate_gaussian(Lx, Ly, j_magnitude, seed)
        else:
            raise ValueError(f"Unknown coupling_type: {coupling_type}")
            
    def _qubit_index(self, x: int, y: int) -> int:
        """Map 2D coordinate to linear qubit index."""
        return y * self.Lx + x
        
    def _build_bonds(self) -> List[Tuple[int, int]]:
        """
        Build a list of all nearest-neighbor bonds on a 2D square lattice with open boundaries.
        Horizontal bonds are added first, then vertical bonds.
        
        Returns:
            List of (i, j) tuples representing the interacting qubits.
        """
        bonds = []
        # Horizontal bonds
        for y in range(self.Ly):
            for x in range(self.Lx - 1):
                bonds.append((self._qubit_index(x, y), self._qubit_index(x + 1, y)))
        # Vertical bonds
        for x in range(self.Lx):
            for y in range(self.Ly - 1):
                bonds.append((self._qubit_index(x, y), self._qubit_index(x, y + 1)))
        return bonds
        
    def get_bonds(self) -> List[Tuple[int, int]]:
        """Get the list of bonds."""
        return self.bonds
        
    def get_couplings(self) -> np.ndarray:
        """Get the array of coupling values."""
        return self.J
        
    @property
    def num_bonds(self) -> int:
        """Number of bonds in the lattice."""
        return len(self.bonds)
        
    def build_sparse_matrix(self) -> sp.csr_matrix:
        """
        Build the full 2^N x 2^N sparse Hamiltonian matrix.
        
        Uses vectorized numpy operations for efficient construction.
        Only safe for num_qubits <= 22. For larger systems, use get_pauli_terms()
        with SPD (pauli-prop) instead.
        
        Returns:
            scipy.sparse.csr_matrix representing the Hamiltonian.
            
        Raises:
            ValueError: If num_qubits > 22 (OOM risk).
        """
        N = self.num_qubits
        if N > 22:
            raise ValueError(
                f"build_sparse_matrix()는 22큐비트까지만 안전합니다 "
                f"(요청: {N}). get_pauli_terms()와 SPD를 사용하세요."
            )
        dim = 1 << N
        states = np.arange(dim, dtype=np.int64)
        
        # Diagonal terms: - sum J_ij Z_i Z_j (vectorized)
        diag = np.zeros(dim, dtype=np.float64)
        for idx, (i, j) in enumerate(self.bonds):
            bit_i = (states >> i) & 1
            bit_j = (states >> j) & 1
            zz = 1 - 2 * ((bit_i + bit_j) & 1)  # (-1)^(bit_i + bit_j)
            diag -= self.J[idx] * zz
        
        H = sp.diags(diag, format='csr')
        
        # Off-diagonal terms: -h sum X_i (vectorized)
        if self.h != 0.0:
            for i in range(N):
                flipped = states ^ (1 << i)
                X_i = sp.csr_matrix(
                    (-self.h * np.ones(dim), (states, flipped)),
                    shape=(dim, dim),
                )
                H = H + X_i
        
        return H
        
    def get_pauli_terms(self) -> SparsePauliOp:
        """
        Get the Qiskit SparsePauliOp representation of the Hamiltonian.
        Qiskit uses little-endian qubit ordering, meaning qubit 0 is the rightmost character.
        
        Returns:
            SparsePauliOp: The Hamiltonian in Pauli basis.
        """
        N = self.num_qubits
        pauli_strings = []
        coeffs = []
        
        # -J_ij Z_i Z_j
        for idx, (i, j) in enumerate(self.bonds):
            J_ij = self.J[idx]
            if J_ij != 0.0:
                chars = ['I'] * N
                chars[N - 1 - i] = 'Z'
                chars[N - 1 - j] = 'Z'
                pauli_strings.append("".join(chars))
                coeffs.append(-J_ij)
                
        # -h X_i
        if self.h != 0.0:
            for i in range(N):
                chars = ['I'] * N
                chars[N - 1 - i] = 'X'
                pauli_strings.append("".join(chars))
                coeffs.append(-self.h)
                
        if not pauli_strings:
            return SparsePauliOp.from_list([("I" * N, 0.0)])
            
        return SparsePauliOp.from_list(list(zip(pauli_strings, coeffs)))
