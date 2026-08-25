import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, List, Dict, Union
import warnings

# ED는 상태벡터 2^N을 다루므로 큐비트 수 제한 필요
MAX_QUBITS_ED = 22  # 2^22 = 4M 상태, ~64MB state vector

class ExactDiag:
    """
    Exact Diagonalization and time evolution for the 2D spin glass model.
    
    WARNING: ED는 2^N 차원의 상태벡터를 사용하므로, num_qubits ≤ 22 에서만
    안전하게 동작합니다. 대규모 시스템의 관측량 계산에는 SPD (Pauli Propagation)를
    사용해야 합니다.
    """
    
    def __init__(self, hamiltonian_sparse: sp.csr_matrix, num_qubits: int):
        """
        Initialize the ExactDiag module.
        
        Args:
            hamiltonian_sparse: Sparse matrix representation of the Hamiltonian.
            num_qubits: Number of qubits in the system.
            
        Raises:
            ValueError: If num_qubits exceeds MAX_QUBITS_ED.
        """
        if num_qubits > MAX_QUBITS_ED:
            raise ValueError(
                f"ExactDiag은 {MAX_QUBITS_ED}큐비트까지만 지원합니다 "
                f"(요청: {num_qubits}). "
                f"대규모 시스템에는 SPD (pauli-prop)를 사용하세요."
            )
        if num_qubits > 18:
            warnings.warn(
                f"num_qubits={num_qubits}: 상태벡터 크기 2^{num_qubits} = "
                f"{2**num_qubits:,}. 메모리 사용량이 클 수 있습니다.",
                ResourceWarning,
            )
        self.H = hamiltonian_sparse
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

    def ground_state(self, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the k lowest energy states and their eigenvalues.
        
        Args:
            k: Number of lowest states to compute.
            
        Returns:
            Tuple of (energies, states) where energies is an array of k lowest 
            eigenvalues and states is an array of shape (2^N, k) containing eigenvectors.
        """
        if self.dim <= k + 1:
            # Fallback to dense exact diagonalization for very small systems
            # where sparse iterative solvers might fail due to small dimension
            eigvals, eigvecs = np.linalg.eigh(self.H.toarray())
            return eigvals[:k], eigvecs[:, :k]
        else:
            energies, states = spla.eigsh(self.H, k=k, which='SA')
            return energies, states

    def ground_energy(self) -> float:
        """
        Compute the ground state energy.
        
        Returns:
            The ground state energy.
        """
        energies, _ = self.ground_state(k=1)
        return float(energies[0])

    def time_evolve(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """
        Perform exact time evolution of an initial state under the Hamiltonian.
        
        Args:
            psi0: Initial state vector of shape (2^N,).
            t: Real time t to evolve for.
            
        Returns:
            The time-evolved state vector.
        """
        # A = -i * H * t
        A = -1j * t * self.H
        return spla.expm_multiply(A, psi0)

    def bitstring_distribution(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution over computational basis states.
        
        Args:
            psi: State vector of shape (2^N,).
            
        Returns:
            Array of shape (2^N,) containing probabilities P(x) = |<x|psi>|^2.
        """
        return np.abs(psi) ** 2

    def local_observables(self, psi: np.ndarray, bonds: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """
        Compute local expectation values for X, Z, and ZZ.
        
        Args:
            psi: State vector of shape (2^N,).
            bonds: List of qubit index pairs (i, j) defining the lattice bonds.
            
        Returns:
            Dictionary containing expectation values for 'X', 'Z', and 'ZZ'.
        """
        probs = self.bitstring_distribution(psi)
        
        X_vals = np.zeros(self.num_qubits)
        Z_vals = np.zeros(self.num_qubits)
        ZZ_vals = np.zeros(len(bonds))
        
        indices = np.arange(self.dim)
        
        for i in range(self.num_qubits):
            mask = 1 << i
            # Z_i: +1 if bit is 0, -1 if bit is 1
            signs = 1 - 2 * ((indices & mask) >> i)
            Z_vals[i] = np.sum(probs * signs)
            
            # X_i: flip bit i
            X_vals[i] = np.real(np.sum(np.conj(psi) * psi[indices ^ mask]))

        for idx, (i, j) in enumerate(bonds):
            mask_i = 1 << i
            mask_j = 1 << j
            bit_i = (indices & mask_i) >> i
            bit_j = (indices & mask_j) >> j
            # ZZ_ij: +1 if bits match, -1 if bits differ
            signs = 1 - 2 * ((bit_i + bit_j) & 1)
            ZZ_vals[idx] = np.sum(probs * signs)
            
        return {
            'X': X_vals,
            'Z': Z_vals,
            'ZZ': ZZ_vals
        }

    def compute_energy(self, psi: np.ndarray, bonds: List[Tuple[int, int]], J: Union[List[float], np.ndarray], h: float) -> float:
        """
        Compute total energy expectation value E = -sum J_ij <Z_i Z_j> - h sum <X_i>.
        
        Args:
            psi: State vector of shape (2^N,).
            bonds: List of qubit index pairs (i, j).
            J: Array or list of coupling strengths corresponding to bonds.
            h: Transverse field strength.
            
        Returns:
            Total expected energy.
        """
        obs = self.local_observables(psi, bonds)
        ZZ = obs['ZZ']
        X = obs['X']
        
        # Element-wise multiplication of J_ij and <Z_i Z_j>
        energy = -np.sum(np.array(J) * ZZ) - h * np.sum(X)
        return float(energy)

    def state_fidelity(self, psi1: np.ndarray, psi2: np.ndarray) -> float:
        """
        Compute the state fidelity F = |<psi1|psi2>|^2.
        
        Args:
            psi1: First state vector.
            psi2: Second state vector.
            
        Returns:
            Fidelity value between 0 and 1.
        """
        return float(np.abs(np.vdot(psi1, psi2)) ** 2)
