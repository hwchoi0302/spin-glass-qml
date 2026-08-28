import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, List, Dict, Optional, Union
import warnings

# ED는 상태벡터 2^N을 다루므로 큐비트 수 제한 필요.
# 26큐비트 = 2^26 = 67M 진폭 = 1.07 GB/복소벡터. 여기까지가 데스크탑 한계이고,
# 그 위로는 SPD(파울리 전파)만 남습니다.
MAX_QUBITS_ED = 26

# 이 크기를 넘어가면 H를 희소행렬로 쌓는 것 자체가 불가능하므로
# SpinGlass2D.build_linear_operator() (matrix-free) 를 써야 합니다.
MAX_QUBITS_SPARSE = 22


def statevector_gb(num_qubits: int, complex_dtype: bool = True) -> float:
    """Memory of one state vector, in GiB."""
    return (1 << num_qubits) * (16 if complex_dtype else 8) / 1024 ** 3

class ExactDiag:
    """
    Exact Diagonalization and time evolution for the 2D spin glass model.
    
    WARNING: ED는 2^N 차원의 상태벡터를 사용하므로, num_qubits ≤ 22 에서만
    안전하게 동작합니다. 대규모 시스템의 관측량 계산에는 SPD (Pauli Propagation)를
    사용해야 합니다.
    """
    
    def __init__(self, hamiltonian, num_qubits: int):
        """
        Initialize the ExactDiag module.

        Args:
            hamiltonian: The Hamiltonian, either as a scipy sparse matrix
                (fine up to ~22 qubits) or as a matrix-free LinearOperator
                from SpinGlass2D.build_linear_operator() (needed beyond that,
                since the sparse matrix itself would not fit).
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
        vec_gb = statevector_gb(num_qubits)
        if num_qubits > 22:
            warnings.warn(
                f"num_qubits={num_qubits}: 상태벡터 하나가 {vec_gb:.2f} GiB 입니다. "
                f"ground_state()는 ncv개의 Lanczos 벡터를 동시에 들고 있으므로 "
                f"약 {vec_gb * 20:.0f} GiB (기본 ncv=20)가 필요합니다. "
                f"ncv를 줄여 호출하세요.",
                ResourceWarning,
            )
        elif num_qubits > 18:
            warnings.warn(
                f"num_qubits={num_qubits}: 상태벡터 크기 2^{num_qubits} = "
                f"{2**num_qubits:,} ({vec_gb:.2f} GiB). "
                f"메모리 사용량이 클 수 있습니다.",
                ResourceWarning,
            )
        self.H = hamiltonian
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

    def ground_state(self, k: int = 1, ncv: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the k lowest energy states and their eigenvalues.

        Args:
            k: Number of lowest states to compute.
            ncv: Number of Lanczos vectors held simultaneously. scipy defaults
                to max(2k+1, 20); each vector costs one full state vector, so
                at 25-26 qubits the default alone is 10-20 GiB. Lowering it
                trades convergence speed for memory.

        Returns:
            Tuple of (energies, states) where energies is an array of k lowest
            eigenvalues and states is an array of shape (2^N, k) containing eigenvectors.
        """
        if self.dim <= k + 1:
            # Fallback to dense exact diagonalization for very small systems
            # where sparse iterative solvers might fail due to small dimension
            dense = self.H.toarray() if hasattr(self.H, 'toarray') else \
                self.H @ np.eye(self.dim)
            eigvals, eigvecs = np.linalg.eigh(dense)
            return eigvals[:k], eigvecs[:, :k]
        energies, states = spla.eigsh(self.H, k=k, which='SA', ncv=ncv)
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
        # A = -i * H * t. Tr[H] = 0 exactly (every Pauli term is traceless),
        # so pass it explicitly: otherwise scipy estimates the trace, which
        # costs extra matrix-vector products and warns.
        A = -1j * t * self.H
        return spla.expm_multiply(A, psi0, traceA=0.0)

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

            # X_i: flip bit i. Reshaping to (high, bit_i, low) and reversing
            # the middle axis is a strided view, so this avoids materialising
            # a 2^N index array (537 MB at 26 qubits).
            flipped = psi.reshape(-1, 2, mask)[:, ::-1, :].reshape(-1)
            X_vals[i] = np.real(np.sum(np.conj(psi) * flipped))

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
