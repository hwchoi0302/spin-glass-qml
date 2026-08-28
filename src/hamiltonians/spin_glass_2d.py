import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import List, Tuple
from qiskit.quantum_info import SparsePauliOp

from .frustration import generate_villain, generate_ea_bimodal, generate_gaussian


def classify_substep_bonds(bonds: List[Tuple[int, int]], Lx: int) -> dict:
    """Colour the bonds of a square lattice into 4 conflict-free substeps.

    Two gates in the same substep never share a qubit, so each substep is one
    layer of parallel 2-qubit gates:

        1: horizontal bonds starting at even x
        2: horizontal bonds starting at odd x
        3: vertical bonds starting at even y
        4: vertical bonds starting at odd y

    This is the single definition used by the Pauli-propagation engine, the
    Qiskit circuit builders and (via model_config.json) the Julia pipeline.

    Args:
        bonds: List of (i, j) bond tuples, in canonical order.
        Lx: Lattice width, used to map a linear qubit index back to (x, y).

    Returns:
        Dict {1..4: [(bond_index, i, j), ...]}.
    """
    substeps = {1: [], 2: [], 3: [], 4: []}
    for idx, (i, j) in enumerate(bonds):
        xi, yi = i % Lx, i // Lx
        xj, yj = j % Lx, j // Lx
        if yi == yj:                                   # horizontal
            substeps[1 if min(xi, xj) % 2 == 0 else 2].append((idx, i, j))
        elif xi == xj:                                 # vertical
            substeps[3 if min(yi, yj) % 2 == 0 else 4].append((idx, i, j))
        else:                                          # not a grid neighbour
            substeps[1].append((idx, i, j))
    return substeps


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

        self.coupling_type = coupling_type
        self.j_magnitude = j_magnitude
        self.seed = seed
        self.substep_bonds = classify_substep_bonds(self.bonds, self.Lx)

    # ------------------------------------------------------------------
    # Serialisation — model_config.json is the contract between the Python
    # and the Julia pipeline. Both sides must read the same J array in the
    # same bond order; neither may regenerate the couplings on its own.
    # ------------------------------------------------------------------

    def to_config_dict(self) -> dict:
        """Serialise the full model definition, including the bond ordering."""
        return {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'h': self.h,
            'seed': self.seed,
            'coupling_type': self.coupling_type,
            'j_magnitude': self.j_magnitude,
            'num_qubits': self.num_qubits,
            'num_bonds': self.num_bonds,
            # bonds[k] and J[k] refer to the same physical bond, by index.
            'bonds': [list(b) for b in self.bonds],
            'J': [float(j) for j in self.J],
            'substep_bonds': {
                str(s): [[int(idx), int(i), int(j)] for idx, i, j in v]
                for s, v in self.substep_bonds.items()
            },
        }

    @classmethod
    def from_config_dict(cls, config: dict) -> "SpinGlass2D":
        """Rebuild a model from ``to_config_dict`` output.

        The stored ``bonds``/``J`` arrays win over anything the constructor
        would regenerate, so a run always uses the couplings that were
        actually written to disk.
        """
        model = cls(
            Lx=config['Lx'], Ly=config['Ly'], h=config['h'],
            coupling_type=config.get('coupling_type', 'ea_bimodal'),
            j_magnitude=config.get('j_magnitude', 1.0),
            seed=config['seed'],
        )
        stored_bonds = [tuple(b) for b in config['bonds']]
        if stored_bonds != model.bonds:
            raise ValueError(
                "Bond ordering in model_config.json does not match "
                "SpinGlass2D._build_bonds(). Regenerate the config with "
                "scripts/00_build_model.py."
            )
        model.J = np.asarray(config['J'], dtype=float)
        return model

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
        
    def diagonal(self) -> np.ndarray:
        """Diagonal of H in the computational basis: -sum_ij J_ij z_i z_j.

        This is the whole coupling contribution: J only ever enters the
        diagonal, which is why the model is stoquastic for every choice of
        J (see build_linear_operator).
        """
        N = self.num_qubits
        states = np.arange(1 << N, dtype=np.int64)
        diag = np.zeros(1 << N, dtype=np.float64)
        for idx, (i, j) in enumerate(self.bonds):
            bit_i = (states >> i) & 1
            bit_j = (states >> j) & 1
            diag -= self.J[idx] * (1 - 2 * ((bit_i + bit_j) & 1))
        return diag

    def build_linear_operator(self) -> spla.LinearOperator:
        """Matrix-free Hamiltonian, for exact diagonalisation above ~22 qubits.

        Storing H as a sparse matrix costs N * 2^N nonzeros for the transverse
        field alone (26 qubits: ~1.7 billion entries, tens of GB). Applying it
        instead costs one precomputed diagonal, 2^N floats, plus N strided
        gathers per matrix-vector product and no matrix storage at all.

        Note the structure this exposes: the only off-diagonal contribution is
        -h from each X_i, so every off-diagonal element of H is <= 0 for h > 0
        regardless of the couplings. The Hamiltonian is therefore *stoquastic*
        for any J -- frustration lives entirely in the diagonal and never
        produces negative quantum Monte Carlo weights. Frustration and the
        sign problem are separate things for this model.

        Returns:
            A Hermitian scipy LinearOperator of shape (2^N, 2^N).
        """
        N = self.num_qubits
        dim = 1 << N
        diag = self.diagonal()
        h = self.h

        def matvec(psi: np.ndarray) -> np.ndarray:
            psi = np.asarray(psi).ravel()
            out = diag * psi
            if h != 0.0:
                # X_i flips bit i, i.e. swaps the two halves along that axis.
                # Reshaping to (high, bit_i, low) and reversing the middle axis
                # does that as a strided view, with no index array.
                for i in range(N):
                    view = psi.reshape(-1, 2, 1 << i)
                    out -= h * view[:, ::-1, :].reshape(-1)
            return out

        return spla.LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec,
                                   dtype=np.float64)

    def build_sparse_matrix(self) -> sp.csr_matrix:
        """
        Build the full 2^N x 2^N sparse Hamiltonian matrix.

        Uses vectorized numpy operations for efficient construction.
        Only safe for num_qubits <= 22; the transverse field alone contributes
        N * 2^N nonzeros. Above that use build_linear_operator() (matrix-free)
        or get_pauli_terms() with SPD.

        Returns:
            scipy.sparse.csr_matrix representing the Hamiltonian.

        Raises:
            ValueError: If num_qubits > 22 (OOM risk).
        """
        N = self.num_qubits
        if N > 22:
            raise ValueError(
                f"build_sparse_matrix()는 22큐비트까지만 안전합니다 "
                f"(요청: {N}). build_linear_operator() 를 쓰거나, "
                f"더 큰 계에서는 get_pauli_terms()와 SPD를 사용하세요."
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
