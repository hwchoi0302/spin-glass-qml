"""BP-PPS Trainer: Training loop for variational circuit optimization.

Supports two training modes:
1. Time-Evolution Compression: minimize L_{X,Z} = Σ_G Σ_P (a_P - ã_P)²
2. Ground State Preparation: minimize L_E = ⟨0|U†HU|0⟩

Uses Adam optimizer with optional L-BFGS-B refinement.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from .pauli_utils import make_observable_label, is_iz_only
from .propagation import (
    SPO,
    build_hva_gate_sequence,
    propagate_forward,
    propagate_backward,
)
from .ose_regularizer import compute_ose, ose_gradient_seed


class BPPPSTrainer:
    """BP-PPS training loop.

    Attributes:
        num_qubits: Number of qubits.
        bonds: List of (i, j) bond tuples.
        substep_bonds: Brickwork bond classification.
        n_layers: Number of HVA layers.
        n_params: Total number of trainable parameters.
        target_spos: Target SPOs from TargetGenerator (time-evolution mode).
        hamiltonian_spo: Hamiltonian SPO (ground state mode).
        delta: Truncation threshold for Pauli propagation.
        lambda_ose: OSE regularization strength.
        mode: 'time_evolution' or 'ground_state'.
    """

    def __init__(
        self,
        num_qubits: int,
        bonds: List[Tuple[int, int]],
        substep_bonds: dict,
        n_layers: int,
        delta: float = 1e-4,
        lambda_ose: float = 0.0,
        mode: str = 'time_evolution',
        target_spos: Optional[Dict[str, SPO]] = None,
        hamiltonian_spo: Optional[SPO] = None,
    ):
        self.num_qubits = num_qubits
        self.bonds = bonds
        self.substep_bonds = substep_bonds
        self.n_layers = n_layers
        self.n_params = n_layers * (num_qubits + len(bonds))
        self.delta = delta
        self.lambda_ose = lambda_ose
        self.mode = mode
        self.target_spos = target_spos or {}
        self.hamiltonian_spo = hamiltonian_spo or {}

    def _build_gate_sequence(self, params: np.ndarray) -> list:
        """Build HVA gate sequence from current parameters."""
        return build_hva_gate_sequence(
            self.num_qubits, self.bonds, self.substep_bonds,
            self.n_layers, params
        )

    # ------------------------------------------------------------------
    # Time-Evolution Compression
    # ------------------------------------------------------------------

    def _time_evolution_step(self, params: np.ndarray
                             ) -> Tuple[float, np.ndarray]:
        """One training step for time-evolution compression.

        For each observable G ∈ {X_i, Z_i}:
        1. Forward: propagate G through HVA → evolved SPO
        2. Loss: L_G = Σ_P (a_P - ã_P)²
        3. Seed: ∂L_G/∂a_P = 2(a_P - ã_P) + OSE gradient
        4. Backward: propagate seed → parameter gradients

        Returns:
            (total_loss, gradient_array)
        """
        gate_seq = self._build_gate_sequence(params)

        total_loss = 0.0
        total_grad = np.zeros(self.n_params)

        for obs_key, target_spo in self.target_spos.items():
            # Parse observable key (e.g., 'X_3' → pauli='X', qubit=3)
            pauli, q_str = obs_key.split('_')
            q = int(q_str)
            init_label = make_observable_label(self.num_qubits, pauli, q)
            init_spo = {init_label: 1.0}

            # Forward pass
            evolved = propagate_forward(init_spo, gate_seq, self.delta)

            # Compute loss: Σ_P (a_P - ã_P)²
            all_paulis = set(evolved.keys()) | set(target_spo.keys())
            loss_g = 0.0
            seed = {}

            for P in all_paulis:
                a_P = evolved.get(P, 0.0)
                a_target = target_spo.get(P, 0.0)
                diff = a_P - a_target
                loss_g += diff ** 2
                if abs(diff) > 1e-15:
                    seed[P] = 2.0 * diff

            total_loss += loss_g

            # Add OSE gradient to seed
            if self.lambda_ose > 0:
                ose_val = compute_ose(evolved)
                total_loss += self.lambda_ose * ose_val
                ose_seed = ose_gradient_seed(evolved, self.lambda_ose)
                for P, g in ose_seed.items():
                    seed[P] = seed.get(P, 0.0) + g

            # Backward pass
            grad_g = propagate_backward(
                evolved, seed, gate_seq, self.n_params, self.delta
            )
            total_grad += grad_g

        return total_loss, total_grad

    # ------------------------------------------------------------------
    # Ground State Preparation
    # ------------------------------------------------------------------

    def _ground_state_step(self, params: np.ndarray
                           ) -> Tuple[float, np.ndarray]:
        """One training step for ground state preparation.

        Propagate H through HVA, then L = Σ_{P∈P_{I,Z}} a_P^[T].
        Seed: ∂L/∂a_P = 1 if P ∈ P_{I,Z}, else 0.

        Returns:
            (energy, gradient_array)
        """
        gate_seq = self._build_gate_sequence(params)

        # Forward: propagate H through circuit
        evolved = propagate_forward(
            self.hamiltonian_spo, gate_seq, self.delta
        )

        # Loss = energy = Σ_{P∈P_{I,Z}} a_P
        energy = sum(
            a_P for P, a_P in evolved.items() if is_iz_only(P)
        )

        # Seed: ∂L/∂a_P = 1 for I,Z-only strings, 0 otherwise
        seed = {P: 1.0 for P in evolved if is_iz_only(P)}

        # Backward
        grad = propagate_backward(
            evolved, seed, gate_seq, self.n_params, self.delta
        )

        return energy, grad

    # ------------------------------------------------------------------
    # Unified training step
    # ------------------------------------------------------------------

    def train_step(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Execute one training step.

        Returns:
            (loss, gradients) tuple.
        """
        if self.mode == 'time_evolution':
            return self._time_evolution_step(params)
        elif self.mode == 'ground_state':
            return self._ground_state_step(params)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Adam optimizer
    # ------------------------------------------------------------------

    def train(self, n_epochs: int = 100, lr: float = 0.01,
              params_init: Optional[np.ndarray] = None,
              verbose: bool = True,
              callback=None) -> Tuple[np.ndarray, List[float]]:
        """Run full training with Adam optimizer.

        Args:
            n_epochs: Number of training epochs.
            lr: Learning rate.
            params_init: Initial parameters. Random if None.
            verbose: Print progress.
            callback: Optional function(epoch, loss, params) called each epoch.

        Returns:
            (optimized_params, loss_history)
        """
        if params_init is None:
            rng = np.random.default_rng(42)
            params = rng.uniform(-0.1, 0.1, self.n_params)
        else:
            params = np.copy(params_init)

        # Adam state
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        losses = []
        t_start = time.time()

        for epoch in range(1, n_epochs + 1):
            loss, grad = self.train_step(params)
            losses.append(loss)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** epoch)
            v_hat = v / (1 - beta2 ** epoch)
            params -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if callback:
                callback(epoch, loss, params)

            if verbose and (epoch % max(1, n_epochs // 10) == 0
                            or epoch == 1):
                elapsed = time.time() - t_start
                print(f"  Epoch {epoch:4d}/{n_epochs}: "
                      f"loss={loss:.6f}, "
                      f"|grad|={np.linalg.norm(grad):.4f}, "
                      f"time={elapsed:.1f}s")

        return params, losses
