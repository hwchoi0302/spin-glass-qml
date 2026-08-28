"""BP-PPS Trainer: classical optimization of a variational circuit.

Two training modes, both driven by the backward pass of
"Backpropagating Pauli Propagation" (arXiv:2607.15184), Eqs. 20-21:

1. Time-evolution compression (Eq. 30)
       L_{X,Z} = sum_G sum_P (a_P - a_P_target)^2 ,   G in {X_i, Z_i}
   The rHS norm makes Pauli strings orthonormal, so the squared operator
   distance is the squared l2 distance of the coefficient vectors.

2. Ground-state preparation (Sec. III A)
       E(theta) = <0| U(theta)^dag H U(theta) |0> = sum_{P in {I,Z}^n} a_P
   The observable propagated forward is the Hamiltonian itself, because that
   is the operator whose expectation value we want; the Heisenberg picture
   moves the circuit onto the observable rather than onto the state.

Both modes optionally add the OSE regulariser of Eq. 32.

Optimisation follows the paper's two-stage recipe: Adam for the initial
descent, then L-BFGS-B to converge, which is worthwhile precisely because
BP-PPS supplies exact analytic gradients rather than stochastic estimates.

Every propagation is instrumented with the Appendix B truncation-error
estimate, and the truncation threshold can tighten itself whenever that
estimate grows large compared to the quantity being optimised.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .ose_regularizer import compute_ose, ose_gradient_seed
from .pauli_utils import is_iz_only, make_observable_label
from .propagation import (
    SPO,
    TruncationStats,
    build_hva_gate_sequence,
    propagate_backward,
    propagate_forward,
)


class BPPPSTrainer:
    """BP-PPS training loop.

    Attributes:
        num_qubits: Number of qubits.
        bonds: List of (i, j) bond tuples.
        substep_bonds: Square-lattice bond colouring.
        n_layers: Number of HVA layers.
        n_params: Total number of trainable parameters.
        delta: Current truncation threshold (mutated when adaptive).
        last_stats: TruncationStats of the most recent train_step.
        last_error_estimate: eps_emp (Eq. B16) of the most recent train_step.
    """

    def __init__(
        self,
        num_qubits: int,
        bonds: List[Tuple[int, int]],
        substep_bonds: dict,
        n_layers: int,
        delta: float = 1e-3,
        lambda_ose: float = 0.0,
        mode: str = 'time_evolution',
        target_spos: Optional[Dict[str, SPO]] = None,
        hamiltonian_spo: Optional[SPO] = None,
        ose_alpha: float = 1.0,
        min_delta: float = 1e-5,
        adaptive_delta: bool = False,
        delta_factor: float = 0.1,
        error_ratio: float = 0.1,
        patience: int = 10,
    ):
        self.num_qubits = num_qubits
        self.bonds = bonds
        self.substep_bonds = substep_bonds
        self.n_layers = n_layers
        self.n_params = n_layers * (num_qubits + len(bonds))
        self.delta = delta
        self.lambda_ose = lambda_ose
        self.ose_alpha = ose_alpha
        self.mode = mode
        self.target_spos = target_spos or {}
        self.hamiltonian_spo = hamiltonian_spo or {}

        # Adaptive truncation schedule
        self.min_delta = min_delta
        self.adaptive_delta = adaptive_delta
        self.delta_factor = delta_factor
        self.error_ratio = error_ratio
        self.patience = patience

        # Diagnostics from the most recent step
        self.last_stats = TruncationStats()
        self.last_error_estimate = 0.0
        self.delta_history: List[Tuple[int, float]] = []

    def _build_gate_sequence(self, params: np.ndarray) -> list:
        """Build HVA gate sequence (circuit order) from current parameters."""
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

        For each observable G in {X_i, Z_i}:
        1. Forward: propagate G through the HVA -> evolved SPO
        2. Loss: L_G = sum_P (a_P - a_P_target)^2, summed over the union of
           the evolved and target supports (the target-only strings are a
           genuine part of the error and must be counted).
        3. Seed: dL_G/da_P = 2(a_P - a_P_target), restricted to the evolved
           support. BP-PPS Sec. III A: "we perform optimization by propagating
           only the derivative with non-zero coefficient support". The joint
           truncation rule in the backward pass would prune everything else at
           the first gate anyway, so building the restricted seed directly
           just avoids materialising a dict the size of the target.
        4. Backward: propagate the seed -> parameter gradients

        Returns:
            (total_loss, gradient_array)
        """
        gate_seq = self._build_gate_sequence(params)
        stats = TruncationStats()

        total_loss = 0.0
        total_grad = np.zeros(self.n_params)

        for obs_key, target_spo in self.target_spos.items():
            pauli, q_str = obs_key.split('_')
            init_label = make_observable_label(
                self.num_qubits, pauli, int(q_str)
            )

            evolved = propagate_forward(
                {init_label: 1.0}, gate_seq, self.delta, stats
            )

            # Loss over the full union; seed only where the SPO has support.
            loss_g = 0.0
            seed = {}
            for P, a_P in evolved.items():
                diff = a_P - target_spo.get(P, 0.0)
                loss_g += diff * diff
                if diff != 0.0:
                    seed[P] = 2.0 * diff
            for P, a_target in target_spo.items():
                if P not in evolved:
                    loss_g += a_target * a_target

            total_loss += loss_g

            if self.lambda_ose > 0:
                total_loss += self.lambda_ose * compute_ose(
                    evolved, self.ose_alpha
                )
                for P, g in ose_gradient_seed(
                    evolved, self.lambda_ose, self.ose_alpha
                ).items():
                    seed[P] = seed.get(P, 0.0) + g

            total_grad += propagate_backward(
                evolved, seed, gate_seq, self.n_params, self.delta, stats
            )

        self.last_stats = stats
        self.last_error_estimate = stats.error_estimate
        return total_loss, total_grad

    # ------------------------------------------------------------------
    # Ground State Preparation
    # ------------------------------------------------------------------

    def _ground_state_step(self, params: np.ndarray
                           ) -> Tuple[float, np.ndarray]:
        """One training step for ground-state preparation.

        Propagate H through the HVA, then E = sum_{P in P_{I,Z}} a_P, since
        Tr[|0><0| P] = 1 exactly for the I/Z strings and 0 otherwise.

        Seed (Eq. 24): dL/da_P = 1 on P_{I,Z}. That vector is dense over the
        2^n I/Z strings, so - as the paper does - it is restricted to the
        strings actually carried by the propagated operator.

        Returns:
            (energy, gradient_array)
        """
        gate_seq = self._build_gate_sequence(params)
        stats = TruncationStats()

        evolved = propagate_forward(
            self.hamiltonian_spo, gate_seq, self.delta, stats
        )

        energy = sum(a_P for P, a_P in evolved.items() if is_iz_only(P))
        seed = {P: 1.0 for P in evolved if is_iz_only(P)}
        loss = energy

        if self.lambda_ose > 0:
            loss += self.lambda_ose * compute_ose(evolved, self.ose_alpha)
            for P, g in ose_gradient_seed(
                evolved, self.lambda_ose, self.ose_alpha
            ).items():
                seed[P] = seed.get(P, 0.0) + g

        grad = propagate_backward(
            evolved, seed, gate_seq, self.n_params, self.delta, stats
        )

        self.last_stats = stats
        self.last_error_estimate = stats.error_estimate
        return loss, grad

    # ------------------------------------------------------------------
    # Unified training step
    # ------------------------------------------------------------------

    def train_step(self, params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Execute one training step.

        Returns:
            (loss, gradients) tuple. The truncation diagnostics of the step
            are left in ``self.last_stats`` / ``self.last_error_estimate``.
        """
        if self.mode == 'time_evolution':
            return self._time_evolution_step(params)
        if self.mode == 'ground_state':
            return self._ground_state_step(params)
        raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Adaptive truncation
    # ------------------------------------------------------------------

    def error_scale(self, loss: float) -> float:
        """Coefficient-space quantity the truncation error is compared to.

        eps_emp of Eq. (B16) lives in coefficient space (it is an l2 norm of
        discarded weight), so it has to be compared against something in the
        same units, not against the loss directly:

          * ground state: the energy is *linear* in the coefficients, so its
            error is of order eps_emp itself - compare to |E|;
          * time-evolution: L_{X,Z} is a *squared* coefficient distance, so
            the comparable residual is sqrt(L).

        Returns:
            The scale against which ``error_ratio`` is applied.
        """
        if self.mode == 'ground_state':
            return abs(loss)
        return float(np.sqrt(max(loss, 0.0)))

    def _maybe_tighten_delta(self, epoch: int, loss: float,
                             last_change: int) -> Tuple[bool, int]:
        """Tighten delta when the tracked error dominates the residual."""
        if not self.adaptive_delta or self.delta <= self.min_delta:
            return False, last_change
        if epoch - last_change < self.patience:
            return False, last_change

        scale = self.error_scale(loss)
        if self.last_error_estimate <= self.error_ratio * scale:
            return False, last_change

        self.delta = max(self.min_delta, self.delta * self.delta_factor)
        self.delta_history.append((epoch, self.delta))
        return True, epoch

    # ------------------------------------------------------------------
    # Stage 1: Adam
    # ------------------------------------------------------------------

    def train(self, n_epochs: int = 100, lr: float = 0.01,
              params_init: Optional[np.ndarray] = None,
              verbose: bool = True,
              callback: Optional[Callable] = None,
              beta1: float = 0.9, beta2: float = 0.999,
              eps: float = 1e-8,
              init_seed: int = 42) -> Tuple[np.ndarray, List[float]]:
        """Stage 1 of the optimisation: Adam.

        Args:
            n_epochs: Number of training epochs.
            lr: Learning rate.
            params_init: Initial parameters. Uniform random if None; prefer
                passing a Trotter warm start from bppps.warm_start.
            verbose: Print progress including the tracked truncation error.
            callback: Optional function(epoch, loss, params).
            beta1, beta2, eps: Adam hyperparameters.
            init_seed: Seed for the random fallback initialiser.

        Returns:
            (optimized_params, loss_history)
        """
        if params_init is None:
            params = np.random.default_rng(init_seed).uniform(
                -0.1, 0.1, self.n_params
            )
        else:
            params = np.array(params_init, dtype=float, copy=True)

        m = np.zeros_like(params)
        v = np.zeros_like(params)

        losses: List[float] = []
        t_start = time.time()
        last_delta_change = 0

        for epoch in range(1, n_epochs + 1):
            loss, grad = self.train_step(params)
            losses.append(loss)

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** epoch)
            v_hat = v / (1 - beta2 ** epoch)
            params -= lr * m_hat / (np.sqrt(v_hat) + eps)

            tightened, last_delta_change = self._maybe_tighten_delta(
                epoch, loss, last_delta_change
            )
            if tightened and verbose:
                print(f"    [epoch {epoch}] truncation error "
                      f"{self.last_error_estimate:.2e} exceeded "
                      f"{self.error_ratio:g} x scale; delta -> {self.delta:.1e}")

            if callback:
                callback(epoch, loss, params)

            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == 1):
                print(f"  Epoch {epoch:4d}/{n_epochs}: "
                      f"loss={loss:.6f}, "
                      f"|grad|={np.linalg.norm(grad):.4f}, "
                      f"eps_trunc={self.last_error_estimate:.2e}, "
                      f"delta={self.delta:.1e}, "
                      f"time={time.time() - t_start:.1f}s")

        return params, losses

    # ------------------------------------------------------------------
    # Stage 2: L-BFGS-B
    # ------------------------------------------------------------------

    def refine_lbfgsb(self, params_init: np.ndarray,
                      max_iter: int = 200,
                      tolerance_grad: float = 1e-6,
                      verbose: bool = True
                      ) -> Tuple[np.ndarray, List[float]]:
        """Stage 2: quasi-Newton refinement on the exact analytic gradient.

        BP-PPS passes cost and gradient straight to L-BFGS-B; Adam alone
        leaves the last few digits on the table because its step size never
        adapts to the local curvature. The truncation threshold is held fixed
        here: L-BFGS-B builds a curvature model across iterations and a
        changing delta would make the objective non-stationary.

        Args:
            params_init: Starting point, normally the Adam output.
            max_iter: Maximum L-BFGS-B iterations.
            tolerance_grad: Projected-gradient convergence tolerance (``gtol``).
            verbose: Print a one-line summary of the outcome.

        Returns:
            (refined_params, loss_history)
        """
        history: List[float] = []
        was_adaptive = self.adaptive_delta
        self.adaptive_delta = False

        def objective(x: np.ndarray) -> Tuple[float, np.ndarray]:
            loss, grad = self.train_step(x)
            history.append(loss)
            return float(loss), np.asarray(grad, dtype=float)

        t_start = time.time()
        result = minimize(
            objective,
            np.asarray(params_init, dtype=float),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': tolerance_grad,
                     'maxcor': 20},
        )
        self.adaptive_delta = was_adaptive

        if verbose:
            print(f"  L-BFGS-B: {result.nit} iterations, "
                  f"{result.nfev} evaluations, "
                  f"loss {history[0]:.6f} -> {result.fun:.6f}, "
                  f"eps_trunc={self.last_error_estimate:.2e}, "
                  f"time={time.time() - t_start:.1f}s")
            print(f"    status: {result.message}")

        return result.x, history

    # ------------------------------------------------------------------
    # Full two-stage optimisation
    # ------------------------------------------------------------------

    def optimize(self, optimizer_config: dict,
                 params_init: Optional[np.ndarray] = None,
                 verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """Run the configured Adam -> L-BFGS-B schedule.

        Args:
            optimizer_config: The ``optimizer`` block of the merged YAML
                config (``stage1``, ``stage2``, optionally ``ground_state``).
            params_init: Starting parameters.
            verbose: Print progress.

        Returns:
            (final_params, record) where record holds both loss histories,
            the truncation diagnostics and the final delta.
        """
        stage1 = optimizer_config['stage1']
        stage2 = optimizer_config.get('stage2', {'enabled': False})

        lr = stage1['learning_rate']
        epochs = stage1['epochs']
        if self.mode == 'ground_state' and 'ground_state' in optimizer_config:
            gs = optimizer_config['ground_state']
            lr = gs.get('learning_rate', lr)
            epochs = gs.get('epochs', epochs)

        t_start = time.time()
        params, adam_losses = self.train(
            n_epochs=epochs, lr=lr, params_init=params_init, verbose=verbose,
            beta1=stage1.get('beta1', 0.9), beta2=stage1.get('beta2', 0.999),
            eps=stage1.get('eps', 1e-8),
        )

        lbfgs_losses: List[float] = []
        if stage2.get('enabled', False):
            if verbose:
                print(f"\n  --- Stage 2: {stage2.get('name', 'L-BFGS-B')} ---")
            params, lbfgs_losses = self.refine_lbfgsb(
                params,
                max_iter=stage2.get('max_iter', 200),
                tolerance_grad=stage2.get('tolerance_grad', 1e-6),
                verbose=verbose,
            )

        final_loss = (lbfgs_losses[-1] if lbfgs_losses
                      else (adam_losses[-1] if adam_losses else float('nan')))
        record = {
            'params': params.tolist(),
            'adam_losses': adam_losses,
            'lbfgsb_losses': lbfgs_losses,
            'losses': adam_losses + lbfgs_losses,
            'final_loss': final_loss,
            'n_layers': self.n_layers,
            'n_params': self.n_params,
            'mode': self.mode,
            'final_delta': self.delta,
            'delta_history': self.delta_history,
            'truncation_error_estimate': self.last_error_estimate,
            'n_discarded_terms': self.last_stats.n_discarded,
            'training_time_s': time.time() - t_start,
        }
        return params, record
