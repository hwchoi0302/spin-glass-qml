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
from .pauli_utils import make_observable_label, product_state_filter
from .propagation import (
    SPO,
    TruncationStats,
    build_hva_gate_sequence,
    propagate_backward,
    propagate_forward,
)
from .propagation_packed import label_to_xz
from .propagation_sorted import (
    MASK32 as _MASK32,
    _lookup,
    propagate_backward_sorted,
    propagate_forward_sorted,
    to_sorted_arrays,
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
        initial_state: str = 'zero',
        engine: str = 'string',
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

        # ||t||^2 per observable. The targets are fixed for the whole
        # optimisation, so this is the one part of the time-evolution loss
        # that need never be recomputed -- see _time_evolution_step.
        self._target_sq_norm = {
            key: sum(c * c for c in spo.values())
            for key, spo in self.target_spos.items()
        }

        # Product state the ground-state circuit starts from. |+...+> is the
        # default choice for this model: Pi_i X_i commutes with H and with every
        # HVA gate, |+...+> is its +1 eigenstate, and stoquasticity forces the
        # ground state into that same +1 sector at any size (Perron-Frobenius
        # makes the ground-state amplitudes positive, so Pi X cannot flip its
        # sign). Starting from |0...0> instead splits the circuit evenly across
        # both parity sectors and caps the ground-state fidelity at 0.5.
        self.initial_state = initial_state
        self._state_filter = product_state_filter(initial_state)

        # Which propagation engine the gradient steps run on. 'string' is the
        # Dict[str, float] implementation in propagation.py -- the oracle every
        # other engine is validated against (TESTs 10-21), and the only one
        # that was ever wired into training. 'sorted' is the whole-array
        # engine of propagation_sorted.py, measured 18.2x faster on the real
        # workload (X_5 at the production L=3 angles, delta=1e-6, 396,896
        # terms: 21.65 s -> 1.19 s, gradients agreeing to 1.1e-14 relative).
        #
        # The default stays 'string' so nothing changes for a caller that does
        # not ask. CLAUDE.md's rule that 4x4 stays on string-keyed dicts is
        # about keeping that oracle available, and it still is.
        if engine not in ('string', 'sorted'):
            raise ValueError(f"engine must be 'string' or 'sorted', got {engine!r}")
        self.engine = engine
        if engine == 'sorted':
            if lambda_ose > 0:
                raise NotImplementedError(
                    "the OSE regulariser has no sorted-array form; run with "
                    "engine='string' or lambda_ose=0")
            self._sorted_setup()

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
    # Sorted-array engine
    # ------------------------------------------------------------------

    @staticmethod
    def _packed_keys(labels):
        """String Pauli labels -> the uint64 keys propagation_sorted uses."""
        out = np.empty(len(labels), dtype=np.uint64)
        for i, label in enumerate(labels):
            x, z = label_to_xz(label)
            out[i] = np.uint64(x) | (np.uint64(z) << np.uint64(32))
        return out

    def _sorted_setup(self):
        """Convert the fixed inputs to sorted arrays, once.

        The targets and the Hamiltonian SPO do not change during an
        optimisation, so their string->key conversion belongs here rather than
        in a step that runs thousands of times.
        """
        self._target_sorted = {}
        for key, spo in self.target_spos.items():
            self._target_sorted[key] = to_sorted_arrays(
                {label_to_xz(P): c for P, c in spo.items()})

        self._obs_key = {}
        for key in self.target_spos:
            pauli, q_str = key.split('_')
            label = make_observable_label(self.num_qubits, pauli, int(q_str))
            x, z = label_to_xz(label)
            self._obs_key[key] = np.uint64(x) | (np.uint64(z) << np.uint64(32))

        if self.hamiltonian_spo:
            self._ham_sorted = to_sorted_arrays(
                {label_to_xz(P): c for P, c in self.hamiltonian_spo.items()})
        else:
            self._ham_sorted = (np.zeros(0, dtype=np.uint64),
                                np.zeros(0, dtype=np.float64))

        # <s|P|s> = 1 exactly on the I/Z strings for |0...0> and the I/X
        # strings for |+...+>. In key space those are "no x bits set" and "no z
        # bits set" -- one mask test, replacing pauli_utils' per-character
        # string scan.
        if self.initial_state == 'zero':
            self._keep_mask = lambda k: (k & _MASK32) == np.uint64(0)
        elif self.initial_state == 'plus':
            self._keep_mask = lambda k: (k >> np.uint64(32)) == np.uint64(0)
        else:
            raise ValueError(
                f"no key-space filter for initial_state={self.initial_state!r}")

    def _time_evolution_step_sorted(self, params: np.ndarray
                                    ) -> Tuple[float, np.ndarray]:
        """_time_evolution_step on the sorted-array engine.

        Same quantities, same truncation rule, no string keys: the
        target-only residual is still ||t||^2 minus the intersection, and the
        intersection is now one vectorised lookup instead of a dict get per
        surviving term.
        """
        gate_seq = self._build_gate_sequence(params)
        stats = TruncationStats()

        total_loss = 0.0
        total_grad = np.zeros(self.n_params)

        for obs_key in self.target_spos:
            t_keys, t_vals = self._target_sorted[obs_key]
            init_keys = np.array([self._obs_key[obs_key]], dtype=np.uint64)
            init_vals = np.array([1.0], dtype=np.float64)

            keys, coeffs = propagate_forward_sorted(
                init_keys, init_vals, gate_seq, self.delta, np, stats)

            t_hit = _lookup(t_keys, t_vals, keys, np)
            diff = coeffs - t_hit
            loss_g = float(np.sum(diff * diff))
            loss_g += self._target_sq_norm[obs_key] - float(np.sum(t_hit * t_hit))
            total_loss += loss_g

            total_grad += propagate_backward_sorted(
                keys, coeffs, 2.0 * diff, gate_seq, self.n_params,
                self.delta, np, stats)

        self.last_stats = stats
        self.last_error_estimate = stats.error_estimate
        return total_loss, total_grad

    def _ground_state_step_sorted(self, params: np.ndarray
                                  ) -> Tuple[float, np.ndarray]:
        """_ground_state_step on the sorted-array engine."""
        gate_seq = self._build_gate_sequence(params)
        stats = TruncationStats()

        h_keys, h_vals = self._ham_sorted
        keys, coeffs = propagate_forward_sorted(
            h_keys, h_vals, gate_seq, self.delta, np, stats)

        keep = self._keep_mask(keys)
        energy = float(np.sum(coeffs[keep]))
        seed = np.where(keep, 1.0, 0.0)

        grad = propagate_backward_sorted(
            keys, coeffs, seed, gate_seq, self.n_params, self.delta, np, stats)

        self.last_stats = stats
        self.last_error_estimate = stats.error_estimate
        return energy, grad

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
            #
            # The target-only strings still have to be counted, but they must
            # not be *walked*: the target is fixed for the whole optimisation
            # while `evolved` is small, so scanning it once per iteration made
            # the step cost O(|target|) forever. Instead
            #
            #   sum_{P in target \ evolved} t_P^2
            #       = ||t||^2 - sum_{P in target ^ evolved} t_P^2
            #
            # where ||t||^2 is precomputed once (self._target_sq_norm) and the
            # intersection term is accumulated in the loop that already looks
            # each t_P up. Measured at a 1.2M-term target against a 3K-term
            # evolved SPO -- the actual training regime, where the target
            # dwarfs what a short ansatz produces -- that is 92 ms -> 0.77 ms
            # per observable per iteration, 120x.
            #
            # Same quantity, not an approximation: checked against the old
            # formula on generated targets at 2x2 and 3x3 (936,200 terms),
            # agreeing to ~1e-14 relative, which is summation order alone. The
            # gradient is untouched -- it comes from `seed`, which never
            # referenced the target-only strings (their derivative w.r.t. the
            # evolved coefficients is zero).
            loss_g = 0.0
            t_sq_hit = 0.0
            seed = {}
            for P, a_P in evolved.items():
                t_P = target_spo.get(P, 0.0)
                diff = a_P - t_P
                loss_g += diff * diff
                t_sq_hit += t_P * t_P
                if diff != 0.0:
                    seed[P] = 2.0 * diff
            loss_g += self._target_sq_norm[obs_key] - t_sq_hit

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

        Propagate H through the HVA, then E = sum over the Pauli strings with
        unit expectation value in the initial product state, since Tr[|s><s| P]
        is 1 for those and 0 otherwise. For |0...0> those are the I/Z strings;
        for |+...+> the I/X strings. See ``initial_state``.

        Seed (Eq. 24): dL/da_P = 1 on that same set. The vector is dense over
        the 2^n strings, so - as the paper does - it is restricted to the
        strings actually carried by the propagated operator.

        Returns:
            (energy, gradient_array)
        """
        gate_seq = self._build_gate_sequence(params)
        stats = TruncationStats()

        evolved = propagate_forward(
            self.hamiltonian_spo, gate_seq, self.delta, stats
        )

        keep = self._state_filter
        energy = sum(a_P for P, a_P in evolved.items() if keep(P))
        seed = {P: 1.0 for P in evolved if keep(P)}
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
            if self.engine == 'sorted':
                return self._time_evolution_step_sorted(params)
            return self._time_evolution_step(params)
        if self.mode == 'ground_state':
            if self.engine == 'sorted':
                return self._ground_state_step_sorted(params)
            return self._ground_state_step(params)
        raise ValueError(f"Unknown mode: {self.mode}")

    # ------------------------------------------------------------------
    # Adaptive truncation
    # ------------------------------------------------------------------

    def recent_progress(self, losses: List[float]) -> float:
        """How much the loss has moved over the last ``patience`` epochs.

        This is the signal the truncation error has to be compared against for
        ground-state runs; see :meth:`error_scale`.

        Args:
            losses: Loss history, most recent last.

        Returns:
            |L(now) - L(patience epochs ago)|, or +inf while the history is
            shorter than that (too early to judge, so never tighten on it).
        """
        if len(losses) <= self.patience:
            return float('inf')
        return abs(losses[-1 - self.patience] - losses[-1])

    def error_scale(self, loss: float,
                    recent_progress: Optional[float] = None) -> float:
        """Quantity the tracked truncation error is compared against.

        eps_emp of Eq. (B16) lives in coefficient space (it is an l2 norm of
        discarded weight), so it has to be compared against something in the
        same units, not against the loss directly.

        For time evolution that is easy: L_{X,Z} is a *squared* coefficient
        distance, so the comparable residual is sqrt(L), which falls as the fit
        improves and so eventually trips the threshold.

        For the ground state there is no such residual - the energy is not a
        distance to anything known, and it does not go to zero. This used to
        compare against |E|, which does not work: |E| is set by the size of the
        lattice, not by how well converged the run is, so the threshold is
        enormous and never trips. At 4x4 that was 0.1 * 21.207 = 2.121 against a
        tracked error of 0.1205, and the ground-state run finished at its
        *starting* delta of 1e-3 having never tightened once. It gets worse with
        size, since |E| grows with the bond count while the precision needed
        does not: at 10x10 the threshold would be about 16.

        What matters instead is whether truncation noise has caught up with the
        progress the optimiser is still making. So the scale is the energy
        decrease over the last ``patience`` epochs: large early on, shrinking
        towards zero as the run converges, which is exactly when a smaller delta
        is worth paying for. It needs no oracle and no absolute tolerance, and
        it is independent of lattice size.

        Args:
            loss: Current loss value.
            recent_progress: Output of :meth:`recent_progress`. Ground-state
                runs fall back to the old |E| behaviour when it is omitted,
                so direct callers of this method keep working.

        Returns:
            The scale against which ``error_ratio`` is applied.
        """
        if self.mode == 'ground_state':
            if recent_progress is None:
                return abs(loss)
            return recent_progress
        return float(np.sqrt(max(loss, 0.0)))

    def _maybe_tighten_delta(self, epoch: int, losses: List[float],
                             last_change: int) -> Tuple[bool, int]:
        """Tighten delta when the tracked error dominates the residual.

        Args:
            epoch: Current epoch, 1-based.
            losses: Loss history, most recent last.
            last_change: Epoch at which delta last changed.

        Returns:
            (whether delta was tightened, updated last_change)
        """
        if not self.adaptive_delta or self.delta <= self.min_delta:
            return False, last_change
        if epoch - last_change < self.patience:
            return False, last_change

        scale = self.error_scale(losses[-1], self.recent_progress(losses))
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
                epoch, losses, last_delta_change
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

    def _record(self, params: np.ndarray, adam_losses: List[float],
               lbfgsb_losses: List[float], t_start: float,
               status: Optional[str] = None) -> dict:
        """Build the JSON-serialisable record :meth:`optimize` returns.

        Factored out so a checkpoint written after Adam and the final record
        written after L-BFGS-B are the same shape -- a consumer does not need
        to know which one it got, only whether ``lbfgsb_losses`` is empty.
        """
        final_loss = (lbfgsb_losses[-1] if lbfgsb_losses
                      else (adam_losses[-1] if adam_losses else float('nan')))
        record = {
            'params': params.tolist(),
            'adam_losses': adam_losses,
            'lbfgsb_losses': lbfgsb_losses,
            'losses': adam_losses + lbfgsb_losses,
            'final_loss': final_loss,
            'n_layers': self.n_layers,
            'initial_state': self.initial_state,
            'n_params': self.n_params,
            'mode': self.mode,
            'final_delta': self.delta,
            'delta_history': self.delta_history,
            'truncation_error_estimate': self.last_error_estimate,
            'n_discarded_terms': self.last_stats.n_discarded,
            'training_time_s': time.time() - t_start,
        }
        if status is not None:
            record['status'] = status
        return record

    def optimize(self, optimizer_config: dict,
                 params_init: Optional[np.ndarray] = None,
                 verbose: bool = True,
                 checkpoint_path: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        """Run the configured Adam -> L-BFGS-B schedule.

        Args:
            optimizer_config: The ``optimizer`` block of the merged YAML
                config (``stage1``, ``stage2``, optionally ``ground_state``).
            params_init: Starting parameters.
            verbose: Print progress.
            checkpoint_path: If given, the Adam-only record is written here as
                JSON the moment Adam finishes, before L-BFGS-B starts. This is
                the fix for the failure that discarded a 41 h ground-state run
                at L=5 (results/4x4/gs_L5_aborted.json): the pipeline used to
                write nothing until *both* stages returned, so a kill inside
                L-BFGS-B lost 19.9 h of completed Adam optimisation along with
                the angles that produced it. A checkpoint costs one JSON write
                of a few hundred KB and makes an interrupted run recoverable
                rather than a total loss. L-BFGS-B still overwrites this path
                with the complete record on normal completion.

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

        if checkpoint_path is not None:
            import json as _json
            checkpoint = self._record(
                params, adam_losses, [], t_start,
                status='Adam complete; L-BFGS-B not yet run or not finished. '
                       'If this is the newest record for this run, the '
                       'process was interrupted during stage 2 -- these '
                       'angles are the last confirmed-good checkpoint.')
            with open(checkpoint_path, 'w') as f:
                _json.dump(checkpoint, f, indent=2)
            if verbose:
                print(f"  Checkpoint after Adam: {checkpoint_path}")

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

        record = self._record(params, adam_losses, lbfgs_losses, t_start)
        return params, record
