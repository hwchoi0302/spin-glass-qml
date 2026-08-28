"""Target SPO generation using fine-grained Trotter propagation.

Generates target (ground truth) data for BP-PPS training by propagating
local observables X_i, Z_i through a precise Trotter circuit via SPD.
"""

import time

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

from .pauli_utils import make_observable_label
from .propagation import (
    SPO,
    TruncationStats,
    build_trotter_gate_sequence,
    propagate_forward,
)


class TargetGenerator:
    """Generate target SPOs for time-evolution compression.

    Propagates each local observable (X_i, Z_i) through a fine-grained
    Trotter circuit to obtain the "ground truth" evolved operators.

    Attributes:
        num_qubits: Number of qubits.
        bonds: List of (i, j) bond tuples.
        substep_bonds: Brickwork classification of bonds.
        J: Coupling constants array.
        h: Transverse field strength.
    """

    def __init__(self, num_qubits: int, bonds: List[Tuple[int, int]],
                 substep_bonds: dict, J: np.ndarray, h: float):
        self.num_qubits = num_qubits
        self.bonds = bonds
        self.substep_bonds = substep_bonds
        self.J = J
        self.h = h

    def generate(self, delta_t: float, dt_trotter: float = 0.001,
                 order: int = 4, delta: float = 1e-8,
                 observables: str = 'XZ',
                 verbose: bool = True) -> Tuple[Dict[str, SPO], TruncationStats]:
        """Generate target SPOs for all local observables.

        The defaults are the BP-PPS paper's precision settings (Sec. III B):
        4th-order Suzuki-Trotter, dt = 0.001, threshold 1e-8, which makes the
        target a numerically exact stand-in for exp(-iH*delta_t) rather than a
        circuit anyone would run on hardware.

        Args:
            delta_t: Total time to simulate, i.e. the target is V = U(delta_t).
            dt_trotter: Fine Trotter step size.
            order: Trotter formula order (1, 2 or 4).
            delta: Truncation threshold for Pauli propagation.
            observables: Which observables to generate ('XZ', 'X', 'Z').
            verbose: Print progress.

        Returns:
            (targets, stats) where targets maps an observable key such as
            'X_0' to its target SPO, and stats carries the Appendix B
            truncation-error estimate accumulated over the whole generation.
        """
        n_steps = int(round(delta_t / dt_trotter))
        if verbose:
            print(f"  Trotter: dt={dt_trotter}, steps={n_steps}, "
                  f"total gates={self._count_gates(n_steps, order)}")

        # Build Trotter gate sequence (once, reused for all observables)
        gate_seq = build_trotter_gate_sequence(
            self.num_qubits, self.substep_bonds,
            self.J, self.h, dt_trotter, n_steps, order
        )

        targets = {}
        stats = TruncationStats()

        # Generate for each local observable
        obs_list = []
        if 'X' in observables:
            obs_list += [('X', q) for q in range(self.num_qubits)]
        if 'Z' in observables:
            obs_list += [('Z', q) for q in range(self.num_qubits)]

        for idx, (pauli, q) in enumerate(obs_list):
            key = f"{pauli}_{q}"
            label = make_observable_label(self.num_qubits, pauli, q)
            init_spo = {label: 1.0}

            evolved = propagate_forward(init_spo, gate_seq, delta, stats)
            targets[key] = evolved

            if verbose and (idx + 1) % max(1, len(obs_list) // 10) == 0:
                print(f"    [{idx+1}/{len(obs_list)}] {key}: "
                      f"{len(evolved)} Pauli terms")

        if verbose:
            total_terms = sum(len(v) for v in targets.values())
            print(f"  Target generation complete: "
                  f"{len(targets)} observables, {total_terms} total terms")

        return targets, stats


    def generate_series(self, delta_t: float, snapshots: List[int],
                        dt_trotter: float = 0.001, order: int = 4,
                        delta: float = 1e-8, observables: str = 'XZ',
                        verbose: bool = True,
                        checkpoint: Optional[Callable[[int, Dict[str, SPO],
                                                       TruncationStats], None]] = None
                        ) -> Dict[int, Dict[str, SPO]]:
        """Targets at T = k * delta_t for several k, for the price of the largest.

        The target unitary for k chunks is V_k = B^k with B = Trotter(delta_t),
        so in the Heisenberg picture

            V_k^dag O V_k = (B^dag)^k O B^k

        which is just "apply the one-chunk propagation k times". Propagating
        chunk by chunk and snapshotting therefore yields every intermediate
        time on the way to the largest one, instead of restarting from scratch
        for each T. For snapshots [1,2,4,8,16] that is a 31x saving.

        Args:
            delta_t: Time per chunk.
            snapshots: Chunk counts k to record, e.g. [1, 2, 4, 8, 16].
            dt_trotter: Fine Trotter step inside one chunk.
            order: Trotter formula order.
            delta: Truncation threshold.
            observables: Which observables ('XZ', 'X', 'Z').
            verbose: Print progress.
            checkpoint: Optional callback(k, targets_at_k, stats) invoked as
                soon as each snapshot completes, so a long run can be resumed.

        Returns:
            Dict mapping k to the target SPOs at T = k * delta_t.
        """
        snapshots = sorted(set(int(k) for k in snapshots))
        if not snapshots or snapshots[0] < 1:
            raise ValueError(f"snapshots must be positive integers, got {snapshots}")
        k_max = snapshots[-1]

        n_steps = int(round(delta_t / dt_trotter))
        block = build_trotter_gate_sequence(
            self.num_qubits, self.substep_bonds, self.J, self.h,
            dt_trotter, n_steps, order,
        )
        if verbose:
            print(f"  chunk: delta_t={delta_t}, dt={dt_trotter}, order={order}, "
                  f"{len(block)} gates")
            print(f"  snapshots at k = {snapshots} "
                  f"(T = {[k * delta_t for k in snapshots]})")

        obs_list = []
        if 'X' in observables:
            obs_list += [('X', q) for q in range(self.num_qubits)]
        if 'Z' in observables:
            obs_list += [('Z', q) for q in range(self.num_qubits)]

        # One running SPO per observable, advanced one chunk at a time.
        running = {}
        for pauli, q in obs_list:
            label = make_observable_label(self.num_qubits, pauli, q)
            running[f"{pauli}_{q}"] = {label: 1.0}

        stats = TruncationStats()
        series: Dict[int, Dict[str, SPO]] = {}
        t_start = time.time()

        for k in range(1, k_max + 1):
            for key in running:
                running[key] = propagate_forward(running[key], block, delta, stats)

            if k in snapshots:
                series[k] = {key: dict(spo) for key, spo in running.items()}
                if verbose:
                    total = sum(len(v) for v in series[k].values())
                    largest = max(len(v) for v in series[k].values())
                    print(f"    k={k:3d} (T={k * delta_t:5.2f}): "
                          f"{total:9d} terms total, {largest:8d} largest, "
                          f"eps_trunc={stats.error_estimate:.3e}, "
                          f"{time.time() - t_start:7.1f}s")
                if checkpoint is not None:
                    checkpoint(k, series[k], stats)

        return series

    def _count_gates(self, n_steps: int, order: int) -> int:
        """Count total gates in the Trotter sequence.

        One S2 step costs n_bonds RZZ plus 2*n_qubits RX; S4 is five S2 steps.
        """
        n_bonds = sum(len(v) for v in self.substep_bonds.values())
        if order == 1:
            return n_steps * (n_bonds + self.num_qubits)
        per_s2 = n_bonds + 2 * self.num_qubits
        n_s2 = 5 if order == 4 else 1
        return n_steps * n_s2 * per_s2
