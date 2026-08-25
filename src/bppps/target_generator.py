"""Target SPO generation using fine-grained Trotter propagation.

Generates target (ground truth) data for BP-PPS training by propagating
local observables X_i, Z_i through a precise Trotter circuit via SPD.
"""

import numpy as np
from typing import Dict, List, Tuple

from .pauli_utils import make_observable_label
from .propagation import (
    SPO,
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
                 order: int = 2, delta: float = 1e-8,
                 observables: str = 'XZ',
                 verbose: bool = True) -> Dict[str, SPO]:
        """Generate target SPOs for all local observables.

        Args:
            delta_t: Total time to simulate (U(Δt)).
            dt_trotter: Fine Trotter step size for accuracy.
            order: Trotter formula order (1 or 2).
            delta: Truncation threshold for Pauli propagation.
            observables: Which observables to generate ('XZ', 'X', 'Z').
            verbose: Print progress.

        Returns:
            Dict mapping observable key (e.g., 'X_0', 'Z_3') to target SPO.
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

            evolved = propagate_forward(init_spo, gate_seq, delta)
            targets[key] = evolved

            if verbose and (idx + 1) % max(1, len(obs_list) // 10) == 0:
                print(f"    [{idx+1}/{len(obs_list)}] {key}: "
                      f"{len(evolved)} Pauli terms")

        if verbose:
            total_terms = sum(len(v) for v in targets.values())
            print(f"  Target generation complete: "
                  f"{len(targets)} observables, {total_terms} total terms")

        return targets

    def _count_gates(self, n_steps: int, order: int) -> int:
        """Count total gates in Trotter sequence."""
        n_bonds = sum(len(v) for v in self.substep_bonds.values())
        if order == 1:
            return n_steps * (n_bonds + self.num_qubits)
        else:
            return n_steps * (n_bonds + 2 * self.num_qubits)
