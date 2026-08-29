"""Trotter circuit for time evolution using Qiskit's PauliEvolutionGate.

Uses Qiskit's built-in Suzuki-Trotter / Lie-Trotter decomposition
for Hamiltonian time evolution exp(-iHt) on a square lattice.
"""

import numpy as np
from typing import List, Tuple, Union, Optional
import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter
from qiskit.quantum_info import SparsePauliOp


class TrotterCircuit:
    """Trotter circuit for time evolution of 2D spin glass models.

    Uses Qiskit's PauliEvolutionGate with Suzuki-Trotter synthesis.
    The Hamiltonian is provided as a SparsePauliOp.

    Attributes:
        hamiltonian_op (SparsePauliOp): The Hamiltonian in Pauli form.
        num_qubits (int): Number of qubits.
    """

    def __init__(
        self,
        hamiltonian_op: SparsePauliOp,
        num_qubits: int,
    ) -> None:
        """Initialize the Trotter circuit builder.

        Args:
            hamiltonian_op: Hamiltonian as SparsePauliOp from
                            SpinGlass2D.get_pauli_terms().
            num_qubits: Number of qubits in the system.
        """
        self.hamiltonian_op = hamiltonian_op
        self.num_qubits = num_qubits

    def num_steps(self, t: float, dt: float) -> int:
        """Number of Trotter steps used to reach time ``t`` with step size ``dt``.

        Rounds *up*, so the circuit never stops short of ``t``. The step size is
        then shrunk to ``t / steps`` (see :meth:`build_circuit`) and the circuit
        lands on ``t`` exactly.

        This used to round to nearest and evolve for ``steps * dt``, which meant
        the circuit reached a different time than the one it was compared
        against whenever ``dt`` did not divide ``t``. At dt=0.2 and t=0.5 it
        built 2 steps and evolved to 0.4, and the fidelity against the exact
        state at 0.5 came out as 0.850 - almost all of that 0.15 infidelity was
        the 0.1 time offset, not Trotter error. On the t = 0.5 ... 2.5 grid that
        produced a sawtooth (bad at 0.5, 1.5, 2.5; fine at 1.0, 2.0) that looked
        like a property of the Trotter formula and was not one. Rounding to zero
        for t < dt/2 also silently dropped points, which callers had to work
        around.
        """
        return max(1, math.ceil(t / dt - 1e-12))

    def build_circuit(
        self,
        t: float,
        dt: float,
        order: int = 2,
    ) -> QuantumCircuit:
        """Build the Trotter circuit for time evolution exp(-iHt).

        Creates n_steps repetitions of PauliEvolutionGate, each evolving
        the Hamiltonian for time dt. The gate is synthesized using
        Lie-Trotter (order=1) or Suzuki-Trotter (order≥2).

        Args:
            t: Total simulation time.
            dt: Time step size per Trotter step.
            order: Trotter formula order (1 = Lie-Trotter, 2+ = Suzuki-Trotter).

        Returns:
            QuantumCircuit: The constructed Trotter circuit, decomposed
            into native RX/RZZ gates.
        """
        steps = self.num_steps(t, dt)
        qc = QuantumCircuit(self.num_qubits)

        if t == 0:
            return qc

        # Land on t exactly: dt is an upper bound on the step size, not a
        # divisor of t. dt_eff <= dt always, so the accuracy claim attached to
        # dt still holds.
        dt_eff = t / steps

        # Choose synthesis method
        if order == 1:
            synthesis = LieTrotter()
        else:
            synthesis = SuzukiTrotter(order=order)

        # Create evolution gate for one time step dt
        evo_gate = PauliEvolutionGate(
            self.hamiltonian_op,
            time=dt_eff,
            synthesis=synthesis,
        )

        # Repeat for n_steps
        for _ in range(steps):
            qc.append(evo_gate, range(self.num_qubits))

        # Decompose PauliEvolutionGate into native gates
        return qc.decompose()

    def circuit_depth(self, t: float, dt: float, order: int = 2) -> int:
        """Estimate the circuit depth.

        Args:
            t: Total time.
            dt: Time step size.
            order: Trotter formula order.

        Returns:
            int: Estimated circuit depth (before transpilation).
        """
        qc = self.build_circuit(t, dt, order)
        return qc.depth()

    def count_2q_gates(self, t: float, dt: float, order: int = 2) -> int:
        """Count the number of 2-qubit gates.

        Args:
            t: Total time.
            dt: Time step size.
            order: Trotter formula order.

        Returns:
            int: Number of 2-qubit gates.
        """
        qc = self.build_circuit(t, dt, order)
        ops = qc.count_ops()
        return ops.get('rzz', 0) + ops.get('cx', 0) + ops.get('cz', 0)
