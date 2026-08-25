"""Utility functions for Pauli string operations.

Pauli string convention:
    - N-qubit Pauli string is a length-N string of {'I','X','Y','Z'}.
    - Qubit ordering: label[k] acts on qubit k (position 0 = qubit 0).
    - This is independent of Qiskit's endianness; the mapping is handled
      elsewhere when interfacing with Qiskit circuits.
"""

from typing import Tuple

# --------------------------------------------------------------------------
# Single-qubit Pauli multiplication table
# (a, b) -> (result, phase) where a·b = phase · result
# --------------------------------------------------------------------------
_PAULI_MULT = {
    ('I', 'I'): ('I', 1),  ('I', 'X'): ('X', 1),  ('I', 'Y'): ('Y', 1),  ('I', 'Z'): ('Z', 1),
    ('X', 'I'): ('X', 1),  ('X', 'X'): ('I', 1),  ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
    ('Y', 'I'): ('Y', 1),  ('Y', 'X'): ('Z', -1j),('Y', 'Y'): ('I', 1),  ('Y', 'Z'): ('X', 1j),
    ('Z', 'I'): ('Z', 1),  ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j),('Z', 'Z'): ('I', 1),
}


def commutes(label1: str, label2: str) -> bool:
    """Check if two N-qubit Pauli strings commute.

    Two N-qubit Pauli strings commute iff the number of qubit positions
    where both are non-identity and different is even.
    """
    n_anti = 0
    for a, b in zip(label1, label2):
        if a != 'I' and b != 'I' and a != b:
            n_anti += 1
    return n_anti % 2 == 0


def pauli_product(label1: str, label2: str) -> Tuple[str, complex]:
    """Compute product of two N-qubit Pauli strings.

    Returns:
        (result_label, phase) where label1 · label2 = phase · result_label
        and phase ∈ {+1, -1, +i, -i}.
    """
    result = []
    phase = 1
    for a, b in zip(label1, label2):
        r, p = _PAULI_MULT[(a, b)]
        result.append(r)
        phase *= p
    return ''.join(result), phase


def is_iz_only(label: str) -> bool:
    """Check if a Pauli string contains only I and Z (no X or Y).

    Used to identify strings that contribute to ⟨0|P|0⟩ = 1.
    """
    return all(c in ('I', 'Z') for c in label)


def make_observable_label(num_qubits: int, pauli: str, qubit: int) -> str:
    """Create a single-site observable label.

    Example: make_observable_label(4, 'X', 1) -> 'IXII'
    """
    label = ['I'] * num_qubits
    label[qubit] = pauli
    return ''.join(label)
