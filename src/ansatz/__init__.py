"""Ansatz module for the spin-glass-qml project.

This module provides quantum circuit ansatzes for 2D spin glass models,
including Hamiltonian Variational Ansatz (HVA) and Trotterized time evolution.
"""

from .hva import HVA
from .trotter import TrotterCircuit

__all__ = ["HVA", "TrotterCircuit"]
