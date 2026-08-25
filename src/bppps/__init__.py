"""BP-PPS: Backpropagating Pauli Propagation Simulation.

Implements the gradient-based training of variational circuits using
Pauli propagation in the Heisenberg picture, as described in
"Backpropagation through Pauli Propagation Simulations" (2025).
"""

from .propagation import (
    propagate_forward,
    propagate_backward,
    build_hva_gate_sequence,
    build_trotter_gate_sequence,
)
from .target_generator import TargetGenerator
from .ose_regularizer import compute_ose, ose_gradient_seed
from .trainer import BPPPSTrainer
