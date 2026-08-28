"""BP-PPS: Backpropagating Pauli Propagation.

Classical, memory-efficient gradient-based optimisation of parameterised
quantum circuits via sparse Pauli dynamics in the Heisenberg picture, after

    S.-H. Lin, E. Granet, K. Hemery, H. Dreyer,
    "Backpropagating Pauli Propagation", arXiv:2607.15184 (2026).

Equation numbers quoted throughout this package refer to that paper.
"""

from .ose_regularizer import compute_ose, operator_norm_sq, ose_gradient_seed
from .propagation import (
    TruncationStats,
    build_hva_gate_sequence,
    build_trotter_gate_sequence,
    propagate_backward,
    propagate_forward,
)
from .target_generator import TargetGenerator
from .trainer import BPPPSTrainer
from .warm_start import build_initial_params, random_init, trotter_warm_start

__all__ = [
    'TruncationStats',
    'propagate_forward',
    'propagate_backward',
    'build_hva_gate_sequence',
    'build_trotter_gate_sequence',
    'TargetGenerator',
    'BPPPSTrainer',
    'compute_ose',
    'ose_gradient_seed',
    'operator_norm_sq',
    'trotter_warm_start',
    'random_init',
    'build_initial_params',
]
