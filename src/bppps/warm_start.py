"""Parameter initialisation for the HVA.

BP-PPS Sec. III B reports that random initialisation of the compression
ansatz "only works for layer-2 and layer-3. The optimization of deeper
circuits requires initialization with values from Trotterized circuits."
Since the 10x10 target needs many more layers than that, the Trotter
warm start below is the default initialiser for this project.

The mapping is exact rather than heuristic. One HVA layer is

    RX(theta_q) on every qubit,  then  RZZ(theta_b) on every bond,

which is structurally one step of a first-order product formula for
H = -sum_b J_b Z_i Z_j - h sum_q X_q. Matching

    exp(-i H_X dt) = prod_q exp(+i h dt X_q) = prod_q RX(-2 h dt)
    exp(-i H_ZZ dt) = prod_b exp(+i J_b dt Z_i Z_j) = prod_b RZZ(-2 J_b dt)

with dt = delta_t / n_layers gives an n_layers-deep circuit that already
reproduces exp(-i H delta_t) to first order in dt, so the optimiser starts
inside the basin instead of at a random point of the landscape.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def trotter_warm_start(num_qubits: int, bonds: Sequence[Tuple[int, int]],
                       J: np.ndarray, h: float, delta_t: float,
                       n_layers: int, jitter: float = 0.0,
                       seed: Optional[int] = None) -> np.ndarray:
    """HVA parameters reproducing a first-order Trotter circuit of the same depth.

    Args:
        num_qubits: Number of qubits.
        bonds: Bond list; ``J[k]`` is the coupling of ``bonds[k]``.
        J: Coupling array.
        h: Transverse field strength.
        delta_t: Total evolution time the circuit should represent.
        n_layers: Number of HVA layers; the Trotter step is delta_t / n_layers.
        jitter: Std-dev of optional Gaussian noise added to break the exact
            degeneracy between identical layers. 0 keeps it deterministic.
        seed: RNG seed used only when ``jitter > 0``.

    Returns:
        Parameter vector of length ``n_layers * (num_qubits + len(bonds))``,
        in the layout used by ``build_hva_gate_sequence``.
    """
    num_bonds = len(bonds)
    params_per_layer = num_qubits + num_bonds
    params = np.zeros(n_layers * params_per_layer)

    dt = delta_t / n_layers
    theta_rx = -2.0 * h * dt
    theta_rzz = -2.0 * np.asarray(J, dtype=float) * dt

    for layer in range(n_layers):
        offset = layer * params_per_layer
        params[offset:offset + num_qubits] = theta_rx
        params[offset + num_qubits:offset + params_per_layer] = theta_rzz

    if jitter > 0.0:
        rng = np.random.default_rng(seed)
        params += rng.normal(0.0, jitter, params.shape)

    return params


def random_init(n_params: int, scale: float = 0.1,
                seed: Optional[int] = None) -> np.ndarray:
    """Uniform fallback initialiser on [-scale, scale].

    Matches the range BP-PPS uses for its randomly initialised runs
    (Sec. III B: parameters drawn uniformly from [-0.05, 0.05]).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, n_params)


def build_initial_params(n_params: int, init_config: dict,
                         num_qubits: int, bonds: Sequence[Tuple[int, int]],
                         J: np.ndarray, h: float, delta_t: float,
                         n_layers: int) -> Tuple[np.ndarray, str]:
    """Pick an initialiser from the ``optimizer.init`` config block.

    Returns:
        (params, description) where description names the strategy used, for
        the run log.
    """
    if init_config.get('trotter_warm_start', False):
        params = trotter_warm_start(
            num_qubits, bonds, J, h, delta_t, n_layers,
            jitter=float(init_config.get('jitter', 0.0)),
            seed=init_config.get('seed'),
        )
        return params, f"Trotter warm start (dt={delta_t / n_layers:.4g})"

    scale = float(init_config.get('random_scale', 0.1))
    params = random_init(n_params, scale, init_config.get('seed'))
    return params, f"uniform random +-{scale}"
