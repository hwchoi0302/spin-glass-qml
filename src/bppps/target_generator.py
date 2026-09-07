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
from .propagation_packed import label_to_xz
from .propagation_sorted import (
    pack_key,
    propagate_forward_sorted,
    sorted_to_labelled_spo,
)
from .propagation_wide import (
    from_wide_arrays,
    n_words,
    pick_kernel,
    propagate_forward_wide,
    to_wide_arrays,
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
                 substep_bonds: dict, J: np.ndarray, h: float,
                 engine: str = 'string'):
        self.num_qubits = num_qubits
        self.bonds = bonds
        self.substep_bonds = substep_bonds
        self.J = J
        self.h = h

        # Which propagation kernel the Trotter walk runs on. This is where the
        # project spends most of its compute -- one 4x4 observable at the
        # production delta=1e-8 took 822 s and one 5x5 observable 2.5 h -- and
        # until now it was the string-dict oracle, the slowest engine there is.
        #
        #   'string'  propagation.py. The oracle. Unchanged, still the default.
        #   'sorted'  propagation_sorted.py on numpy.
        #   'gpu'     the same code with xp=cupy. Worth it only above the
        #             crossover: cupy loses at 30K terms (0.23x) and wins at
        #             100K (2.0x), 300K (12.3x) and 1.2M (17.3x). Target
        #             generation is the workload that sits on the winning side.
        #
        # The *representation* is chosen by lattice size, not by the caller:
        # propagation_sorted's one-uint64 key holds 32 qubits, and above that
        # propagation_wide carries W words of x followed by W words of z. That
        # choice used to be a footgun -- 7x7 in the packed key collided
        # silently -- so pick_kernel() makes it automatic.
        if engine not in ('string', 'sorted', 'gpu'):
            raise ValueError(
                f"engine must be 'string', 'sorted' or 'gpu', got {engine!r}")
        self.engine = engine
        self.xp = np
        self.kernel = 'string' if engine == 'string' else pick_kernel(num_qubits)
        if engine == 'gpu':
            try:
                import cupy as cp
                cp.cuda.Device(0).compute_capability
            except Exception as exc:
                raise RuntimeError(
                    f"engine='gpu' needs cupy and a CUDA device ({exc}). On "
                    f"the desktop, source scripts/cuda12_env.sh first -- "
                    f"torch's CUDA 13 wheel otherwise hides libcublas.so.12."
                ) from exc
            self.xp = cp

    # ------------------------------------------------------------------
    # Engine dispatch
    # ------------------------------------------------------------------

    def _seed_spo(self, label: str):
        """The single-term starting SPO, in whichever representation."""
        xp = self.xp
        if self.kernel == 'string':
            return {label: 1.0}
        if self.kernel == 'packed':
            return (xp.asarray([pack_key(*label_to_xz(label))],
                               dtype=xp.uint64),
                    xp.asarray([1.0], dtype=xp.float64))
        return to_wide_arrays({label: 1.0}, self.num_qubits, xp)

    def _advance(self, spo, gate_seq, delta, stats):
        """One propagation of `spo` through `gate_seq`, engine-agnostic."""
        if self.kernel == 'string':
            return propagate_forward(spo, gate_seq, delta, stats)
        keys, coeffs = spo
        if self.kernel == 'packed':
            return propagate_forward_sorted(keys, coeffs, gate_seq, delta,
                                            xp=self.xp, stats=stats)
        return propagate_forward_wide(keys, coeffs, gate_seq, self.num_qubits,
                                      delta, xp=self.xp, stats=stats)

    @staticmethod
    def _size(spo) -> int:
        return len(spo) if isinstance(spo, dict) else int(spo[0].shape[0])

    def _as_labelled(self, spo) -> SPO:
        """Whatever the engine produced -> {label: coeff}, the shared shape.

        The array engines are converted back here rather than kept native
        because every consumer -- the JSON target cache, the trainer's own
        string->key setup, the verification scripts -- speaks labels. That
        round trip is real waste, but it is a fixed cost per snapshot against
        a propagation that dominates it, and removing it means changing the
        cache format (see docs/issues/03-engine-performance.md, "타겟 캐시 형식").
        """
        if isinstance(spo, dict):
            return spo
        if self.kernel == 'packed':
            return sorted_to_labelled_spo(spo[0], spo[1], self.num_qubits,
                                          xp=self.xp)
        return from_wide_arrays(spo[0], spo[1], self.num_qubits, xp=self.xp)

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

            evolved = self._advance(self._seed_spo(label), gate_seq, delta,
                                    stats)
            n_terms = self._size(evolved)
            targets[key] = self._as_labelled(evolved)

            if verbose and (idx + 1) % max(1, len(obs_list) // 10) == 0:
                print(f"    [{idx+1}/{len(obs_list)}] {key}: "
                      f"{n_terms} Pauli terms")

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
            running[f"{pauli}_{q}"] = self._seed_spo(label)

        stats = TruncationStats()
        series: Dict[int, Dict[str, SPO]] = {}
        t_start = time.time()

        for k in range(1, k_max + 1):
            for key in running:
                running[key] = self._advance(running[key], block, delta, stats)

            if k in snapshots:
                sizes = [self._size(spo) for spo in running.values()]
                series[k] = {key: self._as_labelled(spo)
                             for key, spo in running.items()}
                if verbose:
                    total = sum(sizes)
                    largest = max(sizes)
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
