#!/usr/bin/env python3
"""Two diagnostic figures built purely from scripts/03_statevector_pilot.py and
scripts/03b_trotter_baseline.py's output -- no BP-PPS training or target
generation involved, since both pilots are exact-diagonalization comparisons.

  1. goal1_hva_vs_trotter.png: for each T, infidelity vs 2Q-gate count for
     the HVA ceiling (direct statevector optimum, no truncation) and for S2
     Trotter at the same time steps -- the comparison
     docs/issues/01-scale-plan.md's "no scaling advantage observed" note was
     missing (it only looked at HVA's absolute fidelity vs T, never at
     Trotter's fidelity at a matched gate count).
  2. goal3_energy_vs_2q.png: WITHDRAWN 2026-09-03 -- it duplicated panel (a)
     of report_goal3_layers.png exactly. See plot_goal3's docstring.

Both figures in this file are now withdrawn. It is kept for the two docstrings,
which record why each comparison was wrong or redundant.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import load_config, output_dir  # noqa: E402

RESULTS_DIR = output_dir(load_config(), create=False)
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, 'statevector_pilot.json')) as f:
    pilot = json.load(f)

N_BONDS = 24  # 4x4 lattice, fixed


def plot_goal1():
    """WITHDRAWN 2026-08-31 -- do not re-enable without reading this.

    This figure plotted the statevector pilot's "ceiling" against Trotter on a
    common 2Q axis, and the 290x/63x/27x advantage read off it is retracted.
    The pilot optimises fidelity for |0...0> alone, and once its angles were
    saved and re-scored on random product states the L=3 optimum came out at
    0.43 average infidelity -- an interpolant of one state, not an
    approximation of exp(-iHT). It is not an upper bound on BP-PPS, which
    approximates the operator, so the two curves were never comparable.
    See docs/issues/01-scale-plan.md, "pilot ceiling 은 시간진화 회로가
    아닙니다", and scripts/03g_state_averaged.py for the metric that replaces
    it. Goal 1 still holds on the state-averaged metric, but it has to be
    redrawn there; plot_goal3 below is unaffected because energy is not a
    single-state quantity.
    """
    print("plot_goal1: WITHDRAWN -- the pilot ceiling is not a time-evolution "
          "circuit (0.43 avg infidelity). Redraw on the state-averaged metric; "
          "see this function's docstring.")
    return None

    te = pilot['time_evolution']
    tr = pilot.get('trotter', {})
    T_list = sorted(te.keys(), key=float)

    fig, axes = plt.subplots(1, len(T_list), figsize=(4.2 * len(T_list), 4.2),
                             sharey=True)
    if len(T_list) == 1:
        axes = [axes]

    for ax, T in zip(axes, T_list):
        hva = te[T]
        L_list = sorted(hva.keys(), key=int)
        x_hva = [N_BONDS * int(L) for L in L_list]
        y_hva = [max(1e-9, 1.0 - hva[L]["fidelity"]) for L in L_list]
        ax.semilogy(x_hva, y_hva, 'o-', color='#E91E63', linewidth=2,
                    markersize=7, label='HVA ceiling (statevector-exact)')

        # The grouped builders (src/bppps/propagation.py) are the honest
        # baseline: qiskit's generic SuzukiTrotter emits every RZZ twice per
        # step, so quoting it would hand HVA a free factor of 2 on this axis.
        for key, colour, marker, label in (
                ('trotter_grouped_s2', '#2196F3', 's', r'Trotter $S_2$ (grouped)'),
                ('trotter_grouped_s4', '#FF9800', '^', r'Trotter $S_4$ (grouped)')):
            block = pilot.get(key, {}).get(T)
            if not block:
                continue
            pts = sorted(block.values(), key=lambda d: d['n_2q'])
            ax.semilogy([p['n_2q'] for p in pts],
                        [max(1e-9, 1.0 - p["fidelity"]) for p in pts],
                        marker + '--', color=colour, linewidth=2,
                        markersize=6, label=label)

        ax.axhline(0.01, color='gray', linestyle=':', linewidth=1,
                   label='accuracy target (F=0.99)' if T == T_list[0] else None)
        ax.set_title(f'T = {T}')
        ax.set_xlabel('2Q gate count')
        ax.grid(True, which='both', alpha=0.3)

    axes[0].set_ylabel('Infidelity  $1 - F$  (log scale)')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Goal 1: HVA ceiling vs grouped Trotter, on a common 2Q-gate axis '
                '(4x4, exact statevector, no BP-PPS truncation)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(PLOT_DIR, 'goal1_hva_vs_trotter.png')
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
    return path


def plot_goal3():
    """WITHDRAWN 2026-09-03 -- superseded, do not re-enable.

    This drew goal3_energy_vs_2q.png: the ground-state energy gap against 2Q
    gate count, statevector ceiling plus the one real BP-PPS point at L=3.
    Every element of it is panel (a) of report_goal3_layers.png
    (scripts/plot_report_4x4.py), which draws the same ceiling from the same
    statevector_pilot.json, the same L=3 BP-PPS marker, and additionally the
    aborted L=5 point and the fidelity panel. Two files drawing one figure only
    creates a chance for them to disagree after the next retrain.

    Use scripts/plot_report_4x4.py::fig_goal3.
    """
    raise SystemExit(
        'goal3_energy_vs_2q.png was withdrawn on 2026-09-03: it duplicates '
        'report_goal3_layers.png panel (a). Run scripts/plot_report_4x4.py '
        'instead.')


if __name__ == '__main__':
    # Both figures in this file are withdrawn; each raises with the reason.
    plot_goal1()
    plot_goal3()
