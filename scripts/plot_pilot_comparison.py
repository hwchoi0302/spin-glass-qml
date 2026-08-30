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
  2. goal3_energy_vs_2q.png: ground-state energy gap vs 2Q-gate count -- the
     item docs/issues/01-scale-plan.md:938 asks for. Only the statevector
     ceiling (L=1..8, exact) and the one validated real BP-PPS point (L=3,
     post truncation-fix) are plotted; L=5's real retrain is still running.
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
    gs = pilot['ground_state']
    L_list = sorted(gs.keys(), key=int)
    x_ceiling = [N_BONDS * int(L) for L in L_list]
    y_ceiling = [max(1e-6, gs[L]['gap']) for L in L_list]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogy(x_ceiling, y_ceiling, 'o-', color='#4CAF50', linewidth=2,
               markersize=7, label='statevector ceiling (exact, no truncation)')

    # The one validated real BP-PPS point (post truncation-fix, L=3).
    real_path = os.path.join(RESULTS_DIR, 'gs_trained_params.json')
    if os.path.exists(real_path):
        with open(real_path) as f:
            real = json.load(f)
        if real.get('initial_state') == 'plus':
            L_real = real['n_layers']
            # E0 = ceiling energy - ceiling gap, same E0 for every L.
            e0 = next((gs[L]['energy'] - gs[L]['gap'] for L in L_list
                      if int(L) == L_real), None)
            if e0 is not None:
                gap_real = abs(real['final_loss'] - e0)
                ax.plot(N_BONDS * L_real, max(1e-6, gap_real), 'D',
                       color='#E91E63', markersize=11,
                       label=f'real BP-PPS training (L={L_real}, post-fix)')

    ax.set_xlabel('2Q gate count')
    ax.set_ylabel(r'Energy gap  $E - E_0$  (log scale)')
    ax.set_title('Goal 3: ground-state energy gap vs 2Q-gate count (4x4)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, 'goal3_energy_vs_2q.png')
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
    return path


if __name__ == '__main__':
    plot_goal1()
    plot_goal3()
