#!/usr/bin/env python3
"""4×4 Spin Glass — Comprehensive Visualization.

Generates publication-quality plots:
  1. Lattice diagram with J couplings and h fields
  2. Time-evolution fidelity comparison (ED, Trotter S2/S4, HVA)
  4. Training loss curves
  6. Circuit cost (depth, 2Q count) and the accuracy it buys

Figures 3, 5, 7 and 8 were removed on 2026-09-03. 3 (per-site <X>/<Z> at
t=0.5), 5 (ground-state observables) and 7 (per-qubit loss heatmap) carried no
finding the scalar summaries do not already state, and 8 was a montage of the
other panels. Figure 9 went at the same time, from plot_extended.py: its (a)
and (b) were figure 2's two panels redrawn, and its (c) plotted infidelity
against the withdrawn qiskit depth numbers, annotated with the 4.7x
"compression" that docs/issues/05-writing.md forbids quoting.
"""

import sys
import os
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from config import load_config, output_dir, resolve_params_path  # noqa: E402

CONFIG = load_config()
RESULTS_DIR = output_dir(CONFIG, create=False)
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Load all data
with open(os.path.join(RESULTS_DIR, 'model_config.json')) as f:
    config = json.load(f)
with open(os.path.join(RESULTS_DIR, 'ed_results.json')) as f:
    ed = json.load(f)
_N_LAYERS = CONFIG['ansatz']['n_layers']
_te_path = resolve_params_path(RESULTS_DIR, 'te', _N_LAYERS)
if _te_path is None:
    raise SystemExit(f"no time-evolution parameters at n_layers={_N_LAYERS} "
                     f"in {RESULTS_DIR}")
_gs_path = resolve_params_path(RESULTS_DIR, 'gs', _N_LAYERS)
if _gs_path is None:
    raise SystemExit(f"no ground-state parameters at n_layers={_N_LAYERS} "
                     f"in {RESULTS_DIR}")
with open(_te_path) as f:
    te_data = json.load(f)
with open(_gs_path) as f:
    gs_data = json.load(f)
with open(os.path.join(RESULTS_DIR, 'validation_results.json')) as f:
    val = json.load(f)

# Composition fidelities of U(theta; 0.5)^k, produced by plot_extended.py --part 1.
# Optional: absent on a fresh run that has not built it yet.
_comp_path = os.path.join(RESULTS_DIR, 'composition_fidelity.json')
comp = None
if os.path.exists(_comp_path):
    with open(_comp_path) as f:
        comp = json.load(f)
else:
    print("[warn] composition_fidelity.json missing - "
          "HVA curve will show the t=0.5 point only. "
          "Run: python scripts/plot_extended.py --part 1")

# Global style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

LX, LY = config['Lx'], config['Ly']
J = config['J']
bonds = config['bonds']
h = config['h']


# ============================================================================
# Grouped Trotter reference circuits
# ============================================================================
# Every Trotter number in this file is produced here, by the repository's own
# grouped builder, and none of them is read out of trotter_results.json any
# more. That file holds qiskit's generic Suzuki synthesis, which symmetrises
# term by term even though all the ZZ terms commute, so it emits 2*n_bonds RZZ
# per step where the grouped circuit emits n_bonds. Checked on 2026-09-03 at
# t=0.5, dt=0.1, 5 steps:
#
#     qiskit   F = 0.99983542   240 2Q
#     grouped  F = 0.99975960   120 2Q
#
# A hair more fidelity for exactly twice the gates is strictly worse per gate,
# so the qiskit circuit is a bad compilation of the formula rather than a
# better baseline. Figure 6 already counted the grouped way; figure 2 did not,
# and the two figures disagreed about what "Trotter" meant.
#
# `dt` is an UPPER BOUND on the step size, never a divisor of t. num_steps
# rounds ceil(t/dt) and the step then shrinks to t/steps so the circuit lands
# on t exactly (see TrotterCircuit.num_steps). Consequences worth knowing
# before reading either figure:
#
#     t=0.5, "dt=0.2"  ->  3 steps of 0.1667, not 2 steps of 0.2
#     t=0.3, "dt=0.2"  ->  2 steps of 0.1500
#     t=0.1, "dt=0.2"  ->  1 step of 0.1, i.e. the same circuit as "dt=0.1"
#
# So the labels here read "dt <= x" and every curve carries its step count.

# One S4 step is five S2 sub-steps, S4(dt) = S2(p dt)^2 S2((1-4p) dt) S2(p dt)^2.
_S2_SUBSTEPS = {2: 1, 4: 5}

_tv_ctx = None


def _trotter_ctx():
    """Model, exact solver and |0...0> for the grouped Trotter evaluations."""
    global _tv_ctx
    if _tv_ctx is None:
        from qiskit.quantum_info import Statevector
        from hamiltonians import SpinGlass2D
        from classical_bench import ExactDiag

        model = SpinGlass2D.from_config_dict(config)
        n = model.num_qubits
        _tv_ctx = {
            'model': model,
            'n': n,
            'n_bonds': model.num_bonds,
            'ed': ExactDiag(model.build_sparse_matrix(), n),
            'psi0': np.array(Statevector.from_label('0' * n), dtype=complex),
            'substep_bonds': {int(k): [tuple(x) for x in v]
                              for k, v in config['substep_bonds'].items()},
            'J': np.array(model.J),
            'h': model.h,
        }
    return _tv_ctx


def trotter_steps(t, dt):
    """Steps used to reach t with step size at most dt (rounds up)."""
    return max(1, math.ceil(t / dt - 1e-12))


def grouped_cost(n_steps, order, n_bonds):
    """(depth, n_2q) of a grouped Suzuki circuit, counted analytically.

    Counted the same way the HVA is counted: from the 4-colour schedule, before
    any transpiler sees the circuit. One S2 sub-step is
    RX(tau/2) . [4 RZZ colour layers] . RX(tau/2), and the trailing RX of one
    sub-step fuses with the leading RX of the next, so m sub-steps cost 5m+1
    layers and n_bonds*m two-qubit gates.
    """
    sub = _S2_SUBSTEPS[order] * n_steps
    return 5 * sub + 1, n_bonds * sub


_fid_cache = {}


def grouped_fidelity(t, n_steps, order):
    """Statevector fidelity of the grouped Suzuki circuit against exact e^{-iHt}.

    Cached: figures 2 and 6 ask for overlapping (t, steps, order) triples and
    the S4 circuits run to a few thousand gates.
    """
    key = (round(t, 12), n_steps, order)
    if key in _fid_cache:
        return _fid_cache[key]

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from bppps.propagation import build_trotter_gate_sequence

    c = _trotter_ctx()
    seq = build_trotter_gate_sequence(c['n'], c['substep_bonds'], c['J'], c['h'],
                                      dt=t / n_steps, n_steps=n_steps, order=order)
    qc = QuantumCircuit(c['n'])
    for g in seq:
        if g[0] == 'rx':
            qc.rx(g[2], g[1])
        else:
            qc.rzz(g[3], g[1], g[2])
    psi = np.array(Statevector(c['psi0']).evolve(qc))
    exact = c['ed'].time_evolve(c['psi0'].copy(), t)
    f = float(np.abs(np.vdot(exact, psi)) ** 2)
    _fid_cache[key] = f
    return f


# ============================================================================
# Figure 1: Lattice with J couplings and h fields
# ============================================================================
def plot_lattice():
    """Visualize 4×4 lattice with J couplings (bond colors) and h field."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_aspect('equal')

    # Draw bonds with colors
    for idx, (i, j) in enumerate(bonds):
        xi, yi = i % LX, i // LX
        xj, yj = j % LX, j // LX
        j_val = J[idx]
        color = '#2196F3' if j_val > 0 else '#F44336'  # Blue=FM, Red=AFM
        lw = 3.0
        ax.plot([xi, xj], [LY-1-yi, LY-1-yj], color=color, linewidth=lw, zorder=1)

        # J label on bond
        mx, my = (xi+xj)/2, (LY-1-yi+LY-1-yj)/2
        offset_x = 0.12 if yi == yj else 0.0
        offset_y = 0.0 if yi == yj else 0.12
        ax.text(mx + offset_x, my + offset_y, f'{j_val:+.0f}',
                fontsize=9, fontweight='bold', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor=color, alpha=0.9))

    # Draw qubit nodes
    for q in range(LX * LY):
        x, y = q % LX, q // LX
        circle = plt.Circle((x, LY-1-y), 0.22, facecolor='#FFFFFF',
                           edgecolor='#333333', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, LY-1-y, f'{q}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='#333333', zorder=3)

    # h field arrows
    for q in range(LX * LY):
        x, y = q % LX, q // LX
        ax.annotate('', xy=(x + 0.35, LY-1-y + 0.15),
                   xytext=(x + 0.35, LY-1-y - 0.15),
                   arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))

    # Legend
    fm_line = mpatches.Patch(color='#2196F3', label=r'$J_{ij}=+1$ (FM)')
    afm_line = mpatches.Patch(color='#F44336', label=r'$J_{ij}=-1$ (AFM)')
    h_line = mpatches.Patch(color='#FF9800', label=r'$h=1.0$ (transverse field)')
    ax.legend(handles=[fm_line, afm_line, h_line], loc='upper right',
             fontsize=11, framealpha=0.9)

    ax.set_xlim(-0.6, LX - 0.3)
    ax.set_ylim(-0.6, LY - 0.3)
    ax.set_title(r'4×4 EA Bimodal Spin Glass ($H = -\sum J_{ij} Z_i Z_j - h\sum X_i$)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')

    plt.savefig(os.path.join(PLOT_DIR, '01_lattice_J_h.png'))
    plt.close()
    print("✅ 01_lattice_J_h.png")


# ============================================================================
# Figure 2: Time-evolution Fidelity comparison
# ============================================================================
def plot_fidelity():
    """Fidelity vs time for the HVA and three grouped Trotter baselines.

    The HVA is trained on a single chunk Delta t = 0.5; longer times are reached
    by composing that chunk k times, U(theta; 0.5)^k, and those composed
    fidelities live in composition_fidelity.json (plot_extended.py --part 1).
    So the HVA only exists on t = 0.5 ... 2.5, and every curve here is drawn on
    that same grid. The figure used to put the Trotter curves on a second,
    shorter grid (t = 0.1 ... 1.0) as well, which made the left half of the
    plot a comparison against nothing.

    Trotter is evaluated by grouped_fidelity(), not read from
    trotter_results.json -- see the "Grouped Trotter reference circuits" block
    for why the two differ by a factor of two in gate count.

    S4 is on the figure because it is the honest ceiling of the Trotter family,
    but it is close enough to exact here (1 - F ~ 1e-7) that a bare curve reads
    as "S4 wins". It does not: it costs five S2 sub-steps per step. The 2Q
    count of every curve is therefore annotated at its right-hand end, and the
    iso-cost comparison lives in figure 6(c).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_bonds = _trotter_ctx()['n_bonds']
    n_layers = config.get('n_layers', 3)

    if comp is not None:
        time_pts = list(comp['time_pts'])
        hva_fid = list(comp['hva_fid'])
        ks = list(comp['k_values'])
    else:
        time_pts = [0.5]
        hva_fid = [val['time_evolution']['fidelity']]
        ks = [1]
    hva_2q = [n_bonds * n_layers * k for k in ks]

    # (label, colour, marker, fidelities, 2Q counts)
    series = [(r'HVA %d-layer, $U(\theta;0.5)^k$' % n_layers,
               '#E91E63', 'D', hva_fid, hva_2q)]
    for order, dt, color, mk in ((2, 0.1, '#2196F3', 's'),
                                 (2, 0.2, '#FF9800', '^'),
                                 (4, 0.2, '#4CAF50', 'v')):
        f, q = [], []
        for t in time_pts:
            steps = trotter_steps(t, dt)
            f.append(grouped_fidelity(t, steps, order))
            q.append(grouped_cost(steps, order, n_bonds)[1])
        series.append((r'Trotter $S_%d$ grouped ($\Delta t \leq %.1f$)' % (order, dt),
                       color, mk, f, q))

    # The 2Q cost rides in the legend rather than at the end of each curve. The
    # S4 line sits at 1 - F ~ 1e-7 and the HVA and S2 lines end within a factor
    # of two of each other, so end-of-line labels either collide or float far
    # from the curve they belong to; and the cost is the first thing to read
    # about the S4 curve, not a footnote to it.
    cost_lab = [f'{lab}  [{q[-1]} 2Q]' for lab, _, _, _, q in series]

    # --- Panel (a): fidelity vs time ---
    for (lab, color, mk, f, _) in series:
        ax1.plot(time_pts, f, mk + '-', color=color, markersize=8, lw=2,
                 zorder=5 if mk == 'D' else 3, label=lab)
    ax1.plot(time_pts[0], hva_fid[0], 'D', color='#E91E63', markersize=13,
             markeredgecolor='black', markeredgewidth=1.2, zorder=6,
             label='HVA trained chunk ($k=1$)')
    # Last, and on top: S4 is flat at 1.0 and would otherwise bury this line.
    ax1.plot([min(time_pts), max(time_pts)], [1.0, 1.0], 'k--', lw=1.4,
             alpha=0.8, zorder=8, label='ED (exact)')

    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Fidelity $|\\langle\\psi_{\\rm approx}|\\psi_{\\rm exact}\\rangle|^2$')
    ax1.set_title('(a) State fidelity vs time')
    ax1.legend(fontsize=9, loc='lower left')
    lo = min(min(f) for _, _, _, f, _ in series)
    ax1.set_ylim(lo - 0.15 * (1 - lo), 1 + 0.03 * (1 - lo))
    ax1.set_xticks(time_pts)
    ax1.grid(True, alpha=0.3)

    # --- Panel (b): infidelity, log scale, with the cost of each curve ---
    for (lab, color, mk, f, q), cl in zip(series, cost_lab):
        ax2.semilogy(time_pts, [1 - v for v in f], mk + '-', color=color,
                     markersize=8, lw=2, zorder=5 if mk == 'D' else 3, label=cl)
    ax2.annotate(f'trained chunk: {1 - hva_fid[0]:.2e}',
                 xy=(time_pts[0], 1 - hva_fid[0]), xytext=(14, -22),
                 textcoords='offset points', fontsize=9, color='#E91E63',
                 arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1))

    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Infidelity $1 - F$')
    ax2.set_title('(b) Infidelity (log scale); brackets give the 2Q cost at $t$=%.1f'
                  % time_pts[-1])
    ax2.set_xticks(time_pts)
    ax2.set_ylim(top=3e-2)
    ax2.legend(fontsize=8.5, loc='center left')
    ax2.grid(True, alpha=0.3, which='both')

    fig.text(0.5, -0.02,
             r'$\Delta t$ is an upper bound, not a divisor of $t$: the circuit uses '
             r'$\lceil t/\Delta t\rceil$ steps and shrinks the step to land on $t$ '
             r'exactly. At $t$=0.5 the "$\Delta t\leq$0.2" circuit is 3 steps of 0.167.',
             ha='center', fontsize=9, color='#555555')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_fidelity_comparison.png'))
    plt.close()
    print("✅ 02_fidelity_comparison.png")
    print(f"   {'t':>5} " + ' '.join(f'{lab[:22]:>24}' for lab, _, _, _, _ in series))
    for i, t in enumerate(time_pts):
        print(f"   {t:5.1f} " + ' '.join(
            f'{1 - f[i]:>11.3e}/{q[i]:<12d}' for _, _, _, f, q in series))


# ============================================================================
# Figure 4: Training loss curves
# ============================================================================
def _split_optimizer_segments(data):
    """Split a recorded loss history into (adam, lbfgsb) and flag probe points.

    ``losses`` is the plain concatenation ``adam_losses + lbfgsb_losses``, but
    the two halves are not the same kind of sequence. Adam records one loss per
    epoch, and every one of those points is a real iterate - the small bumps in
    it are momentum overshoot and belong on the curve. scipy's L-BFGS-B callback
    also fires on the *trial* points of its line search, and a rejected trial can
    sit far above the accepted iterates on either side of it: in the t=0.5 run
    lbfgsb_losses[1] = 0.5428 sits between two points at 0.0153, a factor of 36.
    Plotting the raw concatenation makes the run look like it diverged and
    recovered.

    So probes are only ever looked for inside the L-BFGS-B segment, where they
    actually exist, and a point counts as one when it rises above the running
    minimum by more than 5% of that segment's own span. The threshold is
    relative to the segment rather than absolute so that this works unchanged
    for the compression loss (order 1, positive) and for the ground-state
    energy (order 20, negative). Probes are still drawn, as hollow markers,
    but they are left out of the connecting line.

    Returns ``(segments, n_total)`` where each segment is
    ``(name, kept, probes)`` and ``kept``/``probes`` are ``(index, value)``
    lists carrying the index into the concatenated history.
    """
    PROBE_SPAN_FRACTION = 0.05

    adam = list(data.get('adam_losses') or [])
    lbfgsb = list(data.get('lbfgsb_losses') or [])
    if not adam and not lbfgsb:
        # Older records only carry the merged history; show it as one segment.
        merged = list(data['losses'])
        return [('optimizer', list(enumerate(merged)), [])], len(merged)

    segments = []
    if adam:
        segments.append(('Adam', list(enumerate(adam)), []))
    if lbfgsb:
        offset = len(adam)
        span = max(lbfgsb) - min(lbfgsb)
        tol = span * PROBE_SPAN_FRACTION
        kept, probes = [], []
        best = float('inf')
        for i, v in enumerate(lbfgsb):
            if v > best + tol:
                probes.append((offset + i, v))
            else:
                kept.append((offset + i, v))
                best = min(best, v)
        segments.append(('L-BFGS-B', kept, probes))
    return segments, len(adam) + len(lbfgsb)


def plot_training():
    """Plot training loss curves for time-evolution and ground state."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Time-evolution loss ---
    segments, n_total = _split_optimizer_segments(te_data)
    colors = {'Adam': '#E91E63', 'L-BFGS-B': '#3F51B5', 'optimizer': '#E91E63'}
    shades = {'Adam': '#E91E63', 'L-BFGS-B': '#3F51B5', 'optimizer': 'none'}

    x0 = 0
    for name, kept, probes in segments:
        n_seg = len(kept) + len(probes)
        if len(segments) > 1:
            ax1.axvspan(x0 + 0.5, x0 + n_seg + 0.5, color=shades[name],
                        alpha=0.05, zorder=0)
            ax1.text(x0 + n_seg / 2, 1.2, name, ha='center', fontsize=11,
                     color=colors[name], fontweight='bold')
        if kept:
            xs = [i + 1 for i, _ in kept]
            ys = [v for _, v in kept]
            ax1.semilogy(xs, ys, '-', color=colors[name], lw=2,
                         label=f'{name} (accepted)' if len(segments) > 1 else None)
        if probes:
            ax1.semilogy([i + 1 for i, _ in probes], [v for _, v in probes],
                         'o', mfc='none', mec='#9E9E9E', ms=6, lw=0,
                         label='line-search trial' if name == 'L-BFGS-B' else None)
        x0 += n_seg

    if len(segments) > 1:
        boundary = len(segments[0][1]) + len(segments[0][2]) + 0.5
        ax1.axvline(boundary, color='#616161', ls=':', lw=1.5)

    final = te_data['losses'][-1]
    ax1.axhline(y=final, color='gray', ls='--', alpha=0.5)
    ax1.set_xlabel('Optimizer step')
    ax1.set_ylabel(r'$\mathcal{L}_{X,Z}$')
    ax1.set_title(r'(a) Time-Evolution Compression Loss ($\Delta t = 0.5$)')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.text(n_total * 0.58, final * 1.35, f'Final: {final:.4f}',
             fontsize=11, color='gray')
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(fontsize=9, loc='lower left')

    # --- Ground state energy ---
    # Same two-optimizer structure as panel (a); here the loss is the energy.
    gs_losses = gs_data['losses']
    gs_segments, _ = _split_optimizer_segments(gs_data)
    gs_colors = {'Adam': '#9C27B0', 'L-BFGS-B': '#00897B', 'optimizer': '#9C27B0'}

    x0 = 0
    for name, kept, probes in gs_segments:
        n_seg = len(kept) + len(probes)
        if len(gs_segments) > 1:
            ax2.axvspan(x0 + 0.5, x0 + n_seg + 0.5, color=gs_colors[name],
                        alpha=0.05, zorder=0)
        if kept:
            ax2.plot([i + 1 for i, _ in kept], [v for _, v in kept], '-',
                     color=gs_colors[name], lw=2,
                     label=f'BP-PPS energy ({name})' if len(gs_segments) > 1
                     else 'BP-PPS energy')
        if probes:
            ax2.plot([i + 1 for i, _ in probes], [v for _, v in probes], 'o',
                     mfc='none', mec='#9E9E9E', ms=6, lw=0,
                     label='line-search trial' if name == 'L-BFGS-B' else None)
        x0 += n_seg
    if len(gs_segments) > 1:
        ax2.axvline(len(gs_segments[0][1]) + len(gs_segments[0][2]) + 0.5,
                    color='#616161', ls=':', lw=1.5)

    # --- The aborted L=5 run, overlaid ---
    # The 41 h L=5 attempt died inside L-BFGS-B, and run_pipeline.py only writes
    # gs_trained_params.json after both stages, so its angles are gone and it can
    # never be plotted as a circuit. Its energies survive: 06a07ae recovered them
    # from the run log into gs_L5_aborted.json. They belong on this panel because
    # they answer the question the L=3 curve raises -- whether the 0.47 gap is
    # the optimiser or the ansatz -- and the answer is the ansatz: five layers
    # reach 0.28 with Adam alone, before any refinement.
    # Drawn dashed and marker-only: the log is 11 sampled epochs out of 100, not
    # a continuous history, and the run is not a deployable result.
    _l5_path = os.path.join(RESULTS_DIR, 'gs_L5_aborted.json')
    if os.path.exists(_l5_path):
        with open(_l5_path) as f:
            l5 = json.load(f)
        ep = [e['epoch'] for e in l5['epoch_log']]
        en = [e['loss'] for e in l5['epoch_log']]
        ax2.plot(ep, en, 'o--', color='#F57C00', ms=5, lw=1.6, alpha=0.9,
                 zorder=4,
                 label='L=5, Adam only (ABORTED in L-BFGS-B)')
        ax2.plot(ep[-1], en[-1], '*', color='#F57C00', ms=16,
                 markeredgecolor='black', markeredgewidth=0.8, zorder=7)
        ax2.annotate('L=5 after Adam: %.3f\ngap %.4f  (L=3: %.4f)'
                     % (l5['final_adam_loss'], l5['gap_to_E0'],
                        l5['reference_L3_production']['gap_to_E0']),
                     xy=(ep[-1], en[-1]), xytext=(-10, 26),
                     textcoords='offset points', fontsize=9, color='#E65100',
                     ha='right',
                     arrowprops=dict(arrowstyle='->', color='#F57C00', lw=1))

    ax2.axhline(y=ed['ground_energy'], color='#333333', ls='--', lw=2,
               label=f'ED ground (E₀ = {ed["ground_energy"]:.2f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Energy $E(\\theta)$')
    ax2.set_title('(b) Ground-state training: L=3 production, L=5 aborted')
    ax2.legend(fontsize=8.5, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.text(len(gs_losses)*0.42, gs_losses[-1] + 0.9,
            f'L=3 final: {gs_losses[-1]:.2f}\ngap: {gs_losses[-1] - ed["ground_energy"]:.4f}',
            fontsize=9, color='#9C27B0')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_training_curves.png'))
    plt.close()
    print("✅ 04_training_curves.png")


# ============================================================================
# Figure 6: Circuit depth & compression summary
# ============================================================================
def plot_depth_comparison():
    """What the circuit costs, and what that cost buys.

    (a) and (b) are cost against time; (c) is accuracy against cost at a single
    time. Splitting it that way is deliberate. (c) used to plot infidelity
    against time, which made it a redrawing of figure 2(b) -- the same
    redundancy that got figure 9 deleted -- and, worse, it invited the reader to
    compare curves vertically at fixed t even though the curves sit at wildly
    different gate counts there. The comparison that carries the goal-1 claim is
    at *matched* cost, so (c) fixes t = 0.5 and sweeps the step count instead.

    All costs are counted analytically from the 4-colour schedule, on both
    sides, before transpilation. The previous version of this figure read
    qiskit's ``depth`` and ``n_2q_gates`` out of trotter_results.json -- an
    unsorted ``qc.depth()`` and a doubled RZZ count -- against the HVA's
    analytic numbers, and reported a 4.7x depth and 3.3x gate "compression"
    that was almost entirely the two sides being counted differently. See
    scripts/03f_depth_audit.py and docs/issues/05-writing.md.

    S4 is included from 2026-09-03. It is four orders of magnitude more
    accurate than S2 here and costs five times the gates, which is exactly why
    it belongs on a figure that shows cost on two of its three panels.
    """
    n_layers = config.get('n_layers', 3)
    n_bonds = _trotter_ctx()['n_bonds']

    # HVA only exists where a composed circuit exists: k blocks of delta_t.
    if comp is not None:
        time_pts = list(comp['time_pts'])
        fid_hva = list(comp['hva_fid'])
        ks = list(comp['k_values'])
    else:
        time_pts, fid_hva, ks = [0.5], [val['time_evolution']['fidelity']], [1]
    depth_hva = [5 * n_layers * k for k in ks]
    n2q_hva = [n_bonds * n_layers * k for k in ks]

    # (label, colour, marker, depths, 2Q counts, fidelities)
    series = [('HVA %d-layer (composed)' % n_layers, '#E91E63', 'D',
               depth_hva, n2q_hva, fid_hva)]
    for order, dt, color, mk in ((2, 0.1, '#2196F3', 's'),
                                 (2, 0.2, '#FF9800', '^'),
                                 (4, 0.2, '#4CAF50', 'v')):
        d, q, f = [], [], []
        for t in time_pts:
            steps = trotter_steps(t, dt)
            dep, n2q = grouped_cost(steps, order, n_bonds)
            d.append(dep)
            q.append(n2q)
            f.append(grouped_fidelity(t, steps, order))
        series.append((r'Trotter $S_%d$ grouped ($\Delta t \leq %.1f$)' % (order, dt),
                       color, mk, d, q, f))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.2))

    for lab, color, mk, d, q, f in series:
        ax1.semilogy(time_pts, d, mk + '-', color=color, markersize=8, lw=2,
                     label=lab)
        ax2.semilogy(time_pts, q, mk + '-', color=color, markersize=8, lw=2,
                     label=lab)

    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Circuit depth (analytic, 4-colour schedule)')
    ax1.set_title('(a) Circuit depth')
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Number of 2-qubit gates')
    ax2.set_title('(b) 2Q gate count')
    for ax in (ax1, ax2):
        ax.set_xticks(time_pts)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3, which='both')

    # The half of the story the pre-2026-08-31 figure hid: composing the HVA
    # block costs a fixed 144 2Q per unit t, and once t > 0.5 the dt<=0.2
    # Trotter circuit is cheaper than the composition on both cost axes. Placed
    # as a bare label on the curve -- an arrow from the corner would have to
    # cross the S4 and dt<=0.1 lines to get here.
    ax2.annotate(r'$\Delta t\leq$0.2 Trotter is cheaper than the'
                 '\ncomposed HVA for every $t>$0.5',
                 xy=(0.03, 0.97), xycoords='axes fraction', fontsize=9,
                 color='#B26A00', ha='left', va='top')

    # Both circuits split the same 24 bonds into the same 4 colours, so depth is
    # 5 x (2Q / n_bonds) for all four series and (a) is (b) rescaled. That is a
    # result, not a redundancy -- it is why the depth axis is not an independent
    # advantage -- so it is stated rather than left to be noticed.
    ax1.annotate('depth $\\approx 5\\times$(2Q$/%d$) for all four:\n'
                 'the depth axis is (b) rescaled' % n_bonds,
                 xy=(0.04, 0.94), xycoords='axes fraction', fontsize=9,
                 color='#444444', ha='left', va='top')

    # --- (c) accuracy at matched cost, at one time ---
    T_ISO = time_pts[0]
    ax3.set_xlabel('2Q gate count at $t$ = %.1f' % T_ISO)
    ax3.set_ylabel('Infidelity $1-F$ vs ED')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.set_title('(c) Accuracy at matched cost, $t$ = %.1f' % T_ISO)
    # Explicit ticks at the gate counts that actually occur: the default log
    # locator packs 3x10^1 / 4x10^1 / 6x10^1 on top of each other here.
    ax3.set_xticks([24, 48, 96, 192, 384, 768, 1536])
    ax3.set_xticklabels(['24', '48', '96', '192', '384', '768', '1536'])
    ax3.xaxis.set_minor_formatter(plt.NullFormatter())

    for order, color, mk, steps_list in ((2, '#2196F3', 's', (1, 2, 3, 4, 6, 8, 12)),
                                         (4, '#4CAF50', 'v', (1, 2, 3, 4))):
        xs, ys = [], []
        for s in steps_list:
            xs.append(grouped_cost(s, order, n_bonds)[1])
            ys.append(1 - grouped_fidelity(T_ISO, s, order))
        ax3.plot(xs, ys, mk + '-', color=color, markersize=7, lw=2,
                 label=r'Trotter $S_%d$ grouped, 1..%d steps' % (order, steps_list[-1]))

    # BP-PPS blocks at the same time: L=3 is the production run, L=2 is the
    # cheaper block measured in te_trained_params_L2.json.
    hva_pts = [(n_bonds * n_layers, 1 - fid_hva[0], 'L=%d' % n_layers)]
    _l2_path = os.path.join(RESULTS_DIR, 'te_trained_params_L2.json')
    if os.path.exists(_l2_path):
        with open(_l2_path) as f:
            l2 = json.load(f)
        hva_pts.append((n_bonds * 2, 1 - l2['composition'][0]['fidelity'], 'L=2'))
    ax3.plot([p[0] for p in hva_pts], [p[1] for p in hva_pts], 'D',
             color='#E91E63', markersize=11, markeredgecolor='black',
             markeredgewidth=1.0, lw=0, zorder=6, label='BP-PPS HVA block')
    for x, y, lab in hva_pts:
        ax3.annotate(lab, xy=(x, y), xytext=(-14, -6),
                     textcoords='offset points', fontsize=9, color='#E91E63',
                     fontweight='bold', ha='right', va='center')

    # The one exactly iso-cost pair on the figure: 72 2Q buys either the L=3
    # BP-PPS block or three grouped S2 steps.
    iso_2q = n_bonds * n_layers
    iso_s2 = 1 - grouped_fidelity(T_ISO, iso_2q // n_bonds, 2)
    ax3.annotate('%d 2Q buys either:\nBP-PPS  %.2e\ngrouped $S_2$  %.2e\n(%.1fx)'
                 % (iso_2q, 1 - fid_hva[0], iso_s2, iso_s2 / (1 - fid_hva[0])),
                 xy=(iso_2q, iso_s2), xytext=(0.03, 0.30),
                 textcoords='axes fraction', fontsize=9, color='#333333',
                 ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#BBBBBB',
                           lw=0.8, alpha=0.92),
                 arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2,
                                 connectionstyle='arc3,rad=0.25'))
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3, which='both')

    fig.suptitle(r'HVA vs grouped Trotter, both counted analytically before '
                 r'transpilation.  $\Delta t$ is an upper bound: the circuit '
                 r'uses $\lceil t/\Delta t\rceil$ steps and shrinks the step to '
                 r'land on $t$ exactly.', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(PLOT_DIR, '06_depth_comparison.png'))
    plt.close()

    print("✅ 06_depth_comparison.png")
    head = ' '.join(f'{lab[:20]:>22}' for lab, _, _, _, _, _ in series)
    print(f"   {'t':>5} {'steps':>5} " + head)
    for i, t in enumerate(time_pts):
        row = ' '.join(f'{d[i]:>7d}d/{q[i]:<6d}2Q' for _, _, _, d, q, _ in series)
        print(f"   {t:5.1f} {'':>5} " + row)


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating plots...\n")
    plot_lattice()
    plot_fidelity()
    plot_training()
    plot_depth_comparison()
    print(f"\nAll plots saved to: {PLOT_DIR}/")
