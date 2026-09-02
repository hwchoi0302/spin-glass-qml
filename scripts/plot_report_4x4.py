"""Figures for the 4x4 progress report.

Three figures that the existing plot scripts do not cover, each built only from
committed JSON in ``results/4x4/``:

  report_goal1_deployable.png  the goal-1 comparison on the *deployable*
                               circuit (composed BP-PPS blocks), not the
                               statevector ceiling
  report_goal2_hardness.png    anticoncentration, entanglement, and the
                               hardware-total trade-off between L=2 and L=3
  report_goal3_layers.png      ground-state layer sweep with the measured
                               BP-PPS point on top of the ceiling

Run:  .venv/bin/python scripts/plot_report_4x4.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results', '4x4')
OUT = os.path.join(RES, 'plots')

# 4x4: 24 bonds, so one HVA layer costs 24 RZZ. 10x10: 180 bonds.
BONDS_4X4 = 24
BONDS_10X10 = 180

C_HVA = '#d81b60'      # BP-PPS / HVA, deployable
C_CEIL = '#8e24aa'     # statevector ceiling
C_TROT = '#1e88e5'     # grouped Suzuki-2 Trotter
C_REF = '#607d8b'      # reference lines


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def style(ax):
    ax.grid(True, which='both', alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
# Figure 1 -- goal 1 on the deployable circuit
# --------------------------------------------------------------------------
def fig_goal1():
    pilot = load('statevector_pilot.json')
    comp3 = load('composition_fidelity.json')
    comp2 = load('te_trained_params_L2.json')['composition']

    # deployable = composed BP-PPS blocks. k blocks of L layers -> 24*L*k gates.
    dep3 = {t: (1 - f, BONDS_4X4 * 3 * k)
            for t, f, k in zip(comp3['time_pts'], comp3['hva_fid'], comp3['k_values'])}
    dep2 = {r['t']: (1 - r['fidelity'], BONDS_4X4 * 2 * r['k']) for r in comp2}

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharey=True)
    for ax, T in zip(axes, [0.5, 1.0, 2.0]):
        # The "HVA ceiling (statevector-exact)" curve used to be drawn here and
        # dominated all three panels, bottoming out near 3e-7 at T=0.5. It was
        # withdrawn by a45a88a and removed from this figure on 2026-09-03.
        #
        # The pilot optimised its angles against ONE input state, |0...0>, so
        # what it produced is an interpolant of that state's trajectory, not an
        # approximation of exp(-iHT). Re-scored on the 24 random product states
        # of 03g_state_averaged.py, the same T=0.5 L=3 circuit goes from
        # 1.42e-5 on |0...0> to 4.27e-1 averaged -- a factor of 30000, and an
        # average infidelity of 0.43, i.e. it destroys every state it was not
        # fitted to. BP-PPS degrades by 1.08x on the same test because it
        # approximates the operator.
        #
        # CLAUDE.md words goal 1 as evolution from an *arbitrary* product
        # state, so the ceiling is not a bound on anything this figure claims.
        # Leaving it in made the deployable circuits look 100x worse than a
        # circuit that cannot be deployed at all.

        # grouped Suzuki-2 Trotter, the honest hardware baseline
        tr = pilot['trotter_grouped_s2'][str(T)]
        x = [tr[s]['n_2q'] for s in tr]
        y = [1 - tr[s]['fidelity'] for s in tr]
        o = np.argsort(x)
        ax.plot(np.array(x)[o], np.array(y)[o], '-s', color=C_TROT, lw=1.8,
                ms=5, label='Trotter $S_2$ (grouped)')

        # what we can actually deploy
        for dep, lab, mk in ((dep3, 'BP-PPS block $L$=3, composed', 'o'),
                             (dep2, 'BP-PPS block $L$=2, composed', 'D')):
            if T in dep:
                infid, n2q = dep[T]
                ax.plot([n2q], [infid], mk, color=C_HVA, ms=11, zorder=5,
                        mec='white', mew=1.4, label=lab)

        ax.set_yscale('log')
        ax.set_xlim(0, 400)
        ax.set_title(f'$T$ = {T}', fontsize=13)
        ax.set_xlabel('2Q gate count')
        style(ax)

    axes[0].set_ylabel('Infidelity $1-F$')
    axes[0].legend(fontsize=8.5, loc='lower left', framealpha=0.95)
    fig.suptitle('Goal 1 on the circuit we can actually deploy: composed BP-PPS '
                 'blocks vs grouped Trotter $S_2$ at matched 2Q-gate count (4x4)',
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, 'report_goal1_deployable.png')
    fig.savefig(p, dpi=150, facecolor='white')
    print('wrote', p)


# --------------------------------------------------------------------------
# Figure 2 -- goal 2: anticoncentration, entanglement, hardware total
# --------------------------------------------------------------------------
def fig_goal2():
    sh = load('sampling_hardness.json')
    rows3 = sh['rows']
    rows2 = load('te_trained_params_L2.json')['composition']
    haar_S = sh['haar_entropy_bits']

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # (a) collision probability Z
    ax = axes[0]
    ax.plot([r['k'] for r in rows3], [r['collision_Z'] for r in rows3],
            '-o', color=C_HVA, lw=1.8, ms=6, label='$L$=3 block')
    ax.plot([r['k'] for r in rows2], [r['collision_Z'] for r in rows2],
            '--D', color=C_TROT, lw=1.6, ms=5, label='$L$=2 block')
    ax.axhline(2.0, color=C_REF, ls=':', lw=1.5)
    ax.text(8.6, 2.06, 'Porter-Thomas ($Z$=2)', fontsize=8.5, color=C_REF,
            va='bottom', ha='right')
    ax.annotate('$k$=1: peaked,\nunusable for goal 2',
                xy=(1, 139.5), xytext=(2.6, 60), fontsize=8.5, color='#444',
                arrowprops=dict(arrowstyle='->', color='#888', lw=1))
    ax.set_yscale('log')
    ax.set_xlabel('composition depth $k$   ($t = 0.5k$)')
    ax.set_ylabel('collision  $Z = 2^n \\sum_x p(x)^2$')
    ax.set_title('(a) Anticoncentration', fontsize=12)
    ax.legend(fontsize=9)
    style(ax)

    # (b) half-cut entanglement entropy
    ax = axes[1]
    ax.plot([r['k'] for r in rows3], [r['half_entropy_bits'] for r in rows3],
            '-o', color=C_HVA, lw=1.8, ms=6, label='$L$=3 block')
    ax.plot([r['k'] for r in rows2], [r['half_entropy_bits'] for r in rows2],
            '--D', color=C_TROT, lw=1.6, ms=5, label='$L$=2 block')
    ax.axhline(haar_S, color=C_REF, ls=':', lw=1.5)
    ax.text(12, haar_S + 0.08, f'Haar ({haar_S:.2f} bits)', fontsize=8.5,
            color=C_REF, va='bottom', ha='right')
    ax.axhline(8.0, color=C_REF, ls='--', lw=1, alpha=0.5)
    ax.text(12, 8.02, 'maximum (8 bits)', fontsize=8.5, color=C_REF,
            va='bottom', ha='right')
    ax.set_ylim(0, 8.6)
    ax.set_xlabel('composition depth $k$   ($t = 0.5k$)')
    ax.set_ylabel('half-cut entropy $S$ (bits)')
    ax.set_title('(b) Entanglement growth', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    style(ax)

    # (c) hardware total on 10x10 -- why L=2 wins
    ax = axes[2]
    eps = 1e-3
    for rows, L, color, mk, ls in ((rows3, 3, C_HVA, 'o', '-'),
                                   (rows2, 2, C_TROT, 'D', '--')):
        t = np.array([r['t'] for r in rows])
        f_algo = np.array([r['fidelity'] for r in rows])
        n2q = BONDS_10X10 * L * np.array([r['k'] for r in rows])
        f_hw = np.exp(-n2q * eps)
        ax.plot(t, f_algo * f_hw, ls, color=color, marker=mk, lw=1.8, ms=6,
                label=f'$L$={L} block  (total)')
        ax.plot(t, f_algo, ls, color=color, lw=1, alpha=0.35,
                label=f'$L$={L}  algorithm only')
    ax.axvspan(2.0, 2.6, color='#ffd54f', alpha=0.25)
    ax.text(2.3, 1.7, 'anticoncentrated\n($k\\geq$4)', fontsize=8.5,
            ha='center', va='top', color='#7a5c00')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 3.0)
    ax.set_xlabel('evolution time $t$')
    ax.set_ylabel('fidelity')
    ax.set_title('(c) 10x10 projection, $\\epsilon_{2Q}$ = $10^{-3}$', fontsize=12)
    ax.legend(fontsize=8, loc='lower left')
    style(ax)

    fig.suptitle('Goal 2: the composed circuit does anticoncentrate, and the '
                 'shallower L=2 block wins once hardware error is counted',
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, 'report_goal2_hardness.png')
    fig.savefig(p, dpi=150, facecolor='white')
    print('wrote', p)


# --------------------------------------------------------------------------
# Figure 3 -- goal 3: layer sweep, ceiling vs what BP-PPS reached
# --------------------------------------------------------------------------
def fig_goal3():
    """Ground-state layer sweep: the exact ceiling, and what BP-PPS reached on it.

    The ground-state half of statevector_pilot.json survives the a45a88a
    retraction that killed its time-evolution half. The retraction was about
    overfitting to a single input state; energy is not a single-state quantity,
    so optimising the HVA angles directly on the 2^16 statevector really does
    give the best that ansatz can do at each L.

    Two BP-PPS points are drawn on it, and they say opposite things:

      L=3  gap 0.4729 against a ceiling of 0.4726  -- lands on it, 3e-4
      L=5  gap 0.2796 against a ceiling of 0.0419  -- 6.7x short

    The L=5 point is NOT evidence that BP-PPS cannot reach the L=5 ceiling: the
    run was killed inside L-BFGS-B after 41 h with only Adam completed, so it
    is a lower bound on an unfinished optimisation. It is on the figure because
    leaving it off would let "BP-PPS lands on the ceiling" read as established
    at every depth, and it is established at exactly one.
    """
    gs = load('statevector_pilot.json')['ground_state']
    val = load('validation_results.json')['ground_state']
    l5_path = os.path.join(RES, 'gs_L5_aborted.json')
    l5 = None
    if os.path.exists(l5_path):
        with open(l5_path) as f:
            l5 = json.load(f)

    L = sorted(int(k) for k in gs)
    gap = [gs[str(l)]['gap'] for l in L]
    fid = [gs[str(l)]['fidelity'] for l in L]
    n2q = [BONDS_4X4 * l for l in L]

    bp_gap = val['ed_ground_energy'] * -1 + val['bppps_final_energy'] * -1
    bp_gap = val['bppps_final_energy'] - val['ed_ground_energy']

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.plot(n2q, gap, '-o', color=C_CEIL, lw=1.8, ms=6,
            label='statevector ceiling (exact)')
    ax.plot([BONDS_4X4 * 3], [bp_gap], 'D', color=C_HVA, ms=13, zorder=5,
            mec='white', mew=1.5, label='BP-PPS trained ($L$=3)')
    ax.annotate(f'reaches the ceiling\nto {abs(bp_gap - gs["3"]["gap"]):.1e}',
                xy=(72, bp_gap), xytext=(88, 1.6), fontsize=8.5, color='#444',
                arrowprops=dict(arrowstyle='->', color='#888', lw=1))
    if l5 is not None:
        ax.plot([BONDS_4X4 * 5], [l5['gap_to_E0']], 'D', color='#F57C00',
                ms=13, zorder=5, mec='black', mew=1.0,
                label='BP-PPS $L$=5 (ABORTED, Adam only)')
        ax.annotate('%.1fx above the $L$=5 ceiling;\nrun killed in L-BFGS-B at 41 h'
                    % (l5['gap_to_E0'] / gs['5']['gap']),
                    xy=(BONDS_4X4 * 5, l5['gap_to_E0']), xytext=(126, 0.9),
                    fontsize=8.5, color='#E65100',
                    arrowprops=dict(arrowstyle='->', color='#F57C00', lw=1))
    ax.set_yscale('log')
    ax.set_xlabel('2Q gate count  (4x4)')
    ax.set_ylabel('energy gap  $E - E_0$')
    ax.set_title('(a) Energy above the exact ground state', fontsize=12)
    ax.legend(fontsize=9)
    style(ax)

    ax = axes[1]
    ax.plot(L, fid, '-o', color=C_CEIL, lw=1.8, ms=6,
            label='statevector ceiling (exact)')
    ax.plot([3], [val['gs_fidelity']], 'D', color=C_HVA, ms=13, zorder=5,
            mec='white', mew=1.5, label='BP-PPS trained ($L$=3)')
    ax.axhline(0.5, color=C_REF, ls=':', lw=1.4)
    ax.text(1.1, 0.515, 'parity ceiling if started from $|0\\ldots0\\rangle$',
            fontsize=8.5, color=C_REF, va='bottom', ha='left')
    ax.axvline(5, color='#ffb300', lw=8, alpha=0.22)
    ax.text(5, 0.30, '$L$=5:\nbest value', fontsize=8.5, ha='center',
            color='#7a5c00')
    # No BP-PPS marker at L=5 on this panel: the aborted run recovered its
    # energies from the log but not its angles, and fidelity needs the state.
    if l5 is not None:
        ax.text(5.2, 0.70, 'no BP-PPS $F_0$ at $L$=5:\nthe aborted run kept its\n'
                'energies, not its angles', fontsize=8.5, color='#E65100',
                ha='left', va='top')
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('HVA layers $L$')
    ax.set_ylabel('ground-state fidelity $F_0$')
    ax.set_title('(b) Fidelity vs circuit depth', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    style(ax)

    fig.suptitle('Goal 3: at $L$=3 BP-PPS lands on the exact ceiling of the same '
                 'ansatz, so the remaining gap there is depth, not training.\n'
                 'That is established at $L$=3 only — the single $L$=5 attempt '
                 'was killed 6.7x short of its own ceiling',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    p = os.path.join(OUT, 'report_goal3_layers.png')
    fig.savefig(p, dpi=150, facecolor='white')
    print('wrote', p)


# --------------------------------------------------------------------------
# Figure 4 -- is the depth axis a separate advantage? (no)
# --------------------------------------------------------------------------
def fig_depth():
    rows = load('depth_audit.json')['rows']

    def sel(name):
        r = [x for x in rows if x['circuit'] == name]
        return ([x['n_2q'] for x in r], [x['depth'] for x in r],
                [x['depth_2q'] for x in r])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # (a) depth and 2Q count are collinear, with the same slope for both
    ax = axes[0]
    for name, lab, color, mk in (('hva', 'HVA', C_HVA, 'o'),
                                 ('trotter_s2', 'Trotter $S_2$ (grouped)', C_TROT, 's'),
                                 ('trotter_s4', 'Trotter $S_4$ (grouped)', '#f39c12', '^')):
        x, d, _ = sel(name)
        ax.plot(x, d, '-', color=color, marker=mk, lw=1.8, ms=6, label=lab)
    ax.set_xlabel('2Q gate count')
    ax.set_ylabel('scheduled circuit depth')
    ax.set_title('(a) One ASAP schedule for both:\nthe two axes carry the same information',
                 fontsize=11.5)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.06, 'depth $\\approx$ 5 $\\times$ (2Q / 24)\nfor every curve',
            transform=ax.transAxes, fontsize=9, ha='right', color='#444')
    style(ax)

    # (b) where the old 4.7x came from
    ax = axes[1]
    labels = ['old figure\nqiskit, unsorted', 'matched 2Q\nsame scheduler']
    hva = [15, 15]
    trot = [70, 16]
    xs = np.arange(2)
    w = 0.34
    ax.bar(xs - w / 2, hva, w, color=C_HVA, label='HVA ($L$=3), depth')
    ax.bar(xs + w / 2, trot, w, color=C_TROT, label='Trotter, depth')
    for i, (a, b) in enumerate(zip(hva, trot)):
        ax.text(i, max(a, b) + 2.5, f'{b / a:.2f}$\\times$', ha='center',
                fontsize=12, fontweight='bold',
                color=C_TROT if b / a > 2 else '#555')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel('circuit depth')
    ax.set_ylim(0, 88)
    ax.set_title('(b) The 4.7$\\times$ depth win was a scheduling artefact',
                 fontsize=11.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlabel('left: qiskit also emits 240 2Q gates where the grouped builder needs 72',
                  fontsize=8.5, color='#666', labelpad=9)
    style(ax)

    fig.suptitle('Does the depth axis give an advantage the gate-count axis does not? '
                 'No -- both circuits colour the same 24 bonds the same way',
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(OUT, 'report_depth_audit.png')
    fig.savefig(p, dpi=150, facecolor='white')
    print('wrote', p)


# --------------------------------------------------------------------------
# Figure 5 -- what a gate error actually costs, by what you are measuring
# --------------------------------------------------------------------------
def fig_noise():
    """What a gate error costs, split by what you are actually measuring.

    Read the three panels as one argument with a hard boundary in the middle:

      (a) how much of the circuit can even reach a single-site observable
      (b) the surviving signal that buys, global state vs local observable
      (c) the shots needed to estimate that signal to 10%

    (c) is about ESTIMATING a number, not about producing clean samples, and
    the distinction decides which of the three goals this figure supports.

      * For an OBSERVABLE, depolarising noise multiplies <O> by roughly F. If
        F is known, the estimate can be rescaled, and the only cost is variance
        -- shots ~ 1/F^2. That is what (c) counts, and it is a real technique.
        Goal 1 lives here, and up to t~2 on 10x10 the cost stays inside a
        routine budget.

      * For SAMPLING, it does not work. There is no rescaling of a probability
        distribution: at F = 0.013 (10x10, L=2, t=2, eps=3e-3) roughly 99% of
        the returned bitstrings are drawn from noise, and no number of shots
        removes them. Goal 2 is a sampling claim, so (c) does NOT say goal 2
        survives to t=2; it says the fidelity can be *measured* there.

    The title used to read "averaging recovers the signal cheaply up to t~2",
    which stated the observable case and let it be read as the sampling case.
    Corrected 2026-09-03.

    Two further caveats the exp(-n*eps) model hides, both of which make the
    real numbers worse rather than better: it counts only 2Q depolarising
    error, so no readout error (~1e-2 per qubit on 100 qubits is a factor 0.37
    before a single gate runs), no crosstalk, and no coherent accumulation; and
    the 4x4 curves in (a) show the cone is not a saving on a small lattice --
    82% of the circuit is already inside it at t=0.5.
    """
    rows = load('lightcone_budget.json')['rows']

    def pick(Lx, L):
        r = sorted((x for x in rows if x['Lx'] == Lx and x['n_layers'] == L),
                   key=lambda x: x['k'])
        return r

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) how much of the circuit is even in the cone
    ax = axes[0]
    for Lx, color, mk, lab in ((10, C_HVA, 'o', '10x10'), (4, C_TROT, 's', '4x4')):
        r = pick(Lx, 2)
        ax.plot([x['t'] for x in r],
                [100 * x['n_2q_cone'] / x['n_2q_total'] for x in r],
                '-', color=color, marker=mk, lw=1.8, ms=6, label=f'{lab}, $L$=2')
    ax.axhline(100, color=C_REF, ls=':', lw=1.4)
    ax.text(3.9, 97, 'whole circuit', fontsize=8.5, color=C_REF, ha='right', va='top')
    ax.set_ylim(0, 110)
    ax.set_xlabel('evolution time $t$')
    ax.set_ylabel('2Q gates inside the light cone (%)')
    ax.set_title('(a) A local observable only sees\nits own causal cone', fontsize=11.5)
    ax.legend(fontsize=9, loc='lower right')
    style(ax)

    # (b) the relief that buys, global vs local
    ax = axes[1]
    r = pick(10, 2)
    t = [x['t'] for x in r]
    for key, ls, lab in (('F_global', '-', 'global state $F$'),
                         ('F_local', '--', 'local observable')):
        ax.plot(t, [x['eps_0.003'][key] for x in r], ls, color=C_HVA,
                marker='o', lw=1.8, ms=5, label=f'{lab}, $\\epsilon$=3e-3')
        ax.plot(t, [x['eps_0.001'][key] for x in r], ls, color=C_TROT,
                marker='s', lw=1.8, ms=5, label=f'{lab}, $\\epsilon$=1e-3')
    ax.set_yscale('log')
    ax.set_xlabel('evolution time $t$')
    ax.set_ylabel('surviving signal')
    ax.set_title('(b) 10x10, $L$=2 block:\nlocal beats global by ~2.4x', fontsize=11.5)
    ax.legend(fontsize=7.5, loc='lower left')
    style(ax)

    # (c) and what it costs in shots -- the point the exp(-n*eps) table hides
    ax = axes[2]
    for eps, color, mk in (('eps_0.003', C_HVA, 'o'), ('eps_0.001', C_TROT, 's')):
        ax.plot(t, [x[eps]['xeb_shots_10pct'] for x in r], '-', color=color,
                marker=mk, lw=1.8, ms=6,
                label='$\\epsilon$=%s' % eps.split('_')[1])
    ax.axhspan(1e2, 1e6, color='#66bb6a', alpha=0.16)
    ax.text(0.55, 3e4, 'routine shot budget', fontsize=9, color='#2e6b31')
    ax.set_yscale('log')
    ax.set_ylim(50, 1e11)
    ax.set_xlabel('evolution time $t$')
    ax.set_ylabel('shots for 10% precision on $F$')
    ax.set_title('(c) Shots to *measure* $F$ to 10%\n(not to sample correctly)',
                 fontsize=11.5)
    ax.legend(fontsize=9, loc='upper left')
    style(ax)

    fig.suptitle('A 3e-3 gate error costs an *observable* little up to $t\\approx$2 '
                 '(rescale by $F$, pay $1/F^2$ in shots).\nIt still destroys '
                 '*sampling* there: at $F$=0.013 about 99% of the bitstrings are noise, '
                 'and shots cannot undo that',
                 fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    p = os.path.join(OUT, 'report_noise_budget.png')
    fig.savefig(p, dpi=150, facecolor='white')
    print('wrote', p)


if __name__ == '__main__':
    fig_goal1()
    fig_goal2()
    fig_goal3()
    fig_depth()
    fig_noise()
