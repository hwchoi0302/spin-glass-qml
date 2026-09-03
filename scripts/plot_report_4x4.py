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
# Figure 3 -- goal 3: the three routes into the register, at equal 2Q cost
# --------------------------------------------------------------------------
def _hva_ceiling():
    """Best statevector optimum of the HVA at each L, over both runs that made one.

    Two scripts have optimised the *same* ansatz on the *same* Hamiltonian from
    the same |+...+>, and they disagree because they were given different
    budgets and different starts:

      03_statevector_pilot.py   Trotter warm start, 600 s per L   (L = 1..8)
      03f_gs_competitors.py     +-0.1 random x2, 300 s per L      (L = 1..12)

    The pilot wins at every L >= 3 -- at L=8 by 3.3x in dE -- so taking 03f's
    HVA row as "the ceiling" would understate the ansatz. Neither is a proof of
    optimality; both are lower bounds on what the ansatz can do, so the ceiling
    drawn here is the better of the two at each L, and which run supplied it is
    recorded in `src`.
    """
    comp = load('gs_competitors.json')['hva']
    pilot = load('statevector_pilot.json')['ground_state']
    out = {}
    for L in sorted({int(k) for k in comp} | {int(k) for k in pilot}):
        cands = []
        if str(L) in comp:
            cands.append((comp[str(L)]['gap'], comp[str(L)]['fidelity'], '03f'))
        if str(L) in pilot:
            cands.append((pilot[str(L)]['gap'], pilot[str(L)]['fidelity'], 'pilot'))
        gap, fid, src = min(cands)
        out[L] = dict(n_2q=BONDS_4X4 * L, gap=gap, fidelity=fid, src=src)
    return out


def _monotone(g, y):
    """Running minimum: drop points that a capped optimiser made non-monotone."""
    keep, best = [], np.inf
    for gi, yi in zip(g, y):
        if yi < best:
            best = yi
            keep.append((gi, yi))
    return np.array([k[0] for k in keep]), np.array([k[1] for k in keep])


def _gates_for(g, y, target):
    """2Q gates a route needs to reach `target`, log-log. Returns (gates, extrapolated)."""
    g, y = _monotone(g, y)
    lg, ly, t = np.log(g), np.log(y), np.log(target)
    if t > ly[0]:
        return None, False
    for i in range(len(ly) - 1):
        if ly[i] >= t >= ly[i + 1]:
            f = (ly[i] - t) / (ly[i] - ly[i + 1])
            return float(np.exp(lg[i] + f * (lg[i + 1] - lg[i]))), False
    a, b = np.polyfit(lg[-3:], ly[-3:], 1)
    return float(np.exp((t - b) / a)), True


def fig_goal3():
    """Goal 3 against the two circuits that produce the same deliverable.

    All three routes emit the same gate alphabet -- RX on every qubit, RZZ on
    every bond -- so one layer is 24 two-qubit gates whichever route drew it,
    and matching layer count is matching 2Q count. They differ only in what
    fixes the angles: nothing (adiabatic schedule), 2 free angles per layer
    (QAOA), or 40 (HVA, which is multi-angle QAOA's parameterisation).

    Two things on this figure are measured rather than argued, and one is not:

      measured    the adiabatic curve, re-derived independently to 1.5e-14
      measured    BP-PPS L=3: its angles evaluate to -21.999320 on the
                  statevector, 4.7e-6 from what BP-PPS itself reported
      NOT         that the anneal is a strong opponent. It is a linear
                  schedule with no counterdiabatic term, and the leading CD
                  term is a 1-qubit rotation, i.e. free in this figure's own
                  currency. RUNBOOK 2-6 measures it; until then the ratios
                  here are upper bounds on the advantage over an anneal.
    """
    comp = load('gs_competitors.json')
    ceil = _hva_ceiling()
    val = load('validation_results.json')['ground_state']
    l5_path = os.path.join(RES, 'gs_L5_aborted.json')
    l5 = json.load(open(l5_path)) if os.path.exists(l5_path) else None

    e_gap_ed = comp['ed_first_excited'] - comp['ed_ground_energy']

    ad = sorted(comp['adiabatic'].values(), key=lambda r: r['n_2q'])
    qa = sorted(comp['qaoa'].values(), key=lambda r: r['n_2q'])
    ad_g = np.array([r['n_2q'] for r in ad], float)
    ad_dE = np.array([r['gap'] for r in ad])
    ad_F = np.array([r['fidelity'] for r in ad])
    qa_g = np.array([r['n_2q'] for r in qa], float)
    qa_dE = np.array([r['gap'] for r in qa])
    qa_F = np.array([r['fidelity'] for r in qa])

    # HVA: L<=8 is the ceiling proper; L=10,12 exist only in the 300 s run and
    # come out *worse* than L=8, which is the cap, not the ansatz.
    hv_L = sorted(ceil)
    conv = [L for L in hv_L if L <= 8]
    capped = [L for L in hv_L if L > 8]
    hv_g = np.array([ceil[L]['n_2q'] for L in conv], float)
    hv_dE = np.array([ceil[L]['gap'] for L in conv])
    hv_F = np.array([ceil[L]['fidelity'] for L in conv])

    C_QAOA = '#00897b'
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # ---------------------------------------------------------------- (a)
    ax = axes[0]
    ax.plot(ad_g, ad_dE, '-s', color=C_TROT, lw=1.8, ms=5,
            label='adiabatic Trotter ($M$=2..100, no CD)')
    ax.plot(qa_g, qa_dE, '-^', color=C_QAOA, lw=1.8, ms=6,
            label='QAOA ($p$=1..12, 2 angles/layer)')
    ax.plot(hv_g, hv_dE, '-o', color=C_CEIL, lw=1.8, ms=6,
            label='HVA statevector ceiling (40 angles/layer)')
    ax.plot([ceil[L]['n_2q'] for L in capped], [ceil[L]['gap'] for L in capped],
            'o', mfc='none', mec=C_CEIL, ms=6, lw=0,
            label='HVA, optimiser hit the 300 s cap')
    ax.plot([BONDS_4X4 * 3], [val['energy_gap']], 'D', color=C_HVA, ms=13,
            zorder=6, mec='white', mew=1.5, label='BP-PPS trained, deployable')
    if l5 is not None:
        ax.plot([BONDS_4X4 * 5], [l5['gap_to_E0']], 'D', color='#F57C00', ms=12,
                zorder=6, mec='black', mew=1.0, label='BP-PPS $L$=5 (aborted)')
    ax.axhline(e_gap_ed, color=C_REF, ls=':', lw=1.3)
    ax.text(700, e_gap_ed * 1.2, f'ED gap $E_1-E_0$ = {e_gap_ed:.3f}',
            fontsize=8, color=C_REF)
    ax.annotate('', xy=(72, val['energy_gap']), xytext=(1081, val['energy_gap']),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=1.2))
    ax.text(330, val['energy_gap'] * 0.50, '15.0x fewer 2Q gates\nfor the same energy',
            fontsize=9, color='#444', ha='center', fontweight='bold',
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('2Q gate count  (4x4, 24 per layer)')
    ax.set_ylabel('energy above exact ground state  $E-E_0$')
    ax.set_ylim(6e-3, 12)
    ax.set_title('(a) Accuracy per two-qubit gate', fontsize=12)
    ax.legend(fontsize=7.4, loc='upper right', framealpha=0.97)
    style(ax)

    # ---------------------------------------------------------------- (b)
    ax = axes[1]
    ax.plot(ad_g, ad_F, '-s', color=C_TROT, lw=1.8, ms=5, label='adiabatic Trotter')
    ax.plot(qa_g, qa_F, '-^', color=C_QAOA, lw=1.8, ms=6, label='QAOA')
    ax.plot(hv_g, hv_F, '-o', color=C_CEIL, lw=1.8, ms=6, label='HVA ceiling')
    ax.plot([BONDS_4X4 * 3], [val['gs_fidelity']], 'D', color=C_HVA, ms=13,
            zorder=6, mec='white', mew=1.5, label='BP-PPS trained ($L$=3)')
    ax.axhline(0.5, color=C_REF, ls=':', lw=1.3)
    ax.text(30, 0.52, 'parity ceiling if started from $|0\\ldots0\\rangle$',
            fontsize=8, color=C_REF)
    ax.text(330, 0.10, 'no BP-PPS point at $L$=5:\nthe aborted run kept\n'
            'its energies, not its angles', fontsize=8, color='#E65100')
    ax.set_xscale('log')
    ax.set_xlabel('2Q gate count  (4x4)')
    ax.set_ylabel('ground-state fidelity  $F_0$')
    ax.set_ylim(0, 1.05)
    ax.set_title('(b) Overlap with the true ground state', fontsize=12)
    ax.legend(fontsize=8, loc='center right')
    style(ax)

    # ---------------------------------------------------------------- (c)
    ax = axes[2]
    targets = np.geomspace(2.0, 0.02, 60)
    for g, y, col, lab in ((qa_g, qa_dE, C_QAOA, 'QAOA / HVA'),
                           (ad_g, ad_dE, C_TROT, 'adiabatic / HVA')):
        solid_x, solid_y, dash_x, dash_y = [], [], [], []
        for t in targets:
            num, num_ex = _gates_for(g, y, t)
            den, den_ex = _gates_for(hv_g, hv_dE, t)
            if num is None or den is None:
                continue
            (dash_x if (num_ex or den_ex) else solid_x).append(t)
            (dash_y if (num_ex or den_ex) else solid_y).append(num / den)
        ax.plot(solid_x, solid_y, '-', color=col, lw=2.4, label=lab + '  (measured)')
        ax.plot(dash_x, dash_y, '--', color=col, lw=1.8, alpha=0.75,
                label=lab + '  (extrapolated)')
    ax.plot([val['energy_gap']], [15.0], 'D', color=C_HVA, ms=11, zorder=6,
            mec='white', mew=1.4)
    ax.plot([val['energy_gap']], [1.79], 'D', color=C_HVA, ms=11, zorder=6,
            mec='white', mew=1.4)
    ax.annotate('what BP-PPS actually\ndelivers today: 72 2Q',
                xy=(val['energy_gap'], 15.0), xytext=(1.6, 3.0),
                fontsize=8.5, color='#444', ha='center',
                arrowprops=dict(arrowstyle='->', color='#888', lw=1))
    ax.axhline(1.0, color=C_REF, ls=':', lw=1.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('target accuracy  $E-E_0$   (harder to the right)')
    ax.set_ylabel('2Q gates needed, relative to HVA')
    ax.set_title('(c) The advantage grows with accuracy', fontsize=12)
    ax.legend(fontsize=7.6, loc='upper left')
    style(ax)

    fig.suptitle(
        'Goal 3 — three routes to the same deliverable, priced in two-qubit gates\n'
        'Deployable today: BP-PPS $L$=3, 72 2Q, $E-E_0$=0.473 — 15.0x fewer gates than the anneal, '
        '1.8x fewer than QAOA, both read inside the measured range\n'
        'Caveat: the anneal is a linear schedule with no counterdiabatic term, and the leading CD term '
        'is a 1-qubit rotation, i.e. free in this figure\'s own currency (RUNBOOK 2-6)',
        fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    p = os.path.join(OUT, 'report_goal3_layers.png')
    fig.savefig(p, dpi=150, facecolor='white')
    print('wrote', p)


# --------------------------------------------------------------------------
# Figure 4 -- is the depth axis a separate advantage? (no)
# --------------------------------------------------------------------------
def fig_depth():
    """Accuracy per unit depth at three evolution times, on the fair metric.

    Replaces the old report_depth_audit.png (2026-09-03). That figure's panel
    (b) was a bar chart meant to show where the retracted 4.7x depth win came
    from, and it could not be read: its left pair compared an HVA circuit of 72
    two-qubit gates against a Trotter circuit of 240, counted by two different
    conventions (analytic 4-colour scheduling on one side, qiskit's unsorted
    qc.depth() on the other), while its right pair compared two 72-gate
    circuits under one scheduler. Two independent errors -- a doubled gate
    count and an unscheduled depth -- were collapsed into one "4.67x -> 1.07x"
    story with no way to see which contributed what.

    This puts fidelity on y and scheduled depth on x, which is how goal 1 is
    worded, at T = 0.5, 1.0 and 2.0. Depths are the analytic 4-colour values
    that scripts/03f_depth_audit.py confirmed against a real ASAP scheduler
    exactly: HVA 5*units, grouped Suzuki 5*units+1. Since every circuit here
    splits the same 24 bonds into the same 4 colours, depth = 5*(2Q/24) for all
    of them and this figure is also the gate-count figure -- there is no depth
    advantage separate from the gate count, which was the audit's real point
    and survives as a sentence rather than a bar chart.

    Infidelities are averaged over 24 Haar-random product states, not scored on
    |0...0>. That matters more than it sounds. BP-PPS optimises L_XZ, an
    operator distance, so it degrades by only 1.03-1.07x off |0...0| while
    grouped Trotter degrades by 1.58-1.73x, and the ranking at T >= 1 flips
    between the two metrics: on |0...0> Trotter wins at T = 1.0 and 2.0, and on
    the state average -- the quantity CLAUDE.md's "arbitrary product state"
    actually names -- BP-PPS wins at all three times.
    """
    import glob

    files = {}
    for path in sorted(glob.glob(os.path.join(RES, 'state_averaged*.json'))):
        d = json.load(open(path))
        files[float(d['T'])] = d
    times = sorted(files)

    fams = {
        'grouped S2': dict(color=C_TROT, mk='s', lab='Trotter $S_2$ (grouped)'),
        'grouped S4': dict(color='#f39c12', mk='^', lab='Trotter $S_4$ (grouped)'),
    }

    fig, axes = plt.subplots(1, len(times), figsize=(5.0 * len(times), 4.9),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, T in zip(axes, times):
        curves = {k: {'d': [], 'y': []} for k in fams}
        pts = []
        for r in files[T]['rows']:
            units = r['n_2q'] / BONDS_4X4
            if r['label'].startswith('BP-PPS'):
                # HVA has no leading half-layer; depth is exactly 5 per layer.
                pts.append((5 * units, r['n_2q'], r['infid_avg'],
                            r['label'].split()[2].rstrip(',')))
                continue
            for k in fams:
                if r['label'].startswith(k):
                    curves[k]['d'].append(5 * units + 1)
                    curves[k]['y'].append(r['infid_avg'])

        for k, f in fams.items():
            ax.plot(curves[k]['d'], curves[k]['y'], '-', color=f['color'],
                    marker=f['mk'], lw=1.8, ms=6, label=f['lab'])
        ax.plot([p[0] for p in pts], [p[2] for p in pts], 'D', color=C_HVA,
                ms=12, lw=0, zorder=6, mec='black', mew=1.0,
                label='BP-PPS HVA block, composed')
        # Stagger above/below rather than left/right: the points sit on a
        # steeply falling line, so consecutive labels collide horizontally
        # (L=5 and L=6 at T=0.5) but have room vertically.
        for i, (d, q, y, lab) in enumerate(sorted(pts, key=lambda t: t[0])):
            up = i % 2 == 0
            ax.annotate(lab, xy=(d, y), xytext=(0, 13 if up else -14),
                        textcoords='offset points', fontsize=9.5, color=C_HVA,
                        fontweight='bold', ha='center',
                        va='bottom' if up else 'top')

        # The matched-gate-count advantage across the whole layer sweep.
        #
        # This used to quote L=3 alone, because L=3 was the deepest block that
        # existed. With L=4,5,6 trained the shape of the result is the L trend,
        # not any single point: quoting L=3 now understates it by 2x at
        # T=2.0 (1.20x against L=6's 2.17x) and hides the only thing this
        # figure is here to decide -- whether the advantage grows with depth.
        adv = []
        for d, q, y, lab in sorted(pts, key=lambda t: t[0]):
            match = [(dd, yy) for dd, yy in zip(curves['grouped S2']['d'],
                                                curves['grouped S2']['y'])
                     if abs(dd - 1 - d) < 1e-9]
            if match:
                adv.append((int(lab.split('=')[1]), q, d, match[0][1] / y))
        if adv:
            ax.text(0.03, 0.46,
                    'BP-PPS vs grouped $S_2$, equal 2Q:\n'
                    + '\n'.join('  $L$=%d (%d 2Q):  %.2fx' % (a[0], a[1], a[3])
                                for a in adv),
                    transform=ax.transAxes, fontsize=8.0, color='#333',
                    ha='left', va='top', linespacing=1.35,
                    bbox=dict(boxstyle='round,pad=0.35', fc='white',
                              ec='#BBBBBB', lw=0.8, alpha=0.93))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('scheduled circuit depth')
        ax.set_title('$T$ = %.1f' % T, fontsize=13)
        allx = [v for k in fams for v in curves[k]['d']] + [p[0] for p in pts]
        ax.set_xlim(min(allx) * 0.7, max(allx) * 1.4)
        style(ax)

    axes[0].set_ylabel('state-averaged infidelity $1-F$\n'
                       '(24 Haar-random product states)')
    # Upper right: every curve falls left-to-right, so that corner is the only
    # one guaranteed clear of both the data and the matched-cost callout.
    axes[0].legend(fontsize=8.5, loc='upper right')

    fig.suptitle('Accuracy per unit depth, scored on random product states '
                 'rather than $|0\\ldots0\\rangle$.\n'
                 'The advantage over gate-matched $S_2$ grows with $L$ at '
                 'every time and shrinks with $T$; $L$=2 loses at all three.\n'
                 'Depth $=5\\times$(2Q$/24$) for every circuit here, so this '
                 'is also the gate-count figure — depth is not a second axis.',
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(OUT, 'report_depth_fidelity.png')
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
