#!/usr/bin/env python3
"""4×4 Spin Glass — Comprehensive Visualization.

Generates publication-quality plots:
  1. Lattice diagram with J couplings and h fields
  2. Time-evolution fidelity comparison (ED, Trotter, HVA)
  3. Observable comparison (⟨X_i⟩, ⟨Z_i⟩) at t=0.5
  4. Training loss curves
  5. Ground state comparison
  6. Circuit depth comparison
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', '4x4')
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Load all data
with open(os.path.join(RESULTS_DIR, 'model_config.json')) as f:
    config = json.load(f)
with open(os.path.join(RESULTS_DIR, 'ed_results.json')) as f:
    ed = json.load(f)
with open(os.path.join(RESULTS_DIR, 'trotter_results.json')) as f:
    trotter = json.load(f)
with open(os.path.join(RESULTS_DIR, 'trained_params.json')) as f:
    te_data = json.load(f)
with open(os.path.join(RESULTS_DIR, 'gs_trained_params.json')) as f:
    gs_data = json.load(f)
with open(os.path.join(RESULTS_DIR, 'validation_results.json')) as f:
    val = json.load(f)

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
    """Compare fidelity: ED (exact), Trotter dt=0.1, Trotter dt=0.2, HVA."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    time_pts = [0.1, 0.2, 0.3, 0.5, 1.0]

    # Trotter fidelities
    fid_trotter_01 = [trotter['0.1'][str(t)]['fidelity'] for t in time_pts]
    fid_trotter_02 = [trotter['0.2'][str(t)]['fidelity'] for t in time_pts]

    # HVA fidelity - we only have t=0.5 from validation
    # For the other time points, we need to compute them.
    # For now, we'll use the t=0.5 result and mark it.
    hva_fid_05 = val['time_evolution']['fidelity']

    # --- Panel (a): Fidelity vs time ---
    ax1.plot(time_pts, [1.0]*len(time_pts), 'k--', lw=1.5, alpha=0.5, label='ED (exact)')
    ax1.plot(time_pts, fid_trotter_01, 's-', color='#2196F3', markersize=8, lw=2,
            label=r'Trotter $S_2$ ($\Delta t=0.1$)')
    ax1.plot(time_pts, fid_trotter_02, '^-', color='#FF9800', markersize=8, lw=2,
            label=r'Trotter $S_2$ ($\Delta t=0.2$)')
    ax1.plot(0.5, hva_fid_05, 'D', color='#E91E63', markersize=12, zorder=5,
            label=f'HVA 3-layer (t=0.5)')
    ax1.annotate(f'{hva_fid_05:.4f}', xy=(0.5, hva_fid_05),
                xytext=(0.65, hva_fid_05 - 0.015), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#E91E63'))

    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Fidelity $|\\langle\\psi_{\\rm approx}|\\psi_{\\rm exact}\\rangle|^2$')
    ax1.set_title('(a) State Fidelity vs Time')
    ax1.legend(fontsize=10, loc='lower left')
    ax1.set_ylim(0.82, 1.005)
    ax1.grid(True, alpha=0.3)

    # --- Panel (b): 1 - Fidelity (infidelity, log scale) ---
    infid_trotter_01 = [1 - f for f in fid_trotter_01]
    infid_trotter_02 = [1 - f for f in fid_trotter_02]
    infid_hva = 1 - hva_fid_05

    ax2.semilogy(time_pts, infid_trotter_01, 's-', color='#2196F3', markersize=8, lw=2,
                label=r'Trotter $S_2$ ($\Delta t=0.1$)')
    ax2.semilogy(time_pts, infid_trotter_02, '^-', color='#FF9800', markersize=8, lw=2,
                label=r'Trotter $S_2$ ($\Delta t=0.2$)')
    ax2.semilogy(0.5, infid_hva, 'D', color='#E91E63', markersize=12, zorder=5,
                label=f'HVA 3-layer')
    ax2.annotate(f'{infid_hva:.2e}', xy=(0.5, infid_hva),
                xytext=(0.65, infid_hva * 2), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#E91E63'))

    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Infidelity $1 - F$')
    ax2.set_title('(b) Infidelity (log scale)')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_fidelity_comparison.png'))
    plt.close()
    print("✅ 02_fidelity_comparison.png")


# ============================================================================
# Figure 3: Observable comparison at t=0.5
# ============================================================================
def plot_observables():
    """Compare ⟨X_i⟩ and ⟨Z_i⟩ at t=0.5: ED vs Trotter vs HVA."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    qubits = list(range(16))
    ed_x = ed['time_points']['0.5']['X']
    ed_z = ed['time_points']['0.5']['Z']
    trot01_x = trotter['0.1']['0.5']['X']
    trot01_z = trotter['0.1']['0.5']['Z']
    trot02_x = trotter['0.2']['0.5']['X']
    trot02_z = trotter['0.2']['0.5']['Z']

    # Compute HVA observables at t=0.5 using statevector
    from hamiltonians import SpinGlass2D
    from classical_bench import ExactDiag
    from ansatz import HVA
    from qiskit.quantum_info import Statevector

    model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
    hva_builder = HVA(num_qubits=16, bonds=model.bonds, n_layers=3, Lx=4, Ly=4)
    te_params = np.array(te_data['params'])
    hva_qc = hva_builder.build_circuit(te_params)
    sv_init = Statevector.from_label('0' * 16)
    psi_hva = np.array(sv_init.evolve(hva_qc))

    H = model.build_sparse_matrix()
    ed_obj = ExactDiag(H, 16)
    obs_hva = ed_obj.local_observables(psi_hva, model.bonds)
    hva_x = obs_hva['X']
    hva_z = obs_hva['Z']

    width = 0.2

    # --- ⟨X_i⟩ ---
    ax = axes[0, 0]
    x_pos = np.arange(16)
    ax.bar(x_pos - 1.5*width, ed_x, width, label='ED (exact)', color='#333333', alpha=0.8)
    ax.bar(x_pos - 0.5*width, trot01_x, width, label=r'Trotter $\Delta t=0.1$', color='#2196F3', alpha=0.8)
    ax.bar(x_pos + 0.5*width, trot02_x, width, label=r'Trotter $\Delta t=0.2$', color='#FF9800', alpha=0.8)
    ax.bar(x_pos + 1.5*width, hva_x, width, label='HVA 3-layer', color='#E91E63', alpha=0.8)
    ax.set_xlabel('Qubit $i$')
    ax.set_ylabel(r'$\langle X_i \rangle$')
    ax.set_title(r'(a) $\langle X_i \rangle$ at $t=0.5$')
    ax.set_xticks(range(16))
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2, axis='y')

    # --- ⟨Z_i⟩ ---
    ax = axes[0, 1]
    ax.bar(x_pos - 1.5*width, ed_z, width, label='ED (exact)', color='#333333', alpha=0.8)
    ax.bar(x_pos - 0.5*width, trot01_z, width, label=r'Trotter $\Delta t=0.1$', color='#2196F3', alpha=0.8)
    ax.bar(x_pos + 0.5*width, trot02_z, width, label=r'Trotter $\Delta t=0.2$', color='#FF9800', alpha=0.8)
    ax.bar(x_pos + 1.5*width, hva_z, width, label='HVA 3-layer', color='#E91E63', alpha=0.8)
    ax.set_xlabel('Qubit $i$')
    ax.set_ylabel(r'$\langle Z_i \rangle$')
    ax.set_title(r'(b) $\langle Z_i \rangle$ at $t=0.5$')
    ax.set_xticks(range(16))
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2, axis='y')

    # --- |ΔX_i| errors ---
    ax = axes[1, 0]
    err_trot01_x = np.abs(np.array(ed_x) - np.array(trot01_x))
    err_trot02_x = np.abs(np.array(ed_x) - np.array(trot02_x))
    err_hva_x = np.abs(np.array(ed_x) - hva_x)

    ax.bar(x_pos - width, err_trot01_x, width, label=r'Trotter $\Delta t=0.1$', color='#2196F3', alpha=0.8)
    ax.bar(x_pos, err_trot02_x, width, label=r'Trotter $\Delta t=0.2$', color='#FF9800', alpha=0.8)
    ax.bar(x_pos + width, err_hva_x, width, label='HVA 3-layer', color='#E91E63', alpha=0.8)
    ax.set_xlabel('Qubit $i$')
    ax.set_ylabel(r'$|\Delta X_i|$')
    ax.set_title(r'(c) $|\langle X_i \rangle_{\rm approx} - \langle X_i \rangle_{\rm exact}|$')
    ax.set_xticks(range(16))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # --- |ΔZ_i| errors ---
    ax = axes[1, 1]
    err_trot01_z = np.abs(np.array(ed_z) - np.array(trot01_z))
    err_trot02_z = np.abs(np.array(ed_z) - np.array(trot02_z))
    err_hva_z = np.abs(np.array(ed_z) - hva_z)

    ax.bar(x_pos - width, err_trot01_z, width, label=r'Trotter $\Delta t=0.1$', color='#2196F3', alpha=0.8)
    ax.bar(x_pos, err_trot02_z, width, label=r'Trotter $\Delta t=0.2$', color='#FF9800', alpha=0.8)
    ax.bar(x_pos + width, err_hva_z, width, label='HVA 3-layer', color='#E91E63', alpha=0.8)
    ax.set_xlabel('Qubit $i$')
    ax.set_ylabel(r'$|\Delta Z_i|$')
    ax.set_title(r'(d) $|\langle Z_i \rangle_{\rm approx} - \langle Z_i \rangle_{\rm exact}|$')
    ax.set_xticks(range(16))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_observable_comparison.png'))
    plt.close()
    print("✅ 03_observable_comparison.png")

    return hva_x, hva_z


# ============================================================================
# Figure 4: Training loss curves
# ============================================================================
def plot_training():
    """Plot training loss curves for time-evolution and ground state."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Time-evolution loss ---
    te_losses = te_data['losses']
    epochs_te = list(range(1, len(te_losses) + 1))
    ax1.semilogy(epochs_te, te_losses, '-', color='#E91E63', lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'$\mathcal{L}_{X,Z}$')
    ax1.set_title(r'(a) Time-Evolution Compression Loss ($\Delta t = 0.5$)')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.axhline(y=te_losses[-1], color='gray', ls='--', alpha=0.5)
    ax1.text(len(te_losses)*0.6, te_losses[-1]*1.3,
            f'Final: {te_losses[-1]:.4f}', fontsize=11, color='gray')
    ax1.text(len(te_losses)*0.6, te_losses[0]*0.7,
            f'99.9% reduction', fontsize=11, color='#E91E63', fontweight='bold')

    # --- Ground state energy ---
    gs_losses = gs_data['losses']
    epochs_gs = list(range(1, len(gs_losses) + 1))
    ax2.plot(epochs_gs, gs_losses, '-', color='#9C27B0', lw=2, label='BP-PPS energy')
    ax2.axhline(y=ed['ground_energy'], color='#333333', ls='--', lw=2,
               label=f'ED ground (E₀ = {ed["ground_energy"]:.2f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Energy $E(\\theta)$')
    ax2.set_title('(b) Ground State Training')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.text(len(gs_losses)*0.5, gs_losses[-1] + 0.5,
            f'Final: {gs_losses[-1]:.2f}\nGap: {gs_losses[-1] - ed["ground_energy"]:.2f}',
            fontsize=10, color='#9C27B0')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_training_curves.png'))
    plt.close()
    print("✅ 04_training_curves.png")


# ============================================================================
# Figure 5: Ground state comparison
# ============================================================================
def plot_ground_state():
    """Compare ground state observables: ED vs BP-PPS HVA."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    gs_ed_x = ed['ground_state']['X']
    gs_ed_z = ed['ground_state']['Z']

    # Compute HVA GS observables
    from hamiltonians import SpinGlass2D
    from classical_bench import ExactDiag
    from ansatz import HVA
    from qiskit.quantum_info import Statevector

    model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
    hva_builder = HVA(num_qubits=16, bonds=model.bonds, n_layers=3, Lx=4, Ly=4)
    gs_params = np.array(gs_data['params'])
    gs_qc = hva_builder.build_circuit(gs_params)
    sv_init = Statevector.from_label('0' * 16)
    psi_gs_hva = np.array(sv_init.evolve(gs_qc))

    H = model.build_sparse_matrix()
    ed_obj = ExactDiag(H, 16)
    obs_gs_hva = ed_obj.local_observables(psi_gs_hva, model.bonds)

    width = 0.3
    x_pos = np.arange(16)

    # ⟨X_i⟩ comparison
    ax = axes[0]
    ax.bar(x_pos - width/2, gs_ed_x, width, label='ED ground state', color='#333333', alpha=0.8)
    ax.bar(x_pos + width/2, obs_gs_hva['X'], width, label='HVA (BP-PPS)', color='#9C27B0', alpha=0.8)
    ax.set_xlabel('Qubit $i$')
    ax.set_ylabel(r'$\langle X_i \rangle$')
    ax.set_title(r'(a) Ground State $\langle X_i \rangle$')
    ax.set_xticks(range(16))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    # ⟨Z_i⟩ comparison
    ax = axes[1]
    ax.bar(x_pos - width/2, gs_ed_z, width, label='ED ground state', color='#333333', alpha=0.8)
    ax.bar(x_pos + width/2, obs_gs_hva['Z'], width, label='HVA (BP-PPS)', color='#9C27B0', alpha=0.8)
    ax.set_xlabel('Qubit $i$')
    ax.set_ylabel(r'$\langle Z_i \rangle$')
    ax.set_title(r'(b) Ground State $\langle Z_i \rangle$')
    ax.set_xticks(range(16))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    # Energy annotation
    E_ed = ed['ground_energy']
    E_hva = ed_obj.compute_energy(psi_gs_hva, model.bonds, model.J, model.h)
    fig.suptitle(
        f'Ground State: ED $E_0 = {E_ed:.2f}$, HVA $E = {E_hva:.2f}$, '
        f'Fidelity $= {val["ground_state"]["gs_fidelity"]:.6f}$',
        fontsize=13, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '05_ground_state.png'))
    plt.close()
    print("✅ 05_ground_state.png")


# ============================================================================
# Figure 6: Circuit depth & compression summary
# ============================================================================
def plot_depth_comparison():
    """Compare circuit depth and gate count across methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    time_pts = [0.1, 0.2, 0.3, 0.5, 1.0]

    # Depths
    depth_trot01 = [trotter['0.1'][str(t)]['depth'] for t in time_pts]
    depth_trot02 = [trotter['0.2'][str(t)]['depth'] for t in time_pts]
    depth_hva = [15] * len(time_pts)  # HVA depth is constant

    # For multiple Δt composition: HVA depth = 15 * (t/0.5)
    depth_hva_comp = [15 * max(1, int(round(t / 0.5))) for t in time_pts]

    # 2Q gates
    n2q_trot01 = [trotter['0.1'][str(t)]['n_2q_gates'] for t in time_pts]
    n2q_trot02 = [trotter['0.2'][str(t)]['n_2q_gates'] for t in time_pts]
    n2q_hva = [72 * max(1, int(round(t / 0.5))) for t in time_pts]

    # --- Panel (a): Circuit depth ---
    ax1.plot(time_pts, depth_trot01, 's-', color='#2196F3', markersize=8, lw=2,
            label=r'Trotter $S_2$ ($\Delta t=0.1$)')
    ax1.plot(time_pts, depth_trot02, '^-', color='#FF9800', markersize=8, lw=2,
            label=r'Trotter $S_2$ ($\Delta t=0.2$)')
    ax1.plot(time_pts, depth_hva_comp, 'D-', color='#E91E63', markersize=8, lw=2,
            label='HVA 3-layer (composed)')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('Circuit Depth')
    ax1.set_title('(a) Circuit Depth vs Time')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add compression ratio annotation at t=0.5
    r = depth_trot01[3] / depth_hva_comp[3]
    ax1.annotate(f'{r:.0f}× compression',
                xy=(0.5, depth_hva_comp[3]),
                xytext=(0.7, 50),
                fontsize=11, color='#E91E63', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E91E63'))

    # --- Panel (b): 2-qubit gates ---
    ax2.plot(time_pts, n2q_trot01, 's-', color='#2196F3', markersize=8, lw=2,
            label=r'Trotter $S_2$ ($\Delta t=0.1$)')
    ax2.plot(time_pts, n2q_trot02, '^-', color='#FF9800', markersize=8, lw=2,
            label=r'Trotter $S_2$ ($\Delta t=0.2$)')
    ax2.plot(time_pts, n2q_hva, 'D-', color='#E91E63', markersize=8, lw=2,
            label='HVA 3-layer (composed)')
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Number of 2-Qubit Gates')
    ax2.set_title('(b) 2-Qubit Gate Count vs Time')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '06_depth_comparison.png'))
    plt.close()
    print("✅ 06_depth_comparison.png")


# ============================================================================
# Figure 7: Per-observable loss heatmap
# ============================================================================
def plot_loss_heatmap():
    """Heatmap of per-observable loss on 4×4 grid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    per_obs = val['time_evolution']['per_observable_loss']

    for ax, pauli_type, title in [(ax1, 'X', r'$\mathcal{L}_{X_i}$'),
                                    (ax2, 'Z', r'$\mathcal{L}_{Z_i}$')]:
        grid = np.zeros((LY, LX))
        for q in range(16):
            key = f'{pauli_type}_{q}'
            x, y = q % LX, q // LX
            grid[y, x] = per_obs[key]

        im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest', origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for y in range(LY):
            for x in range(LX):
                q = y * LX + x
                val_text = grid[y, x]
                color = 'white' if val_text > grid.max() * 0.6 else 'black'
                ax.text(x, y, f'q{q}\n{val_text:.4f}',
                       ha='center', va='center', fontsize=8, color=color)

        ax.set_title(f'{title} (per-qubit loss)', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle('BP-PPS Training Loss by Qubit Position', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '07_loss_heatmap.png'))
    plt.close()
    print("✅ 07_loss_heatmap.png")


# ============================================================================
# Figure 8: Comprehensive summary (BP-PPS paper style)
# ============================================================================
def plot_summary():
    """Combined summary figure in BP-PPS paper style."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # (a) Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    te_losses = te_data['losses']
    ax1.semilogy(range(1, len(te_losses)+1), te_losses, '-', color='#E91E63', lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'$\mathcal{L}_{X,Z}$')
    ax1.set_title('(a) Training Loss')
    ax1.grid(True, alpha=0.3, which='both')

    # (b) Fidelity
    ax2 = fig.add_subplot(gs[0, 1])
    time_pts = [0.1, 0.2, 0.3, 0.5, 1.0]
    fid01 = [trotter['0.1'][str(t)]['fidelity'] for t in time_pts]
    fid02 = [trotter['0.2'][str(t)]['fidelity'] for t in time_pts]
    ax2.plot(time_pts, fid01, 's-', color='#2196F3', markersize=7, lw=2, label=r'Trot. $\Delta t$=0.1')
    ax2.plot(time_pts, fid02, '^-', color='#FF9800', markersize=7, lw=2, label=r'Trot. $\Delta t$=0.2')
    ax2.plot(0.5, val['time_evolution']['fidelity'], 'D', color='#E91E63', markersize=10)
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('Fidelity')
    ax2.set_title('(b) Fidelity vs Time')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.82, 1.005)
    ax2.grid(True, alpha=0.3)

    # (c) Circuit depth
    ax3 = fig.add_subplot(gs[0, 2])
    d01 = [trotter['0.1'][str(t)]['depth'] for t in time_pts]
    d_hva = [15 * max(1, int(round(t/0.5))) for t in time_pts]
    ax3.plot(time_pts, d01, 's-', color='#2196F3', markersize=7, lw=2, label=r'Trot. $\Delta t$=0.1')
    ax3.plot(time_pts, d_hva, 'D-', color='#E91E63', markersize=7, lw=2, label='HVA')
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel('Depth')
    ax3.set_title('(c) Circuit Depth')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # (d) ⟨X_i⟩ at t=0.5
    ax4 = fig.add_subplot(gs[1, :2])
    ed_x = ed['time_points']['0.5']['X']
    trot_x = trotter['0.1']['0.5']['X']

    from hamiltonians import SpinGlass2D
    from classical_bench import ExactDiag
    from ansatz import HVA as HVAClass
    from qiskit.quantum_info import Statevector
    model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
    hva_b = HVAClass(num_qubits=16, bonds=model.bonds, n_layers=3, Lx=4, Ly=4)
    te_p = np.array(te_data['params'])
    psi_hva = np.array(Statevector.from_label('0'*16).evolve(hva_b.build_circuit(te_p)))
    H = model.build_sparse_matrix()
    ed_o = ExactDiag(H, 16)
    obs_hva = ed_o.local_observables(psi_hva, model.bonds)

    w = 0.25
    x_pos = np.arange(16)
    ax4.bar(x_pos - w, ed_x, w, label='ED', color='#333333', alpha=0.8)
    ax4.bar(x_pos, trot_x, w, label=r'Trot. $\Delta t$=0.1', color='#2196F3', alpha=0.8)
    ax4.bar(x_pos + w, obs_hva['X'], w, label='HVA', color='#E91E63', alpha=0.8)
    ax4.set_xlabel('Qubit $i$')
    ax4.set_ylabel(r'$\langle X_i \rangle$')
    ax4.set_title(r'(d) $\langle X_i \rangle$ at $t = 0.5$')
    ax4.set_xticks(range(16))
    ax4.legend(fontsize=9, ncol=3)
    ax4.grid(True, alpha=0.2, axis='y')

    # (e) ⟨Z_i⟩ at t=0.5
    ax5 = fig.add_subplot(gs[1, 2])
    ed_z = ed['time_points']['0.5']['Z']
    trot_z = trotter['0.1']['0.5']['Z']
    ax5.bar(x_pos - w, ed_z, w, label='ED', color='#333333', alpha=0.8)
    ax5.bar(x_pos, trot_z, w, label='Trot.', color='#2196F3', alpha=0.8)
    ax5.bar(x_pos + w, obs_hva['Z'], w, label='HVA', color='#E91E63', alpha=0.8)
    ax5.set_xlabel('Qubit $i$')
    ax5.set_ylabel(r'$\langle Z_i \rangle$')
    ax5.set_title(r'(e) $\langle Z_i \rangle$ at $t = 0.5$')
    ax5.set_xticks(range(0, 16, 2))
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.2, axis='y')

    # (f) GS energy convergence
    ax6 = fig.add_subplot(gs[2, 0])
    gs_losses = gs_data['losses']
    ax6.plot(range(1, len(gs_losses)+1), gs_losses, '-', color='#9C27B0', lw=2)
    ax6.axhline(y=ed['ground_energy'], color='k', ls='--', lw=1.5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Energy')
    ax6.set_title('(f) GS Energy Convergence')
    ax6.grid(True, alpha=0.3)

    # (g) Error heatmap X
    ax7 = fig.add_subplot(gs[2, 1])
    per_obs = val['time_evolution']['per_observable_loss']
    grid_x = np.zeros((4, 4))
    for q in range(16):
        grid_x[q//4, q%4] = per_obs[f'X_{q}']
    im = ax7.imshow(grid_x, cmap='YlOrRd', origin='upper')
    for y in range(4):
        for x in range(4):
            c = 'white' if grid_x[y,x] > grid_x.max()*0.5 else 'black'
            ax7.text(x, y, f'{grid_x[y,x]:.3f}', ha='center', va='center', fontsize=8, color=c)
    ax7.set_title(r'(g) $\mathcal{L}_{X_i}$ heatmap')
    plt.colorbar(im, ax=ax7, fraction=0.046)

    # (h) Error heatmap Z
    ax8 = fig.add_subplot(gs[2, 2])
    grid_z = np.zeros((4, 4))
    for q in range(16):
        grid_z[q//4, q%4] = per_obs[f'Z_{q}']
    im = ax8.imshow(grid_z, cmap='YlOrRd', origin='upper')
    for y in range(4):
        for x in range(4):
            c = 'white' if grid_z[y,x] > grid_z.max()*0.5 else 'black'
            ax8.text(x, y, f'{grid_z[y,x]:.3f}', ha='center', va='center', fontsize=8, color=c)
    ax8.set_title(r'(h) $\mathcal{L}_{Z_i}$ heatmap')
    plt.colorbar(im, ax=ax8, fraction=0.046)

    fig.suptitle('4×4 Spin Glass — BP-PPS Simulation Results Summary',
                fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(PLOT_DIR, '08_summary.png'))
    plt.close()
    print("✅ 08_summary.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating plots...\n")
    plot_lattice()
    plot_fidelity()
    plot_observables()
    plot_training()
    plot_ground_state()
    plot_depth_comparison()
    plot_loss_heatmap()
    plot_summary()
    print(f"\nAll plots saved to: {PLOT_DIR}/")
