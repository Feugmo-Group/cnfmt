#!/usr/bin/env python
"""
Regenerate all paper figures with publication-quality formatting.

Handles:
  B1 — Larger fonts on all figures
  B2 — Extended wall profiles to z/sigma = 6
  B5 — Gul et al. comparison in wall profiles

Usage:
    python -m cnfmt.scripts.regenerate_paper_figures
    python -m cnfmt.scripts.regenerate_paper_figures --output-dir paper/CnnFMT_paper_march6_final/NN_FMT_revised/images
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.paper_figure_style import apply_paper_style
apply_paper_style()

from solvers.fmt_1d_wbii_tensor import (
    WallSolver, RosenfeldFMT, WhiteBearIIFMT, ModifiedRSLT, esFMT_Tensor
)
from solvers.wall_profile import MC_WALL_DATA

PI = np.pi


def fig1_wall_profiles(output_dir: Path):
    """
    Figure 1: Wall profiles at four eta values.
    B1: larger fonts; B2: extended to z/sigma=6; B5: includes Gul et al.
    """
    print("="*60)
    print("Figure 1: Wall density profiles (extended range)")
    print("="*60)

    eta_values = [0.367, 0.393, 0.449, 0.492]
    # Use larger domain for extended profiles
    solver = WallSolver(nz=2048, Lz=8.0, R=0.5)

    functionals = [
        ('Rosenfeld', RosenfeldFMT()),
        ('White Bear II', WhiteBearIIFMT()),
        ('mRSLT', ModifiedRSLT()),
        ('Gül et al.', esFMT_Tensor(A=1.3, B=-1.0)),
    ]

    colors = {
        'Rosenfeld': 'C0',
        'White Bear II': 'C1',
        'mRSLT': 'C2',
        'Gül et al.': 'C3',
    }

    all_results = {}
    for eta in eta_values:
        print(f"\neta = {eta}:")
        all_results[eta] = {}
        for name, func in functionals:
            result = solver.solve(eta, func, max_iter=4000, tol=1e-8, verbose=False)
            all_results[eta][name] = result
            contact = float(result['contact'])
            print(f"  {name}: contact = {contact:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, eta in enumerate(eta_values):
        ax = axes[idx]

        # MD data (extended range, subsampled for clarity)
        if eta in MC_WALL_DATA:
            mc = MC_WALL_DATA[eta]
            rho_bulk = mc.get('rho_bulk', eta / ((4/3) * PI * 0.5**3))
            mc_z = mc['z']
            mc_rho_norm = mc['rho'] / rho_bulk
            # Subsample: every point up to z=2, then every 5th point
            mask = (mc_z <= 6.0)
            z_plot = mc_z[mask]
            rho_plot = mc_rho_norm[mask]
            dense = z_plot <= 2.0
            sparse_idx = np.where(~dense)[0][::5]
            plot_idx = np.concatenate([np.where(dense)[0], sparse_idx])
            plot_idx.sort()
            ax.plot(z_plot[plot_idx], rho_plot[plot_idx], 'ko', ms=3, mfc='white',
                    mew=1.0, alpha=0.8, label='MD', zorder=10)

        # FMT results
        for name, result in all_results[eta].items():
            ax.plot(result['z'], result['rho_norm'], '-', color=colors[name],
                    lw=1.5, label=name)

        # Reference lines
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5, lw=0.8)
        ax.axvline(0.5, color='gray', ls=':', alpha=0.3, lw=0.8)

        ax.set_xlabel(r'$z/\sigma$')
        ax.set_ylabel(r'$\rho(z)/\rho_b$')
        ax.set_title(rf'$\eta = {eta}$')

        # Extended range (reviewer requested z/sigma = 6)
        ax.set_xlim([0.4, 6.0])

        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.9)

        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = output_dir / 'wall_profiles.png'
    plt.savefig(path)
    print(f"\nSaved: {path}")
    plt.close()


def fig2_validation(output_dir: Path):
    """
    Figure 2: FMT validation (c(r) + contact densities).
    Moved to appendix but still needs regeneration with larger fonts.
    """
    print("\n" + "="*60)
    print("Figure 2: FMT validation (appendix)")
    print("="*60)

    eta = 0.367
    solver = WallSolver(nz=2048, Lz=8.0, R=0.5)

    functionals = [
        ('Rosenfeld', RosenfeldFMT()),
        ('White Bear II', WhiteBearIIFMT()),
        ('mRSLT', ModifiedRSLT()),
        ('esFMT(1,-1)', esFMT_Tensor(A=1.0, B=-1.0)),
        ('Gül et al.', esFMT_Tensor(A=1.3, B=-1.0)),
    ]
    colors_list = ['C0', 'C1', 'C2', 'C4', 'C3']

    # Solve for each functional
    results = {}
    for name, func in functionals:
        result = solver.solve(eta, func, max_iter=4000, tol=1e-8, verbose=False)
        results[name] = result
        print(f"  {name}: contact = {float(result['contact']):.3f}")

    # c(r) - Percus-Yevick analytical
    def c_PY_real(r, eta_val, sigma=1.0):
        alpha = (1 + 2*eta_val)**2 / (1 - eta_val)**4
        beta = 6*eta_val * (1 + eta_val/2)**2 / (1 - eta_val)**4
        gamma = eta_val * (1 + 2*eta_val)**2 / (2*(1 - eta_val)**4)
        x = r / sigma
        return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # (a) Density profile
    ax1 = fig.add_subplot(gs[0, 0])
    if eta in MC_WALL_DATA:
        mc = MC_WALL_DATA[eta]
        rho_bulk = mc.get('rho_bulk', eta / ((4/3) * PI * 0.5**3))
        mc_z = mc['z']
        mc_rho_norm = mc['rho'] / rho_bulk
        mask = mc_z <= 4.0
        z_plot = mc_z[mask]
        rho_plot = mc_rho_norm[mask]
        dense = z_plot <= 2.0
        sparse_idx = np.where(~dense)[0][::5]
        plot_idx = np.concatenate([np.where(dense)[0], sparse_idx])
        plot_idx.sort()
        ax1.plot(z_plot[plot_idx], rho_plot[plot_idx], 'ko', ms=3, mfc='white',
                 mew=1.0, label='MD', zorder=10)
    for (name, _), c in zip(functionals, colors_list):
        if name in results:
            ax1.plot(results[name]['z'], results[name]['rho_norm'], '-',
                     color=c, lw=1.3, label=name)
    ax1.set_xlabel(r'$z/\sigma$')
    ax1.set_ylabel(r'$\rho(z)/\rho_b$')
    ax1.set_title(rf'(a) Density profile, $\eta = {eta}$')
    ax1.set_xlim([0.4, 3.0])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # (b) Contact density comparison
    ax2 = fig.add_subplot(gs[0, 1])
    eta_vals = [0.367, 0.393, 0.449, 0.492]
    CS_contacts = [(1 + e + e**2 - e**3) / (1 - e)**3 for e in eta_vals]

    ax2.plot(eta_vals, CS_contacts, 'k--', lw=1.5, label='CS')
    if all(e in MC_WALL_DATA for e in eta_vals):
        mc_contacts = []
        for e in eta_vals:
            mc = MC_WALL_DATA[e]
            rb = mc.get('rho_bulk', e / ((4/3) * PI * 0.5**3))
            mc_contacts.append(mc['rho'][0] / rb)
        ax2.plot(eta_vals, mc_contacts, 'ko', ms=6, mfc='white', mew=1.5, label='MD')

    for (name, func), c in zip(functionals[:3], colors_list[:3]):
        contacts = []
        for e in eta_vals:
            r = solver.solve(e, func, max_iter=2000, tol=1e-7, verbose=False)
            contacts.append(float(r['contact']))
        ax2.plot(eta_vals, contacts, 'o-', color=c, ms=5, lw=1.3, label=name)

    ax2.set_xlabel(r'$\eta$')
    ax2.set_ylabel(r'$\rho(R^+)/\rho_b$')
    ax2.set_title('(b) Contact density')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # (c) c(r) comparison
    ax3 = fig.add_subplot(gs[1, 0])
    r_arr = np.linspace(0.01, 1.5, 500)
    c_py = c_PY_real(r_arr, eta)
    ax3.plot(r_arr, c_py, 'k--', lw=2, label='PY (analytical)')

    # Note: numerical c(r) requires second functional derivative — show PY only
    ax3.set_xlabel(r'$r/\sigma$')
    ax3.set_ylabel(r'$c(r)$')
    ax3.set_title(rf'(c) Direct correlation function, $\eta = {eta}$')
    ax3.legend()
    ax3.grid(True, alpha=0.2)

    # (d) c(k) Fourier transform
    ax4 = fig.add_subplot(gs[1, 1])
    dk = 0.1
    k_arr = np.arange(dk, 30, dk)
    sigma = 1.0
    c_hat = np.zeros_like(k_arr)
    for i, k in enumerate(k_arr):
        # Analytical PY c(k)
        ks = k * sigma
        if ks > 1e-6:
            a0 = (1 + 2*eta)**2 / (1 - eta)**4
            b0 = 6*eta*(1 + eta/2)**2 / (1 - eta)**4
            g0 = eta*(1 + 2*eta)**2 / (2*(1 - eta)**4)
            # FT of piecewise polynomial c(r) for r<sigma
            c_hat[i] = -4*PI*sigma**3 * (
                a0*(np.sin(ks) - ks*np.cos(ks))/ks**3
                - b0*(2*ks*np.sin(ks) - (ks**2 - 2)*np.cos(ks) - 2)/ks**4
                + g0*((-ks**4 + 12*ks**2 - 24)*np.cos(ks) +
                      (4*ks**3 - 24*ks)*np.sin(ks) + 24)/ks**6
            )

    ax4.plot(k_arr, c_hat, 'k-', lw=1.5)
    ax4.set_xlabel(r'$k\sigma$')
    ax4.set_ylabel(r'$\hat{c}(k)$')
    ax4.set_title(rf'(d) Fourier transform $\hat{{c}}(k)$, $\eta = {eta}$')
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    path = output_dir / 'fmt_comprehensive_all_methods.png'
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()


def fig3_four_approaches(output_dir: Path):
    """
    Figure 3: Four training approaches comparison.
    Just regenerates with larger fonts — actual training runs separately.
    """
    print("\n" + "="*60)
    print("Figure 3: Training approaches — skipping (requires training run)")
    print("  Run: python -m cnfmt.scripts.fast_four_approaches")
    print("  Then manually copy output to images/")
    print("="*60)


def fig4_lj_lutsko(output_dir: Path):
    """
    Figure 4: LJ Lutsko baseline (appendix).
    Requires LJ solver — skip if not runnable.
    """
    print("\n" + "="*60)
    print("Figure 4: LJ Lutsko baseline — skipping (requires LJ solver run)")
    print("  Run: python -m cnfmt.lj.phase_diagram")
    print("="*60)


def fig5_lj_nn(output_dir: Path):
    """
    Figure 5: LJ NN extension.
    Requires training — skip.
    """
    print("\n" + "="*60)
    print("Figure 5: LJ NN extension — skipping (requires NN training)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Regenerate paper figures')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for figures')
    parser.add_argument('--figures', nargs='+', type=int, default=[1, 2],
                        help='Which figures to regenerate (default: 1 2)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_funcs = {
        1: fig1_wall_profiles,
        2: fig2_validation,
        3: fig3_four_approaches,
        4: fig4_lj_lutsko,
        5: fig5_lj_nn,
    }

    for fig_num in args.figures:
        if fig_num in figure_funcs:
            figure_funcs[fig_num](output_dir)
        else:
            print(f"Unknown figure number: {fig_num}")


if __name__ == '__main__':
    main()
