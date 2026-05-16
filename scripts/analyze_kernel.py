#!/usr/bin/env python3
"""
Kernel Analysis and Comparison Plots
======================================

Loads a trained NonlocalLutskoFunctional checkpoint and produces:

1. Learned kernel K̂(k) in Fourier space — what length scales were selected
2. Learned kernel K(r) in real space — spatial range of nonlocal averaging
3. Wall density profiles: Nonlocal NN vs Lutsko (A=1,B=0) vs Gül (A=1.3,B=-1) vs MD
4. A(η), B(η), C(η) curves vs fixed functionals

Usage
-----
    python -m scripts.analyze_kernel outputs/run3/checkpoints/nonlocal_final.eqx
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from core.grid import Grid
from core.weights import FMTKernels
from core.densities import WeightedDensityCalculator
from nonlocal_ext.functional import NonlocalLutskoFunctional
from nonlocal_ext.kernels import LearnableKernel
from neural.network import NonlocalConditionalNetwork
from solvers.fmt_1d_wbii_tensor import WallSolver


# ─────────────────────────────────────────────────────────────
# Davidchack et al. 2016 MD contact densities
# ─────────────────────────────────────────────────────────────
MD_CONTACT = {
    0.367: 5.36,
    0.393: 6.65,
    0.449: 9.33,
    0.492: 12.32,
}


def load_functional(checkpoint_path: str, grid: Grid) -> NonlocalLutskoFunctional:
    """Load trained functional from checkpoint."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    kernels = FMTKernels(grid, R=0.5)
    calculator = WeightedDensityCalculator(kernels)
    network = NonlocalConditionalNetwork(k1)
    kernel = LearnableKernel(k2)
    functional = NonlocalLutskoFunctional(network, kernel, calculator, grid)

    functional = eqx.tree_deserialise_leaves(checkpoint_path, functional)
    print(f"  Loaded: {checkpoint_path}")
    return functional


def plot_kernel(functional: NonlocalLutskoFunctional, grid: Grid, output_dir: Path):
    """Plot learned kernel in Fourier and real space."""
    kernel = functional.kernel

    print(f"  Kernel: {kernel.n_gaussians} Gaussians, r_cut={kernel.r_cut}σ")
    print(f"  Widths σᵢ: {np.array(kernel.widths)}")
    print(f"  Amplitudes aᵢ: {np.array(kernel.amplitudes)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── (a) K̂(k): Fourier-space kernel ──
    ax = axes[0]
    k_vals = np.linspace(0, 4 * np.pi, 500)  # up to 4π/σ
    K_hat = np.array(kernel.fourier(jnp.array(k_vals)))
    ax.plot(k_vals / (2 * np.pi), K_hat, 'b-', lw=2.5)
    # Mark physically meaningful wavenumbers
    ax.axvline(1.0, color='gray', ls='--', alpha=0.6, label='k=2π/σ (nearest neighbor)')
    ax.axvline(2.0, color='orange', ls=':', alpha=0.6, label='k=4π/σ (2nd shell)')
    # Mark sine-wave training wavenumbers for L=10σ: k=2πm/L
    for m, c in zip([1, 2, 4, 8], ['red', 'green', 'purple', 'brown']):
        k_m = m / 10.0  # in units of 2π/σ
        ax.axvline(k_m, color=c, ls='-', alpha=0.4, lw=1.5, label=f'm={m} (k={m/10:.1f}·2π/σ)')
    ax.set_xlabel(r'$k / (2\pi/\sigma)$', fontsize=13)
    ax.set_ylabel(r'$\hat{K}(k)$', fontsize=13)
    ax.set_title('(a) Kernel — Fourier space', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 1.05])

    # ── (b) Individual Gaussian components in K̂(k) ──
    ax = axes[1]
    widths = np.array(kernel.widths)
    amps = np.array(kernel.amplitudes)
    colors = plt.cm.tab10(np.linspace(0, 0.9, kernel.n_gaussians))
    for i, (a, s, c) in enumerate(zip(amps, widths, colors)):
        comp = a * np.exp(-0.5 * s**2 * k_vals**2)
        ax.plot(k_vals / (2 * np.pi), comp, color=c, lw=1.5,
                label=f'G{i+1}: a={a:.3f}, σ={s:.3f}σ')
    ax.plot(k_vals / (2 * np.pi), K_hat, 'k-', lw=2.5, label='Total K̂(k)')
    ax.set_xlabel(r'$k / (2\pi/\sigma)$', fontsize=13)
    ax.set_ylabel(r'Component contributions', fontsize=13)
    ax.set_title('(b) Gaussian decomposition of K̂(k)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 4])

    # ── (c) K(r): Real-space kernel ──
    ax = axes[2]
    r_vals = np.linspace(0, kernel.r_cut * 1.1, 400)
    K_r = np.array(kernel.real_space(jnp.array(r_vals)))
    ax.plot(r_vals, K_r, 'b-', lw=2.5, label='K(r)')
    ax.axvline(0.5, color='red', ls='--', alpha=0.7, label='r=σ/2 (HS radius)')
    ax.axvline(1.0, color='gray', ls='--', alpha=0.7, label='r=σ (diameter)')
    ax.axvline(kernel.r_cut, color='orange', ls=':', alpha=0.7, label=f'r_cut={kernel.r_cut}σ')
    ax.fill_between(r_vals, K_r, alpha=0.15, color='blue')
    ax.set_xlabel(r'$r/\sigma$', fontsize=13)
    ax.set_ylabel(r'$K(r)$', fontsize=13)
    ax.set_title('(c) Kernel — real space', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, kernel.r_cut * 1.1])

    plt.suptitle('Learned Nonlocal Kernel K(r) / K̂(k)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'kernel_analysis.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_AB_comparison(functional: NonlocalLutskoFunctional, output_dir: Path):
    """Plot A(η), B(η), C(η) vs all fixed functionals."""
    eta_range = np.linspace(0.01, 0.52, 200)

    A_nn, B_nn = [], []
    for eta in eta_range:
        A, B = functional.bulk_parameters(float(eta))
        A_nn.append(float(A))
        B_nn.append(float(B))
    A_nn = np.array(A_nn)
    B_nn = np.array(B_nn)
    C_nn = 8 * A_nn + 2 * B_nn - 9

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Reference values
    refs = {
        'Rosenfeld (C=+3)': (1.5, 0.0, 'red', '--'),
        'Lutsko (C=−1)':    (1.0, 0.0, 'gray', '--'),
        'Gül (C=−0.6)':     (1.3, -1.0, 'orange', '-.'),
        'White Bear (C=−2.25)': (1.125, -1.125, 'green', ':'),
    }

    for ax, (vals, ylabel, title) in zip(axes, [
        (A_nn, 'A(η)', '(a) Learned A(η)'),
        (B_nn, 'B(η)', '(b) Learned B(η)'),
        (C_nn, 'C(η) = 8A+2B−9', '(c) Constraint C(η)'),
    ]):
        ax.plot(eta_range, vals, 'b-', lw=2.5, label='Nonlocal NN')
        for label, (A_ref, B_ref, color, ls) in refs.items():
            ref_val = A_ref if 'A' in ylabel else (B_ref if 'B' in ylabel else 8*A_ref + 2*B_ref - 9)
            ax.axhline(ref_val, color=color, ls=ls, lw=1.5, label=label)
        ax.set_xlabel('η', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.52])

    plt.suptitle('Learned Parameters vs Fixed Functionals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'AB_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_wall_comparison(functional: NonlocalLutskoFunctional, grid: Grid, output_dir: Path):
    """Wall profiles: Nonlocal NN vs Lutsko vs Gül vs MD."""
    from solvers.fmt_1d_wbii_tensor import esFMT_Tensor

    eta_vals = [0.367, 0.393, 0.449, 0.492]
    nz, Lz = 2048, 6.0
    solver = WallSolver(nz=nz, Lz=Lz)

    # Fixed-parameter functionals
    lutsko_fmt = esFMT_Tensor(A=1.0, B=0.0)
    gul_fmt    = esFMT_Tensor(A=1.3, B=-1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, eta in zip(axes, eta_vals):
        rho_bulk = 6.0 * eta / np.pi

        # Nonlocal NN: use bulk A, B at this η with esFMT_Tensor
        A_nn, B_nn = functional.bulk_parameters(eta)
        nn_fmt = esFMT_Tensor(A=float(A_nn), B=float(B_nn))
        try:
            res_nn = solver.solve(eta, nn_fmt, max_iter=5000, verbose=False)
            z = np.array(res_nn['z'])
            rho_nn = np.array(res_nn['rho'])
            ax.plot(z, rho_nn / rho_bulk, 'b-', lw=2,
                    label=f'Nonlocal NN (A={float(A_nn):.3f}, B={float(B_nn):.3f})')
            contact_nn = float(rho_nn[np.argmax(rho_nn)])
        except Exception as e:
            print(f"  Warning: NN wall solve failed at η={eta}: {e}")
            contact_nn = None

        # Lutsko
        try:
            res_l = solver.solve(eta, lutsko_fmt, max_iter=5000, verbose=False)
            z = np.array(res_l['z'])
            ax.plot(z, np.array(res_l['rho']) / rho_bulk,
                    'gray', ls='--', lw=1.5, label='Lutsko (A=1, B=0)')
        except Exception as e:
            print(f"  Warning: Lutsko wall solve failed at η={eta}: {e}")

        # Gül
        try:
            res_g = solver.solve(eta, gul_fmt, max_iter=5000, verbose=False)
            z = np.array(res_g['z'])
            ax.plot(z, np.array(res_g['rho']) / rho_bulk,
                    'orange', ls='-.', lw=1.5, label='Gül (A=1.3, B=−1)')
        except Exception as e:
            print(f"  Warning: Gül wall solve failed at η={eta}: {e}")

        # MD contact reference line
        md_c = MD_CONTACT[eta]
        ax.axhline(md_c / rho_bulk, color='black', ls=':', lw=2,
                   label=f'MD contact = {md_c:.2f}')

        ax.set_xlim([0, 5])
        ax.set_ylim([0, None])
        ax.set_xlabel(r'$z/\sigma$', fontsize=12)
        ax.set_ylabel(r'$\rho(z)/\rho_b$', fontsize=12)
        ax.set_title(f'η = {eta}', fontweight='bold', fontsize=13)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Wall Density Profiles: Nonlocal NN vs Fixed Functionals vs MD',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = output_dir / 'wall_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description='Analyze trained nonlocal kernel')
    parser.add_argument('checkpoint', help='Path to .eqx checkpoint file')
    parser.add_argument('--grid-size', type=int, default=32)
    parser.add_argument('--box-length', type=float, default=10.0)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  KERNEL ANALYSIS & COMPARISON PLOTS")
    print("=" * 60)

    n = args.grid_size
    L = args.box_length
    grid = Grid((n, n, n), L)

    print(f"\n  Loading functional (grid={n}³, L={L}σ)...")
    functional = load_functional(str(checkpoint), grid)

    # Print kernel summary
    kernel = functional.kernel
    widths = np.array(kernel.widths)
    amps = np.array(kernel.amplitudes)
    print(f"\n  Kernel summary:")
    print(f"  {'i':>3}  {'amplitude':>10}  {'width (σ)':>10}  {'cutoff k (2π/σ)':>16}")
    print(f"  {'─'*45}")
    for i, (a, s) in enumerate(sorted(zip(amps, widths), key=lambda x: -x[0])):
        k_cutoff = 1.0 / s  # k where exp(-σ²k²/2) = exp(-0.5) ≈ 0.6
        print(f"  {i+1:>3}  {a:>10.4f}  {s:>10.4f}  {k_cutoff:>16.4f}")

    dominant_width = widths[np.argmax(amps)]
    print(f"\n  Dominant length scale: σ_dom = {dominant_width:.3f}σ")
    print(f"  Corresponding real-space range: ~{2*dominant_width:.2f}σ")
    print(f"  Corresponding k cutoff: k ~ {1/dominant_width:.2f} × (2π/σ)")

    print("\n  Generating kernel plots...")
    plot_kernel(functional, grid, output_dir)

    print("\n  Generating A/B comparison plots...")
    plot_AB_comparison(functional, output_dir)

    print("\n  Generating wall profile comparison...")
    plot_wall_comparison(functional, grid, output_dir)

    print(f"\n  All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
