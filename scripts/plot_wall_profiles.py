#!/usr/bin/env python
"""
Generate Wall Profile Plots
============================

Plots hard-sphere density profiles at a planar hard wall for
different packing fractions, comparing FMT predictions with
Monte Carlo reference data.

Usage
-----
    python -m cnfmt.scripts.plot_wall_profiles --output_dir outputs

Author: Computational Materials Science
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from solvers.wall_profile import (
    WallProfileCalculator, 
    WallProfileConfig,
    compute_wall_profiles,
    get_mc_data,
    MC_WALL_DATA
)


def plot_wall_profiles(results: list, output_dir: Path, 
                       show_mc: bool = True,
                       title_suffix: str = ""):
    """
    Plot wall profiles for multiple packing fractions.
    
    Parameters
    ----------
    results : list
        List of result dicts from compute_wall_profiles
    output_dir : Path
        Output directory
    show_mc : bool
        Show Monte Carlo reference data
    title_suffix : str
        Additional title text
    """
    n_profiles = len(results)
    
    if n_profiles <= 2:
        fig, axes = plt.subplots(1, n_profiles, figsize=(6*n_profiles, 5))
        if n_profiles == 1:
            axes = [axes]
    else:
        ncols = 2
        nrows = (n_profiles + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5*nrows))
        axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_profiles))
    
    for i, result in enumerate(results):
        ax = axes[i]
        eta = result['eta']
        z = result['z']
        rho = result['rho']
        rho_bulk = result['rho_bulk']
        A, B = result['A'], result['B']
        
        # Normalize by bulk density
        rho_norm = rho / rho_bulk
        
        # Plot FMT profile
        ax.plot(z, rho_norm, '-', color=colors[i], lw=2, 
                label=f'FMT (A={A:.1f}, B={B:.1f})')
        
        # Plot MC data if available
        if show_mc:
            mc = get_mc_data(eta)
            if mc is not None:
                mc_rho_bulk = rho_bulk  # Assume same normalization
                ax.plot(mc['z'], np.array(mc['rho']) / (eta / ((4/3)*np.pi*0.5**3)),
                       'o', color='red', ms=6, mfc='none', mew=1.5,
                       label='MC (Davidchack et al.)')
        
        # Bulk density line
        ax.axhline(1.0, color='gray', ls='--', alpha=0.7, label='Bulk')
        
        # Wall location
        ax.axvline(0.5, color='black', ls=':', alpha=0.5)
        ax.fill_between([0, 0.5], 0, 15, color='gray', alpha=0.2)
        
        ax.set_xlabel(r'$z/\sigma$', fontsize=12)
        ax.set_ylabel(r'$\rho(z)/\rho_\mathrm{bulk}$', fontsize=12)
        ax.set_title(f'η = {eta:.3f}', fontsize=12)
        ax.set_xlim([0, 5])
        ax.set_ylim([0, max(3.5, np.max(rho_norm) * 1.1)])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Hard-Sphere Density Profiles at Planar Hard Wall{title_suffix}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = output_dir / 'wall_profiles.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_wall_profiles_combined(results: list, output_dir: Path):
    """
    Plot all profiles on a single figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, result in enumerate(results):
        eta = result['eta']
        z = result['z']
        rho = result['rho']
        
        ax.plot(z, rho, '-', color=colors[i % len(colors)], lw=2,
                label=f'η = {eta:.3f}')
        
        # MC data
        mc = get_mc_data(eta)
        if mc is not None:
            ax.plot(mc['z'], mc['rho'], 'o', color=colors[i % len(colors)],
                   ms=6, mfc='none', mew=1.5)
    
    # Wall
    ax.axvline(0.5, color='black', ls=':', alpha=0.5)
    ax.fill_between([0, 0.5], 0, 12, color='gray', alpha=0.2, label='Wall')
    
    ax.set_xlabel(r'$z/\sigma$', fontsize=14)
    ax.set_ylabel(r'$\rho(z)\sigma^3$', fontsize=14)
    ax.set_title('Hard-Sphere Density Profiles at Planar Hard Wall', fontsize=14)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 12])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = output_dir / 'wall_profiles_combined.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_contact_density(results: list, output_dir: Path):
    """
    Plot contact density comparison with exact values.
    """
    etas = [r['eta'] for r in results]
    contacts = [r['contact'] for r in results]
    contacts_exact = [r['contact_exact'] for r in results]
    
    # Get MC contact values
    mc_etas = list(MC_WALL_DATA.keys())
    mc_contacts = [MC_WALL_DATA[eta]['rho'][0] for eta in mc_etas]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Exact (contact theorem)
    eta_range = np.linspace(0.05, 0.55, 100)
    Z_CS = lambda e: (1 + e + e**2 - e**3) / (1 - e)**3
    rho_contact_exact = [e / ((4/3)*np.pi*0.5**3) * Z_CS(e) for e in eta_range]
    ax.plot(eta_range, rho_contact_exact, 'k-', lw=2, label='Contact theorem')
    
    # FMT predictions
    ax.plot(etas, contacts, 's', color='blue', ms=10, mfc='none', mew=2,
           label='FMT')
    
    # MC data
    ax.plot(mc_etas, mc_contacts, 'o', color='red', ms=10, mfc='none', mew=2,
           label='MC')
    
    ax.set_xlabel(r'$\eta$', fontsize=14)
    ax.set_ylabel(r'$\rho(R^+)\sigma^3$', fontsize=14)
    ax.set_title('Contact Density at Hard Wall', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.55])
    ax.set_ylim([0, 12])
    
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = output_dir / 'contact_density.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate wall profile plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--etas', type=float, nargs='+',
                       default=[0.367, 0.393, 0.449, 0.492],
                       help='Packing fractions to compute')
    parser.add_argument('--A', type=float, default=1.0,
                       help='Lutsko parameter A')
    parser.add_argument('--B', type=float, default=0.0,
                       help='Lutsko parameter B')
    parser.add_argument('--n_iter', type=int, default=2000,
                       help='Picard iterations')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("WALL PROFILE CALCULATION")
    print("="*60)
    print(f"Packing fractions: {args.etas}")
    print(f"Lutsko parameters: A = {args.A}, B = {args.B}")
    
    # Configuration
    config = WallProfileConfig(
        n_iter=args.n_iter,
        verbose=True
    )
    
    # Compute profiles
    results = compute_wall_profiles(
        args.etas, 
        A=args.A, 
        B=args.B,
        config=config
    )
    
    # Generate plots
    output_dir = Path(args.output_dir)
    
    plot_wall_profiles(results, output_dir)
    plot_wall_profiles_combined(results, output_dir)
    plot_contact_density(results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'η':>6} {'ρ_contact':>12} {'ρ_exact':>12} {'Error%':>10}")
    print("-"*45)
    for r in results:
        error = abs(r['contact'] - r['contact_exact']) / r['contact_exact'] * 100
        print(f"{r['eta']:6.3f} {r['contact']:12.4f} {r['contact_exact']:12.4f} {error:10.2f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
