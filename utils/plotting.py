"""
Publication-Quality Plotting
============================

Generate high-quality figures for papers and presentations.

Style Guidelines:
- Font size: 12pt for labels, 14pt for titles
- Line width: 2-3 for main curves
- Markers: 6-8 pt
- DPI: 300 for publication
- Colors: Colorblind-friendly palette
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import List, Dict, Optional, Any

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'lines.linewidth': 2,
})

# Colorblind-friendly colors
COLORS = {
    'blue': '#0077BB',
    'orange': '#EE7733',
    'green': '#009988',
    'red': '#CC3311',
    'purple': '#AA4499',
    'gray': '#888888',
    'black': '#000000',
}


def plot_training_losses(bulk_losses: List[float], dft_results: List[dict],
                        output_path: Optional[Path] = None,
                        n_iter_lbfgs: int = 0) -> plt.Figure:
    """
    Plot training loss curves.
    
    Parameters
    ----------
    bulk_losses : List[float]
        Bulk training losses
    dft_results : List[dict]
        DFT fine-tuning results
    output_path : Path, optional
        Save path
    n_iter_lbfgs : int
        Number of L-BFGS iterations (to mark transition)
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Phase 1: Bulk training
    ax = axes[0]
    if bulk_losses:
        ax.semilogy(bulk_losses, color=COLORS['blue'], lw=2.5)
        
        # Mark L-BFGS transition if applicable
        n_adam = len(bulk_losses) - n_iter_lbfgs
        if n_iter_lbfgs > 0 and n_adam > 0:
            ax.axvline(x=n_adam, color=COLORS['red'], linestyle='--', 
                      lw=1.5, label='L-BFGS start')
            ax.legend()
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Bulk Loss')
    ax.set_title('(a) Bulk Thermodynamic Training', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Phase 2: DFT fine-tuning
    ax = axes[1]
    if dft_results:
        iters = [r['iter'] for r in dft_results]
        losses = [r['loss'] for r in dft_results]
        ax.semilogy(iters, losses, color=COLORS['orange'], lw=2.5, 
                   marker='o', markersize=4)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('DFT Loss')
    ax.set_title('(b) DFT Fine-tuning', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def plot_learned_parameters(network, eta_range: np.ndarray = None,
                           output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot learned A(η), B(η) parameters.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Trained network
    eta_range : array, optional
        Packing fractions to evaluate
    output_path : Path, optional
        Save path
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    if eta_range is None:
        eta_range = np.linspace(0.05, 0.50, 100)
    
    A_vals, B_vals = [], []
    for eta in eta_range:
        A, B = network.from_eta(eta)
        A_vals.append(float(A))
        B_vals.append(float(B))
    
    A_vals = np.array(A_vals)
    B_vals = np.array(B_vals)
    C_vals = 8 * A_vals + 2 * B_vals - 9
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # A(η)
    ax = axes[0]
    ax.plot(eta_range, A_vals, color=COLORS['blue'], lw=2.5, label='Learned')
    ax.axhline(y=1.3, color=COLORS['green'], linestyle='--', lw=2, label='Optimal (1.3)')
    ax.axhline(y=1.0, color=COLORS['orange'], linestyle=':', lw=2, label='Lutsko (1.0)')
    ax.axhline(y=1.5, color=COLORS['purple'], linestyle='-.', lw=2, label='Rosenfeld (1.5)')
    ax.fill_between(eta_range, 0.5, 2.0, alpha=0.1, color=COLORS['blue'])
    ax.set_xlabel('η')
    ax.set_ylabel('A')
    ax.set_title('(a) Parameter A(η)', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.50)
    
    # B(η)
    ax = axes[1]
    ax.plot(eta_range, B_vals, color=COLORS['blue'], lw=2.5, label='Learned')
    ax.axhline(y=-1.0, color=COLORS['green'], linestyle='--', lw=2, label='Optimal (-1.0)')
    ax.axhline(y=0.0, color=COLORS['orange'], linestyle=':', lw=2, label='Lutsko (0.0)')
    ax.fill_between(eta_range, -2.0, 1.0, alpha=0.1, color=COLORS['blue'])
    ax.set_xlabel('η')
    ax.set_ylabel('B')
    ax.set_title('(b) Parameter B(η)', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.50)
    
    # Constraint C = 8A + 2B - 9
    ax = axes[2]
    ax.plot(eta_range, C_vals, color=COLORS['blue'], lw=2.5, label='Learned')
    ax.axhline(y=0, color=COLORS['black'], linestyle='-', lw=1.5, label='PY (C=0)')
    ax.axhline(y=-0.6, color=COLORS['green'], linestyle='--', lw=2, label='Optimal (C=-0.6)')
    ax.axhline(y=-3.0, color=COLORS['red'], linestyle=':', lw=2, label='CS (C=-3)')
    ax.fill_between(eta_range, -4.0, 1.0, alpha=0.1, color=COLORS['green'], label='Valid range')
    ax.set_xlabel('η')
    ax.set_ylabel('C = 8A + 2B - 9')
    ax.set_title('(c) Constraint Value', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(-5, 2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def plot_thermodynamics(network, eta_range: np.ndarray = None,
                       output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot thermodynamic quantities: Z, μ_ex, χ_T.
    """
    from core.thermodynamics import BulkThermodynamics

    if eta_range is None:
        eta_range = np.linspace(0.05, 0.50, 100)
    
    # Reference data
    Z_CS = np.array([float(BulkThermodynamics.Z_CS(e)) for e in eta_range])
    Z_PY = np.array([float(BulkThermodynamics.Z_PY(e)) for e in eta_range])
    mu_CS = np.array([float(BulkThermodynamics.mu_ex_CS(e)) for e in eta_range])
    mu_RF = np.array([float(BulkThermodynamics.mu_ex_RF(e)) for e in eta_range])
    chi_RF = np.array([float(BulkThermodynamics.chi_T_RF(e)) for e in eta_range])
    
    # Learned data
    Z_learned, mu_learned, chi_learned = [], [], []
    for eta in eta_range:
        A, B = network.from_eta(eta)
        Z_learned.append(float(BulkThermodynamics.Z_lutsko(eta, A, B)))
        mu_learned.append(float(BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B)))
        chi_learned.append(float(BulkThermodynamics.chi_T_bulk_lutsko(eta, A, B)))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Z
    ax = axes[0]
    ax.plot(eta_range, Z_CS, color=COLORS['black'], lw=3, label='CS (target)')
    ax.plot(eta_range, Z_PY, color=COLORS['gray'], linestyle='--', lw=2, label='PY')
    ax.plot(eta_range, Z_learned, color=COLORS['blue'], lw=2.5, label='Learned')
    ax.set_xlabel('η')
    ax.set_ylabel('Z = βP/ρ')
    ax.set_title('(a) Equation of State', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # μ_ex
    ax = axes[1]
    ax.plot(eta_range, mu_CS, color=COLORS['black'], lw=3, label='CS (target)')
    ax.plot(eta_range, mu_RF, color=COLORS['gray'], linestyle='--', lw=2, label='RF')
    ax.plot(eta_range, mu_learned, color=COLORS['blue'], lw=2.5, label='Learned')
    ax.set_xlabel('η')
    ax.set_ylabel('βμ_ex')
    ax.set_title('(b) Excess Chemical Potential', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # χ_T
    ax = axes[2]
    ax.semilogy(eta_range, chi_RF, color=COLORS['gray'], linestyle='--', lw=2, label='RF')
    ax.semilogy(eta_range, chi_learned, color=COLORS['blue'], lw=2.5, label='Learned')
    ax.set_xlabel('η')
    ax.set_ylabel('χ_T')
    ax.set_title('(c) Isothermal Compressibility', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def plot_parameter_space(output_path: Optional[Path] = None,
                        network=None) -> plt.Figure:
    """
    Plot parameter space (A, B) with constraint contours.
    """
    from core.thermodynamics import BulkThermodynamics

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Grid for contour
    A_range = np.linspace(0.5, 2.0, 100)
    B_range = np.linspace(-2.0, 1.0, 100)
    A_grid, B_grid = np.meshgrid(A_range, B_range)
    C_grid = 8*A_grid + 2*B_grid - 9
    
    # Contour plot
    contour = ax.contourf(A_grid, B_grid, C_grid, levels=20, cmap='RdYlBu_r')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('C = 8A + 2B - 9')
    
    # PY line (C = 0)
    A_py = np.linspace(0.5, 2.0, 50)
    B_py = (9 - 8*A_py) / 2
    ax.plot(A_py, B_py, 'k-', lw=2.5, label='PY line (C=0)')
    
    # Mark special points
    markers = {
        'Rosenfeld': (1.5, 0.0, 's', COLORS['purple']),
        'Lutsko': (1.0, 0.0, 'o', COLORS['orange']),
        'Optimal': (1.3, -1.0, '*', COLORS['green']),
        'White Bear': (1.125, -1.125, '^', COLORS['red']),
    }
    
    for name, (A, B, marker, color) in markers.items():
        ax.plot(A, B, marker, markersize=12, color=color,
               markeredgecolor='black', markeredgewidth=1.5, label=name)
    
    # Learned trajectory (if network provided)
    if network is not None:
        eta_range = np.linspace(0.1, 0.48, 50)
        A_learned, B_learned = [], []
        for eta in eta_range:
            A, B = network.from_eta(eta)
            A_learned.append(float(A))
            B_learned.append(float(B))
        ax.plot(A_learned, B_learned, color=COLORS['blue'], lw=2.5,
               label='Learned A(η), B(η)')
        ax.scatter(A_learned[0], B_learned[0], color=COLORS['blue'], s=80, 
                  marker='<', label='η=0.1')
        ax.scatter(A_learned[-1], B_learned[-1], color=COLORS['blue'], s=80,
                  marker='>', label='η=0.48')
    
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_title('Lutsko Parameter Space', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0.5, 2.0)
    ax.set_ylim(-2.0, 1.0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")
    
    return fig


def create_publication_figure(network, bulk_losses: List[float],
                             dft_results: List[dict],
                             output_dir: Path) -> Dict[str, plt.Figure]:
    """
    Create all publication-quality figures.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Trained network
    bulk_losses : List[float]
        Bulk training losses
    dft_results : List[dict]
        DFT results
    output_dir : Path
        Output directory
    
    Returns
    -------
    figures : dict
        Dictionary of figure objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    figures = {}
    
    # Training losses
    figures['losses'] = plot_training_losses(
        bulk_losses, dft_results,
        output_path=output_dir / 'training_losses.png'
    )
    
    # Learned parameters
    figures['parameters'] = plot_learned_parameters(
        network,
        output_path=output_dir / 'learned_parameters.png'
    )
    
    # Thermodynamics
    figures['thermodynamics'] = plot_thermodynamics(
        network,
        output_path=output_dir / 'thermodynamics.png'
    )
    
    # Parameter space
    figures['parameter_space'] = plot_parameter_space(
        output_path=output_dir / 'parameter_space.png',
        network=network
    )
    
    plt.close('all')
    
    return figures
