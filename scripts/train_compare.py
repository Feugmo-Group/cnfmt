#!/usr/bin/env python
"""
Compare Training Approaches for A(η), B(η)
==========================================

This script compares three training approaches for learning the
density-dependent Lutsko parameters A(η) and B(η):

Approach 1 (CS EOS): Match Carnahan-Starling bulk thermodynamics
    Loss = |Z_Lut - Z_CS|² + |μ_Lut - μ_CS|²

Approach 2 (Gül Eq. 28): Minimize DFT-bulk deviations
    Loss = δ_μ² + δ_χ²

Approach 3 (Contact): Optimize contact density at hard wall
    Loss = |ρ_contact^DFT - ρ_contact^exact|²

Usage
-----
    python -m cnfmt.scripts.train_compare --n_iter 500 --output_dir outputs

Author: Computational Materials Science
"""

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.flatten_util import ravel_pytree
import optax
import equinox as eqx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Dict

from neural.network import ConditionalNetwork
from core.thermodynamics import BulkThermodynamics
from training.config import TrainingConfig
from training.losses import (
    compute_bulk_loss, 
    compute_contact_loss,
    compute_combined_loss
)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_approach1_cs_eos(config: TrainingConfig, eta_values: jnp.ndarray,
                           n_iter: int = 500, verbose: bool = True
                           ) -> Tuple[ConditionalNetwork, List[float]]:
    """
    Approach 1: Train on Carnahan-Starling EOS.
    
    Loss = |Z_Lut - Z_CS|² + |μ_Lut - μ_CS|²
    """
    if verbose:
        print("\n" + "="*60)
        print("APPROACH 1: Train on Carnahan-Starling EOS")
        print("Loss = |Z_Lut - Z_CS|² + |μ_Lut - μ_CS|²")
        print("="*60)
    
    key = jax.random.PRNGKey(42)
    network = ConditionalNetwork(key, config.n_features, config.hidden_dim, config.n_hidden)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=1e-4)
    )
    params = eqx.filter(network, eqx.is_array)
    opt_state = optimizer.init(params)
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        return compute_bulk_loss(net, eta_values, config)
    
    losses = []
    best_network = network
    best_loss = float('inf')
    
    if verbose:
        print(f"\n{'Iter':>6} {'Loss':>12} {'A(0.3)':>8} {'B(0.3)':>8} {'C(0.3)':>10}")
        print("-"*55)
    
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array)
        )
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_network = network
        
        if verbose and (i % 50 == 0 or i == n_iter - 1):
            A, B = network.from_eta(0.3)
            C = 8 * float(A) + 2 * float(B) - 9
            print(f"{i:6d} {float(loss):12.4e} {float(A):8.4f} {float(B):8.4f} {C:10.4f}")
    
    return best_network, losses


def train_approach2_delta_mu_chi(config: TrainingConfig, eta_values: jnp.ndarray,
                                  n_iter: int = 500, verbose: bool = True
                                  ) -> Tuple[ConditionalNetwork, List[float]]:
    """
    Approach 2: Minimize δ_μ² + δ_χ² (Gül et al. Eq. 28).
    
    δ_μ = (μ_ex^DFT - μ_ex^bulk) / |μ_ex^bulk|
    δ_χ = (χ_T^DFT - χ_T^bulk) / |χ_T^bulk|
    """
    if verbose:
        print("\n" + "="*60)
        print("APPROACH 2: Minimize δ_μ² + δ_χ² (Gül et al. Eq. 28)")
        print("δ_μ = (μ_DFT - μ_bulk)/|μ_bulk|, δ_χ = (χ_DFT - χ_bulk)/|χ_bulk|")
        print("="*60)
    
    key = jax.random.PRNGKey(42)
    network = ConditionalNetwork(key, config.n_features, config.hidden_dim, config.n_hidden)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=1e-4)
    )
    params = eqx.filter(network, eqx.is_array)
    opt_state = optimizer.init(params)
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        """Loss = Σ_η [δ_μ² + δ_χ²]"""
        total_loss = 0.0
        for eta in eta_values:
            A, B = net.from_eta(eta)
            
            # DFT values (Lutsko functional)
            mu_dft = BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B)
            chi_dft = BulkThermodynamics.chi_T_bulk_lutsko(eta, A, B)
            
            # Bulk reference (CS)
            mu_bulk = BulkThermodynamics.mu_ex_CS(eta)
            chi_bulk = BulkThermodynamics.chi_T_CS(eta)
            
            # Relative deviations (Eq. 28)
            delta_mu = (mu_dft - mu_bulk) / (jnp.abs(mu_bulk) + 0.1)
            delta_chi = (chi_dft - chi_bulk) / (jnp.abs(chi_bulk) + 0.01)
            
            total_loss += delta_mu**2 + delta_chi**2
        
        return total_loss / len(eta_values)
    
    losses = []
    best_network = network
    best_loss = float('inf')
    
    if verbose:
        print(f"\n{'Iter':>6} {'Loss':>12} {'A(0.3)':>8} {'B(0.3)':>8} {'δ_μ':>10} {'δ_χ':>10}")
        print("-"*65)
    
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array)
        )
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_network = network
        
        if verbose and (i % 50 == 0 or i == n_iter - 1):
            A, B = network.from_eta(0.3)
            mu_dft = BulkThermodynamics.mu_ex_bulk_lutsko(0.3, A, B)
            chi_dft = BulkThermodynamics.chi_T_bulk_lutsko(0.3, A, B)
            mu_bulk = BulkThermodynamics.mu_ex_CS(0.3)
            chi_bulk = BulkThermodynamics.chi_T_CS(0.3)
            delta_mu = float((mu_dft - mu_bulk) / (jnp.abs(mu_bulk) + 0.1))
            delta_chi = float((chi_dft - chi_bulk) / (jnp.abs(chi_bulk) + 0.01))
            print(f"{i:6d} {float(loss):12.4e} {float(A):8.4f} {float(B):8.4f} "
                  f"{delta_mu:10.4f} {delta_chi:10.4f}")
    
    return best_network, losses


def train_approach3_contact(config: TrainingConfig, eta_values: jnp.ndarray,
                            n_iter: int = 500, verbose: bool = True
                            ) -> Tuple[ConditionalNetwork, List[float]]:
    """
    Approach 3: Optimize contact density (test-particle).
    
    Loss = |ρ_contact^DFT - ρ_contact^exact|² / ρ_contact^exact²
    """
    if verbose:
        print("\n" + "="*60)
        print("APPROACH 3: Contact Density Optimization (Test-Particle)")
        print("Loss = |ρ_contact^DFT - ρ_contact^exact|² / ρ_contact^exact²")
        print("="*60)
    
    key = jax.random.PRNGKey(42)
    network = ConditionalNetwork(key, config.n_features, config.hidden_dim, config.n_hidden)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.learning_rate, weight_decay=1e-4)
    )
    params = eqx.filter(network, eqx.is_array)
    opt_state = optimizer.init(params)
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        return compute_contact_loss(net, eta_values, config)
    
    losses = []
    best_network = network
    best_loss = float('inf')
    
    if verbose:
        print(f"\n{'Iter':>6} {'Loss':>12} {'A(0.3)':>8} {'B(0.3)':>8} {'Error%':>10}")
        print("-"*55)
    
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array)
        )
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_network = network
        
        if verbose and (i % 50 == 0 or i == n_iter - 1):
            A, B = network.from_eta(0.3)
            rho_bulk = 0.3 / ((4.0/3.0) * np.pi * 0.5**3)
            Z_lut = BulkThermodynamics.Z_lutsko(0.3, A, B)
            Z_cs = BulkThermodynamics.Z_CS(0.3)
            error = abs(float(Z_lut - Z_cs) / float(Z_cs)) * 100
            print(f"{i:6d} {float(loss):12.4e} {float(A):8.4f} {float(B):8.4f} {error:10.2f}")
    
    return best_network, losses


# ============================================================================
# PLOTTING
# ============================================================================

def plot_comparison(results: Dict, output_dir: Path):
    """Plot comprehensive comparison of all approaches."""
    
    net1, hist1 = results['approach1']
    net2, hist2 = results['approach2']
    net3, hist3 = results['approach3']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    c1, c2, c3 = 'blue', 'red', 'green'
    etas = np.linspace(0.05, 0.5, 100)
    
    # (a) Training convergence
    ax = axes[0, 0]
    ax.semilogy(hist1, color=c1, lw=2, label='Approach 1 (CS EOS)')
    ax.semilogy(hist2, color=c2, lw=2, label='Approach 2 (δ_μ,δ_χ)')
    ax.semilogy(hist3, color=c3, lw=2, label='Approach 3 (Contact)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Get learned parameters
    A1 = [float(net1.from_eta(e)[0]) for e in etas]
    B1 = [float(net1.from_eta(e)[1]) for e in etas]
    A2 = [float(net2.from_eta(e)[0]) for e in etas]
    B2 = [float(net2.from_eta(e)[1]) for e in etas]
    A3 = [float(net3.from_eta(e)[0]) for e in etas]
    B3 = [float(net3.from_eta(e)[1]) for e in etas]
    
    # (b) Learned A(η)
    ax = axes[0, 1]
    ax.plot(etas, A1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, A2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, A3, color=c3, lw=2, label='Approach 3')
    ax.axhline(1.5, color='gray', ls='--', alpha=0.7, label='Rosenfeld')
    ax.axhline(1.0, color='orange', ls='--', alpha=0.7, label='Lutsko')
    ax.axhline(1.3, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.set_xlabel('η')
    ax.set_ylabel('A(η)')
    ax.set_title('(b) Learned Parameter A')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # (c) Learned B(η)
    ax = axes[0, 2]
    ax.plot(etas, B1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, B2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, B3, color=c3, lw=2, label='Approach 3')
    ax.axhline(0.0, color='gray', ls='--', alpha=0.7, label='Rosenfeld/Lutsko')
    ax.axhline(-1.0, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.set_xlabel('η')
    ax.set_ylabel('B(η)')
    ax.set_title('(c) Learned Parameter B')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # (d) Constraint C = 8A + 2B - 9
    ax = axes[1, 0]
    C1 = [8*A1[i] + 2*B1[i] - 9 for i in range(len(etas))]
    C2 = [8*A2[i] + 2*B2[i] - 9 for i in range(len(etas))]
    C3 = [8*A3[i] + 2*B3[i] - 9 for i in range(len(etas))]
    
    ax.plot(etas, C1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, C2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, C3, color=c3, lw=2, label='Approach 3')
    ax.axhline(0, color='black', ls='-', lw=1, label='PY line')
    ax.axhline(-1, color='orange', ls='--', alpha=0.7, label='Lutsko')
    ax.axhline(-0.6, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.set_xlabel('η')
    ax.set_ylabel('C = 8A + 2B - 9')
    ax.set_title('(d) Constraint Parameter')
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # (e) Z comparison
    ax = axes[1, 1]
    Z_cs = [float(BulkThermodynamics.Z_CS(e)) for e in etas]
    Z1 = [float(BulkThermodynamics.Z_lutsko(e, net1.from_eta(e)[0], net1.from_eta(e)[1])) for e in etas]
    Z2 = [float(BulkThermodynamics.Z_lutsko(e, net2.from_eta(e)[0], net2.from_eta(e)[1])) for e in etas]
    Z3 = [float(BulkThermodynamics.Z_lutsko(e, net3.from_eta(e)[0], net3.from_eta(e)[1])) for e in etas]
    
    ax.plot(etas, Z_cs, 'k-', lw=2.5, label='CS (exact)')
    ax.plot(etas, Z1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, Z2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, Z3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.set_xlabel('η')
    ax.set_ylabel('Z')
    ax.set_title('(e) Compressibility Factor')
    ax.legend(fontsize=7)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # (f) Contact density
    ax = axes[1, 2]
    rho_bulk = [e / ((4.0/3.0) * np.pi * 0.5**3) for e in etas]
    contact_cs = [rho_bulk[i] * Z_cs[i] for i in range(len(etas))]
    contact1 = [rho_bulk[i] * Z1[i] for i in range(len(etas))]
    contact2 = [rho_bulk[i] * Z2[i] for i in range(len(etas))]
    contact3 = [rho_bulk[i] * Z3[i] for i in range(len(etas))]
    
    ax.plot(etas, contact_cs, 'k-', lw=2.5, label='CS (exact)')
    ax.plot(etas, contact1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, contact2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, contact3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.set_xlabel('η')
    ax.set_ylabel('ρ(R⁺)σ³')
    ax.set_title('(f) Contact Density')
    ax.legend(fontsize=7)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Three Training Approaches for A(η), B(η)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_dir / 'three_approaches_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'three_approaches_comparison.png'}")
    plt.close()


def print_summary(results: Dict):
    """Print summary table of learned parameters."""
    
    net1, _ = results['approach1']
    net2, _ = results['approach2']
    net3, _ = results['approach3']
    
    print("\n" + "="*80)
    print("SUMMARY: Learned Parameters at Key Packing Fractions")
    print("="*80)
    print(f"\n{'η':>6} | {'A1':>8} {'B1':>8} {'C1':>8} | "
          f"{'A2':>8} {'B2':>8} {'C2':>8} | {'A3':>8} {'B3':>8} {'C3':>8}")
    print("-"*90)
    
    for eta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        A1, B1 = net1.from_eta(eta)
        A2, B2 = net2.from_eta(eta)
        A3, B3 = net3.from_eta(eta)
        C1 = 8*float(A1) + 2*float(B1) - 9
        C2 = 8*float(A2) + 2*float(B2) - 9
        C3 = 8*float(A3) + 2*float(B3) - 9
        print(f"{eta:6.2f} | {float(A1):8.4f} {float(B1):8.4f} {C1:8.4f} | "
              f"{float(A2):8.4f} {float(B2):8.4f} {C2:8.4f} | "
              f"{float(A3):8.4f} {float(B3):8.4f} {C3:8.4f}")
    
    print("\n" + "-"*90)
    print("Approach 1: CS EOS (Z, μ matching)")
    print("Approach 2: δ_μ² + δ_χ² (Gül et al. Eq. 28)")
    print("Approach 3: Contact density optimization (test-particle)")
    print("\nReference: Gül et al. (fixed): A = 1.3, B = -1.0, C = -0.6")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare Training Approaches for A(η), B(η)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--n_iter', type=int, default=500,
                       help='Training iterations per approach')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--n_hidden', type=int, default=4,
                       help='Number of hidden layers')
    parser.add_argument('--lr', type=float, default=3e-3,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COMPARING THREE TRAINING APPROACHES FOR A(η), B(η)")
    print("="*70)
    
    # Create config
    config = TrainingConfig(
        n_features=5,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        learning_rate=args.lr
    )
    
    eta_values = jnp.array(config.eta_train)
    
    # Train all approaches
    results = {}
    
    results['approach1'] = train_approach1_cs_eos(config, eta_values, args.n_iter)
    results['approach2'] = train_approach2_delta_mu_chi(config, eta_values, args.n_iter)
    results['approach3'] = train_approach3_contact(config, eta_values, args.n_iter)
    
    # Plot comparison
    plot_comparison(results, Path(args.output_dir))
    
    # Print summary
    print_summary(results)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
