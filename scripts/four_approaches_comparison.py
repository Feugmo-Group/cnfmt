#!/usr/bin/env python
"""
Compare Four Training Approaches for A(η), B(η)
================================================

This script compares four training approaches for learning the
density-dependent Lutsko parameters A(η) and B(η):

Approach 1 (CS EOS): Match Carnahan-Starling bulk thermodynamics
    Loss = |Z_Lut - Z_CS|² + |μ_Lut - μ_CS|²

Approach 2 (δ_μ, δ_χ): Minimize DFT-bulk deviations (Gül Eq. 28)
    Loss = δ_μ² + δ_χ²

Approach 3 (Contact): Optimize contact density at hard wall
    Loss = |ρ_contact^DFT - ρ_contact^exact|²

Approach 4 (Combined): Combined loss
    Loss = L_EOS + λ_contact * L_contact

Generates 9-panel comparison figure.

Author: Computational Materials Science
"""

import jax
import jax.numpy as jnp
from jax import value_and_grad
import optax
import equinox as eqx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Tuple, Dict

jax.config.update("jax_enable_x64", True)

PI = jnp.pi


# ============================================================================
# SIMPLE NETWORK (no external dependencies)
# ============================================================================

class SimpleNetwork(eqx.Module):
    """Simple MLP for A(η), B(η)."""
    layers: list
    
    def __init__(self, key, n_features=5, hidden_dim=32, n_hidden=2):
        keys = jax.random.split(key, n_hidden + 2)
        
        layers = []
        # Input layer
        layers.append(eqx.nn.Linear(n_features, hidden_dim, key=keys[0]))
        # Hidden layers
        for i in range(n_hidden):
            layers.append(eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i+1]))
        # Output layer (2 outputs: A, B)
        layers.append(eqx.nn.Linear(hidden_dim, 2, key=keys[-1]))
        
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)
    
    def from_eta(self, eta):
        """Get A, B from packing fraction."""
        # Create feature vector
        eta = jnp.atleast_1d(eta)
        features = jnp.array([eta[0], eta[0]**2, eta[0]**3, 
                              jnp.log(1 - eta[0] + 1e-10), 
                              1.0 / (1 - eta[0] + 1e-10)])
        out = self(features)
        # Constrain outputs
        A = 0.8 + 0.7 * jax.nn.sigmoid(out[0])  # A in [0.8, 1.5]
        B = -1.5 + 1.5 * jax.nn.sigmoid(out[1])  # B in [-1.5, 0]
        return A, B


# ============================================================================
# THERMODYNAMICS
# ============================================================================

def Z_CS(eta):
    """Carnahan-Starling compressibility factor."""
    return (1 + eta + eta**2 - eta**3) / (1 - eta)**3

def mu_ex_CS(eta):
    """CS excess chemical potential."""
    return (8*eta - 9*eta**2 + 3*eta**3) / (1 - eta)**3

def chi_T_CS(eta):
    """CS isothermal compressibility (relative to ideal)."""
    return (1 - eta)**4 / (1 + 4*eta + 4*eta**2 - 4*eta**3 + eta**4)

def Z_lutsko(eta, A, B):
    """Lutsko compressibility factor."""
    C = 8*A + 2*B - 9
    return (1 + eta + eta**2 - eta**3 + C*eta**3/3) / (1 - eta)**3

def mu_ex_lutsko(eta, A, B):
    """Lutsko excess chemical potential."""
    C = 8*A + 2*B - 9
    # From integration of Z
    base = (8*eta - 9*eta**2 + 3*eta**3) / (1 - eta)**3
    correction = C * eta**2 * (3 - eta) / (3 * (1 - eta)**3)
    return base + correction

@jax.jit
def chi_T_lutsko(eta, A, B):
    """Lutsko isothermal compressibility."""
    C = 8*A + 2*B - 9
    # Numerical derivative
    eps = 1e-5
    Z_plus = Z_lutsko(eta + eps, A, B)
    Z_minus = Z_lutsko(eta - eps, A, B)
    dZ_deta = (Z_plus - Z_minus) / (2*eps)
    return 1.0 / (Z_lutsko(eta, A, B) + eta * dZ_deta)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_approach1(eta_values, n_iter=500, lr=3e-3, verbose=True):
    """Approach 1: CS EOS matching."""
    if verbose:
        print("\n[Approach 1: CS EOS]")
    
    key = jax.random.PRNGKey(42)
    network = SimpleNetwork(key)
    
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        total = 0.0
        for eta in eta_values:
            A, B = net.from_eta(eta)
            Z_pred = Z_lutsko(eta, A, B)
            mu_pred = mu_ex_lutsko(eta, A, B)
            Z_target = Z_CS(eta)
            mu_target = mu_ex_CS(eta)
            total += (Z_pred - Z_target)**2 + 0.1 * (mu_pred - mu_target)**2
        return total / len(eta_values)
    
    losses = []
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array))
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        if verbose and i % 200 == 0:
            A, B = network.from_eta(0.3)
            print(f"  Iter {i}: loss={float(loss):.4e}, A={float(A):.3f}, B={float(B):.3f}")
    
    return network, losses


def train_approach2(eta_values, n_iter=500, lr=3e-3, verbose=True):
    """Approach 2: δ_μ, δ_χ minimization."""
    if verbose:
        print("\n[Approach 2: δ_μ, δ_χ]")
    
    key = jax.random.PRNGKey(43)
    network = SimpleNetwork(key)
    
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        total = 0.0
        for eta in eta_values:
            A, B = net.from_eta(eta)
            mu_dft = mu_ex_lutsko(eta, A, B)
            chi_dft = chi_T_lutsko(eta, A, B)
            mu_bulk = mu_ex_CS(eta)
            chi_bulk = chi_T_CS(eta)
            delta_mu = (mu_dft - mu_bulk) / (jnp.abs(mu_bulk) + 0.1)
            delta_chi = (chi_dft - chi_bulk) / (jnp.abs(chi_bulk) + 0.01)
            total += delta_mu**2 + delta_chi**2
        return total / len(eta_values)
    
    losses = []
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array))
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        if verbose and i % 200 == 0:
            A, B = network.from_eta(0.3)
            print(f"  Iter {i}: loss={float(loss):.4e}, A={float(A):.3f}, B={float(B):.3f}")
    
    return network, losses


def train_approach3(eta_values, n_iter=500, lr=3e-3, verbose=True):
    """Approach 3: Contact density optimization."""
    if verbose:
        print("\n[Approach 3: Contact]")
    
    key = jax.random.PRNGKey(44)
    network = SimpleNetwork(key)
    
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        total = 0.0
        for eta in eta_values:
            A, B = net.from_eta(eta)
            # Contact density = ρ_bulk * Z
            Z_pred = Z_lutsko(eta, A, B)
            Z_exact = Z_CS(eta)
            total += ((Z_pred - Z_exact) / Z_exact)**2
        return total / len(eta_values)
    
    losses = []
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array))
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        if verbose and i % 200 == 0:
            A, B = network.from_eta(0.3)
            print(f"  Iter {i}: loss={float(loss):.4e}, A={float(A):.3f}, B={float(B):.3f}")
    
    return network, losses


def train_approach4(eta_values, n_iter=500, lr=3e-3, verbose=True):
    """Approach 4: Combined loss."""
    if verbose:
        print("\n[Approach 4: Combined]")
    
    key = jax.random.PRNGKey(45)
    network = SimpleNetwork(key)
    
    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        total = 0.0
        for eta in eta_values:
            A, B = net.from_eta(eta)
            # EOS loss
            Z_pred = Z_lutsko(eta, A, B)
            mu_pred = mu_ex_lutsko(eta, A, B)
            Z_target = Z_CS(eta)
            mu_target = mu_ex_CS(eta)
            loss_eos = (Z_pred - Z_target)**2 + 0.1 * (mu_pred - mu_target)**2
            # Contact loss
            loss_contact = ((Z_pred - Z_target) / Z_target)**2
            # Combined
            total += loss_eos + 0.5 * loss_contact
        return total / len(eta_values)
    
    losses = []
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array))
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        if verbose and i % 200 == 0:
            A, B = network.from_eta(0.3)
            print(f"  Iter {i}: loss={float(loss):.4e}, A={float(A):.3f}, B={float(B):.3f}")
    
    return network, losses


# ============================================================================
# PLOTTING (9 panels)
# ============================================================================

def plot_comparison(results: Dict, output_path: str):
    """Plot 9-panel comparison of all four approaches."""
    
    net1, hist1 = results['approach1']
    net2, hist2 = results['approach2']
    net3, hist3 = results['approach3']
    net4, hist4 = results['approach4']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    
    c1, c2, c3, c4 = 'blue', 'red', 'green', 'purple'
    etas = np.linspace(0.01, 0.5, 100)
    
    # Get learned parameters for all approaches
    A1 = np.array([float(net1.from_eta(e)[0]) for e in etas])
    B1 = np.array([float(net1.from_eta(e)[1]) for e in etas])
    A2 = np.array([float(net2.from_eta(e)[0]) for e in etas])
    B2 = np.array([float(net2.from_eta(e)[1]) for e in etas])
    A3 = np.array([float(net3.from_eta(e)[0]) for e in etas])
    B3 = np.array([float(net3.from_eta(e)[1]) for e in etas])
    A4 = np.array([float(net4.from_eta(e)[0]) for e in etas])
    B4 = np.array([float(net4.from_eta(e)[1]) for e in etas])
    
    # ===== (a) Training Convergence =====
    ax = axes[0, 0]
    ax.semilogy(hist1, color=c1, lw=2, label='Approach 1 (CS EOS)')
    ax.semilogy(hist2, color=c2, lw=2, label='Approach 2 (δ_μ,δ_χ)')
    ax.semilogy(hist3, color=c3, lw=2, label='Approach 3 (Contact)')
    ax.semilogy(hist4, color=c4, lw=2, label='Approach 4 (Combined)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training Convergence')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # ===== (b) Learned Parameter A =====
    ax = axes[0, 1]
    ax.plot(etas, A1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, A2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, A3, color=c3, lw=2, label='Approach 3')
    ax.plot(etas, A4, color=c4, lw=2, label='Approach 4')
    ax.axhline(1.5, color='gray', ls='--', alpha=0.7, label='Rosenfeld')
    ax.axhline(1.0, color='orange', ls='--', alpha=0.7, label='Lutsko')
    ax.axhline(1.3, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.set_xlabel('η')
    ax.set_ylabel('A(η)')
    ax.set_title('(b) Learned Parameter A')
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # ===== (c) Learned Parameter B =====
    ax = axes[0, 2]
    ax.plot(etas, B1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, B2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, B3, color=c3, lw=2, label='Approach 3')
    ax.plot(etas, B4, color=c4, lw=2, label='Approach 4')
    ax.axhline(0.0, color='gray', ls='--', alpha=0.7, label='Rosenfeld/Lutsko')
    ax.axhline(-1.0, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.axhline(0.0, color='cyan', ls='--', alpha=0.5, label='White Bear')
    ax.set_xlabel('η')
    ax.set_ylabel('B(η)')
    ax.set_title('(c) Learned Parameter B')
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # ===== (d) Constraint Parameter C =====
    ax = axes[1, 0]
    C1 = 8*A1 + 2*B1 - 9
    C2 = 8*A2 + 2*B2 - 9
    C3 = 8*A3 + 2*B3 - 9
    C4 = 8*A4 + 2*B4 - 9
    
    ax.plot(etas, C1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, C2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, C3, color=c3, lw=2, label='Approach 3')
    ax.plot(etas, C4, color=c4, lw=2, label='Approach 4')
    ax.axhline(3.0, color='gray', ls='--', alpha=0.7, label='PY line (C=0)')
    ax.axhline(0.0, color='gray', ls='-', alpha=0.5)
    ax.axhline(-1, color='orange', ls='--', alpha=0.7, label='Lutsko')
    ax.axhline(-0.6, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.set_xlabel('η')
    ax.set_ylabel('C = 8A + 2B - 9')
    ax.set_title('(d) Constraint Parameter')
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([-2, 4])
    ax.grid(True, alpha=0.3)
    
    # ===== (e) Compressibility Factor Z =====
    ax = axes[1, 1]
    Z_cs = np.array([float(Z_CS(e)) for e in etas])
    Z1 = np.array([float(Z_lutsko(e, A1[i], B1[i])) for i, e in enumerate(etas)])
    Z2 = np.array([float(Z_lutsko(e, A2[i], B2[i])) for i, e in enumerate(etas)])
    Z3 = np.array([float(Z_lutsko(e, A3[i], B3[i])) for i, e in enumerate(etas)])
    Z4 = np.array([float(Z_lutsko(e, A4[i], B4[i])) for i, e in enumerate(etas)])
    
    ax.plot(etas, Z_cs, 'k-', lw=2.5, label='CS (exact)')
    ax.plot(etas, Z1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, Z2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, Z3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, Z4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('η')
    ax.set_ylabel('Z')
    ax.set_title('(e) Compressibility Factor')
    ax.legend(fontsize=7)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # ===== (f) Excess Chemical Potential =====
    ax = axes[1, 2]
    mu_cs = np.array([float(mu_ex_CS(e)) for e in etas])
    mu1 = np.array([float(mu_ex_lutsko(e, A1[i], B1[i])) for i, e in enumerate(etas)])
    mu2 = np.array([float(mu_ex_lutsko(e, A2[i], B2[i])) for i, e in enumerate(etas)])
    mu3 = np.array([float(mu_ex_lutsko(e, A3[i], B3[i])) for i, e in enumerate(etas)])
    mu4 = np.array([float(mu_ex_lutsko(e, A4[i], B4[i])) for i, e in enumerate(etas)])
    
    ax.plot(etas, mu_cs, 'k-', lw=2.5, label='CS (exact)')
    ax.plot(etas, mu1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, mu2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, mu3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, mu4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('η')
    ax.set_ylabel(r'$\beta\mu_{ex}$')
    ax.set_title('(f) Excess Chemical Potential')
    ax.legend(fontsize=7)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # ===== (g) Isothermal Compressibility =====
    ax = axes[2, 0]
    chi_cs = np.array([float(chi_T_CS(e)) for e in etas])
    chi1 = np.array([float(chi_T_lutsko(e, A1[i], B1[i])) for i, e in enumerate(etas)])
    chi2 = np.array([float(chi_T_lutsko(e, A2[i], B2[i])) for i, e in enumerate(etas)])
    chi3 = np.array([float(chi_T_lutsko(e, A3[i], B3[i])) for i, e in enumerate(etas)])
    chi4 = np.array([float(chi_T_lutsko(e, A4[i], B4[i])) for i, e in enumerate(etas)])
    
    ax.plot(etas, chi_cs, 'k-', lw=2.5, label='CS (exact)')
    ax.plot(etas, chi1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, chi2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, chi3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, chi4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('η')
    ax.set_ylabel(r'$\chi_T / \chi_T^{id}$')
    ax.set_title('(g) Isothermal Compressibility')
    ax.legend(fontsize=7)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # ===== (h) Contact Density =====
    ax = axes[2, 1]
    rho_bulk = np.array([6*e/PI for e in etas])  # ρσ³ = 6η/π
    contact_cs = rho_bulk * Z_cs
    contact1 = rho_bulk * Z1
    contact2 = rho_bulk * Z2
    contact3 = rho_bulk * Z3
    contact4 = rho_bulk * Z4
    
    ax.plot(etas, contact_cs, 'k-', lw=2.5, label='CS (exact)')
    ax.plot(etas, contact1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, contact2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, contact3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, contact4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('η')
    ax.set_ylabel(r'$\rho(R^+)\sigma^3$')
    ax.set_title('(h) Contact Density')
    ax.legend(fontsize=7)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)
    
    # ===== (i) Parameter Space Trajectories =====
    ax = axes[2, 2]
    
    # Create colormap for C values
    A_grid = np.linspace(0.5, 2.0, 100)
    B_grid = np.linspace(-2.0, 0.5, 100)
    A_mesh, B_mesh = np.meshgrid(A_grid, B_grid)
    C_mesh = 8*A_mesh + 2*B_mesh - 9
    
    # Plot C contour
    levels = np.linspace(-4, 4, 17)
    cmap = plt.cm.RdYlBu_r
    cf = ax.contourf(A_mesh, B_mesh, C_mesh, levels=levels, cmap=cmap, alpha=0.8)
    plt.colorbar(cf, ax=ax, label='C')
    
    # Plot PY line (C = 0)
    A_py = np.linspace(0.5, 2.0, 50)
    B_py = (9 - 8*A_py) / 2
    ax.plot(A_py, B_py, 'k--', lw=2, label='PY line')
    
    # Plot trajectories
    ax.plot(A1, B1, color=c1, lw=2, label='Approach 1')
    ax.plot(A2, B2, color=c2, lw=2, label='Approach 2')
    ax.plot(A3, B3, color=c3, lw=2, label='Approach 3')
    ax.plot(A4, B4, color=c4, lw=2, label='Approach 4')
    
    # Mark reference points
    ax.plot(1.0, 0.0, 'o', color='orange', ms=10, mew=2, mfc='orange', label='Lutsko')
    ax.plot(1.3, -1.0, '*', color='brown', ms=12, mew=2, label='Gül et al.')
    ax.plot(1.5, 0.0, 's', color='gray', ms=8, mew=2, mfc='gray', label='Rosenfeld')
    
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_title('(i) Parameter Space Trajectories')
    ax.legend(fontsize=6, loc='upper left')
    ax.set_xlim([0.5, 2.0])
    ax.set_ylim([-2.0, 0.5])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Four Training Approaches for A(η), B(η)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("COMPARING FOUR TRAINING APPROACHES FOR A(η), B(η)")
    print("="*70)
    
    eta_values = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
    n_iter = 500
    
    results = {}
    results['approach1'] = train_approach1(eta_values, n_iter=n_iter)
    results['approach2'] = train_approach2(eta_values, n_iter=n_iter)
    results['approach3'] = train_approach3(eta_values, n_iter=n_iter)
    results['approach4'] = train_approach4(eta_values, n_iter=n_iter)
    
    # Plot
    plot_comparison(results, '/mnt/user-data/outputs/four_approaches_comparison.png')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Learned Parameters at η = 0.3")
    print("="*70)
    for name, (net, _) in results.items():
        A, B = net.from_eta(0.3)
        C = 8*float(A) + 2*float(B) - 9
        print(f"{name}: A={float(A):.3f}, B={float(B):.3f}, C={C:.3f}")
    
    print("\nReference values:")
    print("  Rosenfeld: A=1.5, B=0.0, C=3.0")
    print("  Lutsko:    A=1.0, B=0.0, C=-1.0")
    print("  Gül et al: A=1.3, B=-1.0, C=-0.6")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
