#!/usr/bin/env python3
"""
Train neural network on bulk thermodynamics (Z, mu, chi).

Modes:
    train (default): Train CNN with regularized loss, plot 6-panel diagnostics,
                     and save model checkpoint.
    compare:         Train CNN (no regularization) and plot 6-panel comparison
                     of CNN vs fixed Optimized (A=1.3, B=-1.0) with error panels.

Usage:
    python -m cnfmt.scripts.train_bulk [--epochs N]
    python -m cnfmt.scripts.train_bulk --compare [--epochs N]
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from core.thermodynamics import BulkThermodynamics as BT
from neural.network import ConditionalNetwork

Z_CS = BT.Z_CS
Z_PY = BT.Z_PY
Z_lutsko = BT.Z_lutsko
mu_ex_CS = BT.mu_ex_CS
mu_ex_RF = BT.mu_ex_RF
mu_ex_lutsko = BT.mu_ex_bulk_lutsko
chi_RF = BT.chi_T_RF
chi_lutsko = BT.chi_T_bulk_lutsko

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Training functions
# ============================================================================

def train(network, eta_train, n_epochs=500, lr=0.01):
    """Train with regularized bulk loss (Z + mu + parameter priors)."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    def loss_fn(net, eta):
        A, B = net.from_eta(eta)
        loss_Z = ((Z_lutsko(eta, A, B) - Z_CS(eta)) / Z_CS(eta))**2
        loss_mu = ((mu_ex_lutsko(eta, A, B) - mu_ex_CS(eta)) / (jnp.abs(mu_ex_CS(eta)) + 0.1))**2
        return loss_Z + loss_mu + 0.01*(A-1)**2 + 0.01*B**2
    @eqx.filter_value_and_grad
    def batch_loss(net):
        return jnp.mean(jnp.array([loss_fn(net, eta) for eta in eta_train]))
    losses = []
    for epoch in range(n_epochs):
        loss_val, grads = batch_loss(network)
        updates, opt_state = optimizer.update(eqx.filter(grads, eqx.is_array), opt_state)
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss_val))
        if epoch % 100 == 0: print(f"Epoch {epoch}: loss={losses[-1]:.4e}")
    return network, losses


def train_compare(n_epochs=500):
    """Train CNN without regularization (for compare mode)."""
    print("Training CNN...")
    eta_train = jnp.linspace(0.05, 0.45, 20)
    network = ConditionalNetwork(jax.random.PRNGKey(42))
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    def loss_fn(net, eta):
        A, B = net.from_eta(eta)
        return ((Z_lutsko(eta, A, B) - Z_CS(eta)) / Z_CS(eta))**2 + ((mu_ex_lutsko(eta, A, B) - mu_ex_CS(eta)) / (jnp.abs(mu_ex_CS(eta)) + 0.1))**2
    @eqx.filter_value_and_grad
    def batch_loss(net): return jnp.mean(jnp.array([loss_fn(net, eta) for eta in eta_train]))
    for epoch in range(n_epochs):
        loss_val, grads = batch_loss(network)
        updates, opt_state = optimizer.update(eqx.filter(grads, eqx.is_array), opt_state)
        network = eqx.apply_updates(network, updates)
        if epoch % 100 == 0: print(f"  Epoch {epoch}: loss = {loss_val:.4e}")
    return network


# ============================================================================
# Plotting functions
# ============================================================================

def plot_results(network, losses, eta_range):
    """Plot 6-panel training diagnostics (train mode)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].semilogy(losses); axes[0,0].set_title('(a) Loss')
    A_vals = [float(network.from_eta(e)[0]) for e in eta_range]
    B_vals = [float(network.from_eta(e)[1]) for e in eta_range]
    axes[0,1].plot(eta_range, A_vals, 'b-', label='A CNN'); axes[0,1].plot(eta_range, B_vals, 'r-', label='B CNN')
    axes[0,1].axhline(1.3, ls='--', color='b', alpha=0.5, label='A Opt'); axes[0,1].axhline(-1.0, ls='--', color='r', alpha=0.5, label='B Opt')
    axes[0,1].legend(); axes[0,1].set_title('(b) A(eta), B(eta)')
    C_vals = [8*float(network.from_eta(e)[0])+2*float(network.from_eta(e)[1])-9 for e in eta_range]
    axes[0,2].plot(eta_range, C_vals, 'g-', label='C CNN'); axes[0,2].axhline(-0.6, ls='--', label='C Opt')
    axes[0,2].legend(); axes[0,2].set_title('(c) C(eta)')
    Z_cs = [float(Z_CS(e)) for e in eta_range]; Z_cnn = [float(Z_lutsko(e, *network.from_eta(e))) for e in eta_range]
    Z_opt = [float(Z_lutsko(e, 1.3, -1.0)) for e in eta_range]
    axes[1,0].plot(eta_range, Z_cs, 'k-', label='CS'); axes[1,0].plot(eta_range, Z_cnn, 'b--', label='CNN')
    axes[1,0].plot(eta_range, Z_opt, 'r:', label='Opt'); axes[1,0].legend(); axes[1,0].set_title('(d) Z')
    mu_cs = [float(mu_ex_CS(e)) for e in eta_range]; mu_cnn = [float(mu_ex_lutsko(e, *network.from_eta(e))) for e in eta_range]
    mu_opt = [float(mu_ex_lutsko(e, 1.3, -1.0)) for e in eta_range]
    axes[1,1].plot(eta_range, mu_cs, 'k-', label='CS'); axes[1,1].plot(eta_range, mu_cnn, 'b--', label='CNN')
    axes[1,1].plot(eta_range, mu_opt, 'r:', label='Opt'); axes[1,1].legend(); axes[1,1].set_title('(e) mu_ex')
    chi_cnn = [float(chi_lutsko(e, *network.from_eta(e))) for e in eta_range]; chi_opt = [float(chi_lutsko(e, 1.3, -1.0)) for e in eta_range]
    axes[1,2].semilogy(eta_range, chi_cnn, 'b--', label='CNN'); axes[1,2].semilogy(eta_range, chi_opt, 'r:', label='Opt')
    axes[1,2].legend(); axes[1,2].set_title('(f) chi')
    for ax in axes.flat: ax.grid(True, alpha=0.3)
    plt.suptitle('Bulk Thermodynamics: CNN vs Optimized', fontweight='bold')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/'bulk_thermodynamics.png', dpi=150); plt.close()
    print(f"Saved: {OUTPUT_DIR/'bulk_thermodynamics.png'}")


def plot_comparison(network):
    """Plot 6-panel CNN vs Optimized comparison with error panels (compare mode)."""
    print("Generating comparison plot...")
    eta = np.linspace(0.01, 0.5, 100)
    A_OPT, B_OPT = 1.3, -1.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    A_cnn = [float(network.from_eta(e)[0]) for e in eta]
    B_cnn = [float(network.from_eta(e)[1]) for e in eta]
    axes[0,0].plot(eta, A_cnn, 'b-', lw=2, label='A(eta) CNN')
    axes[0,0].plot(eta, B_cnn, 'r-', lw=2, label='B(eta) CNN')
    axes[0,0].axhline(A_OPT, ls='--', color='b', alpha=0.7, label='A Opt')
    axes[0,0].axhline(B_OPT, ls='--', color='r', alpha=0.7, label='B Opt')
    axes[0,0].legend(); axes[0,0].set_title('(a) Parameters'); axes[0,0].grid(True, alpha=0.3)

    C_cnn = [8*A_cnn[i] + 2*B_cnn[i] - 9 for i in range(len(eta))]
    axes[0,1].plot(eta, C_cnn, 'g-', lw=2, label='C(eta) CNN')
    axes[0,1].axhline(-0.6, ls='--', color='g', label='C Opt')
    axes[0,1].axhline(-1.0, ls=':', color='orange', label='C Lutsko')
    axes[0,1].legend(); axes[0,1].set_title('(b) C = 8A+2B-9'); axes[0,1].grid(True, alpha=0.3)

    Z_cs = [float(Z_CS(e)) for e in eta]
    Z_cnn = [float(Z_lutsko(e, *network.from_eta(e))) for e in eta]
    Z_opt = [float(Z_lutsko(e, A_OPT, B_OPT)) for e in eta]
    axes[0,2].plot(eta, Z_cs, 'k-', lw=2, label='CS')
    axes[0,2].plot(eta, Z_cnn, 'b--', lw=2, label='CNN')
    axes[0,2].plot(eta, Z_opt, 'r:', lw=2, label='Opt')
    axes[0,2].legend(); axes[0,2].set_title('(c) Z'); axes[0,2].grid(True, alpha=0.3)

    mu_cs = [float(mu_ex_CS(e)) for e in eta]
    mu_cnn = [float(mu_ex_lutsko(e, *network.from_eta(e))) for e in eta]
    mu_opt = [float(mu_ex_lutsko(e, A_OPT, B_OPT)) for e in eta]
    axes[1,0].plot(eta, mu_cs, 'k-', lw=2, label='CS')
    axes[1,0].plot(eta, mu_cnn, 'b--', lw=2, label='CNN')
    axes[1,0].plot(eta, mu_opt, 'r:', lw=2, label='Opt')
    axes[1,0].legend(); axes[1,0].set_title('(d) mu_ex'); axes[1,0].grid(True, alpha=0.3)

    err_Z_cnn = [abs(Z_cnn[i]-Z_cs[i])/Z_cs[i]*100 for i in range(len(eta))]
    err_Z_opt = [abs(Z_opt[i]-Z_cs[i])/Z_cs[i]*100 for i in range(len(eta))]
    axes[1,1].semilogy(eta, err_Z_cnn, 'b-', lw=2, label='CNN')
    axes[1,1].semilogy(eta, err_Z_opt, 'r--', lw=2, label='Opt')
    axes[1,1].legend(); axes[1,1].set_title('(e) Z Error %'); axes[1,1].grid(True, alpha=0.3)

    err_mu_cnn = [abs(mu_cnn[i]-mu_cs[i])/(abs(mu_cs[i])+0.1)*100 for i in range(len(eta))]
    err_mu_opt = [abs(mu_opt[i]-mu_cs[i])/(abs(mu_cs[i])+0.1)*100 for i in range(len(eta))]
    axes[1,2].semilogy(eta, err_mu_cnn, 'b-', lw=2, label='CNN')
    axes[1,2].semilogy(eta, err_mu_opt, 'r--', lw=2, label='Opt')
    axes[1,2].legend(); axes[1,2].set_title('(f) mu Error %'); axes[1,2].grid(True, alpha=0.3)

    plt.suptitle('Optimized (A=1.3, B=-1.0) vs CNN (A(eta), B(eta))', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'optimized_vs_cnn.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'optimized_vs_cnn.png'}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train CNN on bulk thermodynamics or compare methods')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--compare', action='store_true',
                        help='Run CNN vs Optimized comparison instead of standard training')
    args = parser.parse_args()

    if args.compare:
        print("="*60 + "\nCompare: Optimized vs CNN\n" + "="*60)
        network = train_compare(args.epochs)
        plot_comparison(network)
    else:
        print("="*60 + "\nTraining CNN on Bulk Thermodynamics\n" + "="*60)
        eta_train = jnp.linspace(0.05, 0.45, 20)
        network = ConditionalNetwork(jax.random.PRNGKey(42))
        network, losses = train(network, eta_train, n_epochs=args.epochs)
        plot_results(network, losses, np.linspace(0.01, 0.5, 100))
        eqx.tree_serialise_leaves(OUTPUT_DIR/'network_bulk.eqx', network)

    print("COMPLETE")

if __name__ == "__main__": main()
