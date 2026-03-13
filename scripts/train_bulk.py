#!/usr/bin/env python3
"""
Train neural network on bulk thermodynamics (Z, μ, χ).
Usage: python -m cnfmt.scripts.train_bulk [--epochs N]
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def Z_CS(eta):
    return (1 + eta + eta**2 - eta**3) / (1 - eta)**3

def Z_PY(eta):
    return (1 + eta + eta**2) / (1 - eta)**3

def Z_lutsko(eta, A, B):
    C = 8*A + 2*B - 9
    return Z_PY(eta) + C * eta**2 / (3 * (1 - eta)**3)

def mu_ex_CS(eta):
    return eta * (8 - 9*eta + 3*eta**2) / (1 - eta)**3

def mu_ex_RF(eta):
    return -jnp.log(1 - eta) + eta * (14 - 13*eta + 5*eta**2) / (2 * (1 - eta)**3)

def mu_ex_lutsko(eta, A, B):
    C = 8*A + 2*B - 9
    return mu_ex_RF(eta) + C * eta**2 * (3 - eta) / (6 * (1 - eta)**3)

def chi_RF(eta):
    return (1 - eta)**4 / (1 + 2*eta)**2

def chi_lutsko(eta, A, B):
    C = 8*A + 2*B - 9
    return chi_RF(eta) * (1 + 2*eta)**2 / ((1 + 2*eta)**2 - C * eta**2)

class ABNetwork(eqx.Module):
    layers: list
    def __init__(self, hidden=[32, 32], key=None):
        if key is None: key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, len(hidden) + 1)
        dims = [1] + hidden + [2]
        self.layers = [eqx.nn.Linear(d_in, d_out, key=k) for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)]
    def __call__(self, eta):
        x = jnp.atleast_1d(eta)
        for layer in self.layers[:-1]: x = jnp.tanh(layer(x))
        out = self.layers[-1](x)
        return 0.8 + 1.0*jax.nn.sigmoid(out[0]), -1.5 + 2.0*jax.nn.sigmoid(out[1])

def train(network, eta_train, n_epochs=500, lr=0.01):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    def loss_fn(net, eta):
        A, B = net(eta)
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

def plot_results(network, losses, eta_range):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].semilogy(losses); axes[0,0].set_title('(a) Loss')
    A_vals = [float(network(e)[0]) for e in eta_range]
    B_vals = [float(network(e)[1]) for e in eta_range]
    axes[0,1].plot(eta_range, A_vals, 'b-', label='A CNN'); axes[0,1].plot(eta_range, B_vals, 'r-', label='B CNN')
    axes[0,1].axhline(1.3, ls='--', color='b', alpha=0.5, label='A Opt'); axes[0,1].axhline(-1.0, ls='--', color='r', alpha=0.5, label='B Opt')
    axes[0,1].legend(); axes[0,1].set_title('(b) A(η), B(η)')
    C_vals = [8*float(network(e)[0])+2*float(network(e)[1])-9 for e in eta_range]
    axes[0,2].plot(eta_range, C_vals, 'g-', label='C CNN'); axes[0,2].axhline(-0.6, ls='--', label='C Opt')
    axes[0,2].legend(); axes[0,2].set_title('(c) C(η)')
    Z_cs = [float(Z_CS(e)) for e in eta_range]; Z_cnn = [float(Z_lutsko(e, *network(e))) for e in eta_range]
    Z_opt = [float(Z_lutsko(e, 1.3, -1.0)) for e in eta_range]
    axes[1,0].plot(eta_range, Z_cs, 'k-', label='CS'); axes[1,0].plot(eta_range, Z_cnn, 'b--', label='CNN')
    axes[1,0].plot(eta_range, Z_opt, 'r:', label='Opt'); axes[1,0].legend(); axes[1,0].set_title('(d) Z')
    mu_cs = [float(mu_ex_CS(e)) for e in eta_range]; mu_cnn = [float(mu_ex_lutsko(e, *network(e))) for e in eta_range]
    mu_opt = [float(mu_ex_lutsko(e, 1.3, -1.0)) for e in eta_range]
    axes[1,1].plot(eta_range, mu_cs, 'k-', label='CS'); axes[1,1].plot(eta_range, mu_cnn, 'b--', label='CNN')
    axes[1,1].plot(eta_range, mu_opt, 'r:', label='Opt'); axes[1,1].legend(); axes[1,1].set_title('(e) μ_ex')
    chi_cnn = [float(chi_lutsko(e, *network(e))) for e in eta_range]; chi_opt = [float(chi_lutsko(e, 1.3, -1.0)) for e in eta_range]
    axes[1,2].semilogy(eta_range, chi_cnn, 'b--', label='CNN'); axes[1,2].semilogy(eta_range, chi_opt, 'r:', label='Opt')
    axes[1,2].legend(); axes[1,2].set_title('(f) χ')
    for ax in axes.flat: ax.grid(True, alpha=0.3)
    plt.suptitle('Bulk Thermodynamics: CNN vs Optimized', fontweight='bold')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/'bulk_thermodynamics.png', dpi=150); plt.close()
    print(f"Saved: {OUTPUT_DIR/'bulk_thermodynamics.png'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()
    print("="*60 + "\nTraining CNN on Bulk Thermodynamics\n" + "="*60)
    eta_train = jnp.linspace(0.05, 0.45, 20)
    network = ABNetwork(key=jax.random.PRNGKey(42))
    network, losses = train(network, eta_train, n_epochs=args.epochs)
    plot_results(network, losses, np.linspace(0.01, 0.5, 100))
    eqx.tree_serialise_leaves(OUTPUT_DIR/'network_bulk.eqx', network)
    print("COMPLETE")

if __name__ == "__main__": main()
