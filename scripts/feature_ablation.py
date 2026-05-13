#!/usr/bin/env python
"""
B3: Feature ablation study for neural network inputs.

Tests how the choice of input features affects the learned parameters.

Feature sets:
  1-feature:  (eta)
  3-feature:  (eta, eta^2, eta^3)
  5-feature:  (eta, eta^2, eta^3, eta/(1-eta), ln(1-eta))  [default]

Uses Approach 1 (CS EOS) as the training objective.

Usage:
    python -m cnfmt.scripts.feature_ablation
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from scripts.paper_figure_style import apply_paper_style
apply_paper_style()

from neural.network import ConditionalNetwork
from core.thermodynamics import BulkThermodynamics as BT

Z_CS = BT.Z_CS
Z_lutsko = BT.Z_lutsko
mu_ex_CS = BT.mu_ex_CS
mu_ex_lutsko = BT.mu_ex_bulk_lutsko


def train_with_features(n_features, n_iter=500, lr=3e-3, seed=42):
    """Train a network with a given number of input features."""
    key = jax.random.PRNGKey(seed)
    net = ConditionalNetwork(key, n_features=n_features)

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    eta_train = jnp.linspace(0.05, 0.45, 15)

    @eqx.filter_value_and_grad
    def loss_fn(net):
        total = 0.0
        for eta in eta_train:
            A, B = net.from_eta(eta)
            Z_pred = Z_lutsko(eta, A, B)
            mu_pred = mu_ex_lutsko(eta, A, B)
            Z_target = Z_CS(eta)
            mu_target = mu_ex_CS(eta)
            total += (Z_pred - Z_target)**2 + 0.1 * (mu_pred - mu_target)**2
        return total / len(eta_train)

    losses = []
    for i in range(n_iter):
        loss, grads = loss_fn(net)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(net, eqx.is_array))
        net = eqx.apply_updates(net, updates)
        losses.append(float(loss))
        if i % 100 == 0:
            print(f"  [{n_features}f] Iter {i}: loss={float(loss):.4e}")

    return net, losses


def main():
    print("="*60)
    print("FEATURE ABLATION STUDY")
    print("="*60)

    feature_configs = {
        r'1 feature ($\eta$)': 1,
        r'3 features ($\eta, \eta^2, \eta^3$)': 3,
        r'5 features (full)': 5,
    }

    results = {}
    for label, n_feat in feature_configs.items():
        print(f"\nTraining with {n_feat} feature(s)...")
        net, losses = train_with_features(n_feat)
        results[label] = (net, losses, n_feat)
        print(f"  Final loss: {losses[-1]:.4e}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ['C0', 'C1', 'C3']
    etas = np.linspace(0.01, 0.5, 100)

    # (a) Training convergence
    ax = axes[0, 0]
    for (label, (net, losses, _)), c in zip(results.items(), colors):
        ax.semilogy(losses, color=c, lw=1.5, label=label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training convergence')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # (b) Learned A(eta)
    ax = axes[0, 1]
    for (label, (net, _, _)), c in zip(results.items(), colors):
        A_vals = np.array([float(net.from_eta(e)[0]) for e in etas])
        ax.plot(etas, A_vals, color=c, lw=1.5, label=label)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Lutsko')
    ax.axhline(1.3, color='brown', ls=':', alpha=0.5, label='Gül et al.')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$A(\eta)$')
    ax.set_title(r'(b) Learned parameter $A$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (c) Learned B(eta)
    ax = axes[1, 0]
    for (label, (net, _, _)), c in zip(results.items(), colors):
        B_vals = np.array([float(net.from_eta(e)[1]) for e in etas])
        ax.plot(etas, B_vals, color=c, lw=1.5, label=label)
    ax.axhline(0.0, color='gray', ls='--', alpha=0.5, label='Lutsko')
    ax.axhline(-1.0, color='brown', ls=':', alpha=0.5, label='Gül et al.')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$B(\eta)$')
    ax.set_title(r'(c) Learned parameter $B$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # (d) Constraint C = 8A + 2B - 9
    ax = axes[1, 1]
    for (label, (net, _, _)), c in zip(results.items(), colors):
        C_vals = np.array([float(8*net.from_eta(e)[0] + 2*net.from_eta(e)[1] - 9)
                           for e in etas])
        ax.plot(etas, C_vals, color=c, lw=1.5, label=label)
    ax.axhline(0, color='gray', ls='--', alpha=0.5, label='PY line')
    ax.axhline(-3, color='gray', ls=':', alpha=0.5, label='CS')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$C = 8A + 2B - 9$')
    ax.set_title('(d) Constraint parameter')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    path = out / 'feature_ablation.png'
    plt.savefig(path)
    print(f"\nSaved: {path}")
    plt.close()

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Features':>25s}  {'Final Loss':>12s}  {'A(0.3)':>8s}  {'B(0.3)':>8s}  {'C(0.3)':>8s}")
    print("-"*70)
    for label, (net, losses, n_feat) in results.items():
        A, B = net.from_eta(0.3)
        A, B = float(A), float(B)
        C = 8*A + 2*B - 9
        print(f"{label:>25s}  {losses[-1]:12.4e}  {A:8.4f}  {B:8.4f}  {C:8.4f}")


if __name__ == '__main__':
    main()
