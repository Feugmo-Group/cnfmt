#!/usr/bin/env python
"""
Compare Four Training Approaches for A(eta), B(eta)
====================================================

This script compares four training approaches for learning the
density-dependent Lutsko parameters A(eta) and B(eta):

Approach 1 (CS EOS): Match Carnahan-Starling bulk thermodynamics
    Loss = |Z_Lut - Z_CS|^2 + |mu_Lut - mu_CS|^2

Approach 2 (delta_mu, delta_chi): Minimize DFT-bulk deviations (Gul Eq. 28)
    Loss = delta_mu^2 + delta_chi^2

Approach 3 (Contact): Optimize contact density at hard wall
    Loss = |rho_contact^DFT - rho_contact^exact|^2

Approach 4 (Combined): Combined loss
    Loss = L_EOS + lambda_contact * L_contact

Generates 9-panel comparison figure.

Usage:
    python -m cnfmt.scripts.fast_four_approaches          # Full run (500 iters)
    python -m cnfmt.scripts.fast_four_approaches --fast    # Quick run (100 iters)
"""

import argparse

import jax
import jax.numpy as jnp
from jax import value_and_grad
import optax
import equinox as eqx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_figure_style import apply_paper_style
apply_paper_style()

from core.thermodynamics import BulkThermodynamics as BT
from neural.network import ConditionalNetwork

Z_CS = BT.Z_CS
mu_ex_CS = BT.mu_ex_CS
chi_T_CS = BT.chi_T_CS
Z_lutsko = BT.Z_lutsko
mu_ex_lutsko = BT.mu_ex_bulk_lutsko
chi_T_lutsko = BT.chi_T_bulk_lutsko

PI = jnp.pi



def train_approach(eta_values, key_seed, loss_type, n_iter=500, lr=3e-3, verbose=True):
    """Train with specified loss type.

    Parameters
    ----------
    loss_type : int
        1 = CS EOS, 2 = delta_mu/delta_chi, 3 = Contact, 4 = Combined
    """
    key = jax.random.PRNGKey(key_seed)
    network = ConditionalNetwork(key)

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(net):
        total = 0.0
        for eta in eta_values:
            A, B = net.from_eta(eta)
            Z_pred = Z_lutsko(eta, A, B)
            mu_pred = mu_ex_lutsko(eta, A, B)
            chi_pred = chi_T_lutsko(eta, A, B)
            Z_target = Z_CS(eta)
            mu_target = mu_ex_CS(eta)
            chi_target = chi_T_CS(eta)

            if loss_type == 1:  # CS EOS
                total += (Z_pred - Z_target)**2 + 0.1 * (mu_pred - mu_target)**2
            elif loss_type == 2:  # delta_mu, delta_chi
                delta_mu = (mu_pred - mu_target) / (jnp.abs(mu_target) + 0.1)
                delta_chi = (chi_pred - chi_target) / (jnp.abs(chi_target) + 0.01)
                total += delta_mu**2 + delta_chi**2
            elif loss_type == 3:  # Contact
                total += ((Z_pred - Z_target) / Z_target)**2
            else:  # Combined
                loss_eos = (Z_pred - Z_target)**2 + 0.1 * (mu_pred - mu_target)**2
                loss_contact = ((Z_pred - Z_target) / Z_target)**2
                total += loss_eos + 0.5 * loss_contact
        return total / len(eta_values)

    log_interval = 50 if n_iter <= 100 else 200
    losses = []
    for i in range(n_iter):
        loss, grads = loss_fn(network)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array))
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        if verbose and i % log_interval == 0:
            A, B = network.from_eta(0.3)
            print(f"  Iter {i}: loss={float(loss):.4e}, A={float(A):.3f}, B={float(B):.3f}")

    return network, losses


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

    # (a) Training Convergence
    ax = axes[0, 0]
    ax.semilogy(hist1, color=c1, lw=2, label='Approach 1 (CS EOS)')
    ax.semilogy(hist2, color=c2, lw=2, label='Approach 2 (delta_mu,delta_chi)')
    ax.semilogy(hist3, color=c3, lw=2, label='Approach 3 (Contact)')
    ax.semilogy(hist4, color=c4, lw=2, label='Approach 4 (Combined)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training Convergence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Learned Parameter A
    ax = axes[0, 1]
    ax.plot(etas, A1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, A2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, A3, color=c3, lw=2, label='Approach 3')
    ax.plot(etas, A4, color=c4, lw=2, label='Approach 4')
    ax.axhline(1.5, color='gray', ls='--', alpha=0.7, label='Rosenfeld')
    ax.axhline(1.0, color='orange', ls='--', alpha=0.7, label='Lutsko')
    ax.axhline(1.3, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.set_xlabel('eta')
    ax.set_ylabel('A(eta)')
    ax.set_title('(b) Learned Parameter A')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)

    # (c) Learned Parameter B
    ax = axes[0, 2]
    ax.plot(etas, B1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, B2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, B3, color=c3, lw=2, label='Approach 3')
    ax.plot(etas, B4, color=c4, lw=2, label='Approach 4')
    ax.axhline(0.0, color='gray', ls='--', alpha=0.7, label='Rosenfeld/Lutsko')
    ax.axhline(-1.0, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.axhline(0.0, color='cyan', ls='--', alpha=0.5, label='White Bear')
    ax.set_xlabel('eta')
    ax.set_ylabel('B(eta)')
    ax.set_title('(c) Learned Parameter B')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)

    # (d) Constraint Parameter C
    ax = axes[1, 0]
    C1 = 8*A1 + 2*B1 - 9
    C2 = 8*A2 + 2*B2 - 9
    C3 = 8*A3 + 2*B3 - 9
    C4 = 8*A4 + 2*B4 - 9

    ax.plot(etas, C1, color=c1, lw=2, label='Approach 1')
    ax.plot(etas, C2, color=c2, lw=2, label='Approach 2')
    ax.plot(etas, C3, color=c3, lw=2, label='Approach 3')
    ax.plot(etas, C4, color=c4, lw=2, label='Approach 4')
    ax.axhline(3.0, color='gray', ls='--', alpha=0.7, label='PY (C=3)')
    ax.axhline(0.0, color='gray', ls='-', alpha=0.5)
    ax.axhline(-1, color='orange', ls='--', alpha=0.7, label='Lutsko')
    ax.axhline(-0.6, color='brown', ls=':', lw=2, label='Gül et al.')
    ax.set_xlabel('eta')
    ax.set_ylabel('C = 8A + 2B - 9')
    ax.set_title('(d) Constraint Parameter')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([-2, 4])
    ax.grid(True, alpha=0.3)

    # (e) Compressibility Factor Z
    ax = axes[1, 1]
    Z_cs = np.array([float(Z_CS(e)) for e in etas])
    Z1 = np.array([float(Z_lutsko(e, A1[i], B1[i])) for i, e in enumerate(etas)])
    Z2 = np.array([float(Z_lutsko(e, A2[i], B2[i])) for i, e in enumerate(etas)])
    Z3 = np.array([float(Z_lutsko(e, A3[i], B3[i])) for i, e in enumerate(etas)])
    Z4 = np.array([float(Z_lutsko(e, A4[i], B4[i])) for i, e in enumerate(etas)])

    ax.plot(etas, Z_cs, 'k-', lw=2.5, label='CS (reference)')
    ax.plot(etas, Z1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, Z2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, Z3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, Z4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('eta')
    ax.set_ylabel('Z')
    ax.set_title('(e) Compressibility Factor')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)

    # (f) Excess Chemical Potential
    ax = axes[1, 2]
    mu_cs = np.array([float(mu_ex_CS(e)) for e in etas])
    mu1 = np.array([float(mu_ex_lutsko(e, A1[i], B1[i])) for i, e in enumerate(etas)])
    mu2 = np.array([float(mu_ex_lutsko(e, A2[i], B2[i])) for i, e in enumerate(etas)])
    mu3 = np.array([float(mu_ex_lutsko(e, A3[i], B3[i])) for i, e in enumerate(etas)])
    mu4 = np.array([float(mu_ex_lutsko(e, A4[i], B4[i])) for i, e in enumerate(etas)])

    ax.plot(etas, mu_cs, 'k-', lw=2.5, label='CS (reference)')
    ax.plot(etas, mu1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, mu2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, mu3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, mu4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('eta')
    ax.set_ylabel(r'$\beta\mu_{ex}$')
    ax.set_title('(f) Excess Chemical Potential')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)

    # (g) Isothermal Compressibility
    ax = axes[2, 0]
    chi_cs = np.array([float(chi_T_CS(e)) for e in etas])
    chi1 = np.array([float(chi_T_lutsko(e, A1[i], B1[i])) for i, e in enumerate(etas)])
    chi2 = np.array([float(chi_T_lutsko(e, A2[i], B2[i])) for i, e in enumerate(etas)])
    chi3 = np.array([float(chi_T_lutsko(e, A3[i], B3[i])) for i, e in enumerate(etas)])
    chi4 = np.array([float(chi_T_lutsko(e, A4[i], B4[i])) for i, e in enumerate(etas)])

    ax.plot(etas, chi_cs, 'k-', lw=2.5, label='CS (reference)')
    ax.plot(etas, chi1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, chi2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, chi3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, chi4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('eta')
    ax.set_ylabel(r'$\chi_T / \chi_T^{id}$')
    ax.set_title('(g) Isothermal Compressibility')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # (h) Contact Density
    ax = axes[2, 1]
    rho_bulk = np.array([6*e/PI for e in etas])
    contact_cs = rho_bulk * Z_cs
    contact1 = rho_bulk * Z1
    contact2 = rho_bulk * Z2
    contact3 = rho_bulk * Z3
    contact4 = rho_bulk * Z4

    ax.plot(etas, contact_cs, 'k-', lw=2.5, label='CS (reference)')
    ax.plot(etas, contact1, color=c1, ls='--', lw=2, label='Approach 1')
    ax.plot(etas, contact2, color=c2, ls='--', lw=2, label='Approach 2')
    ax.plot(etas, contact3, color=c3, ls='--', lw=2, label='Approach 3')
    ax.plot(etas, contact4, color=c4, ls='--', lw=2, label='Approach 4')
    ax.set_xlabel('eta')
    ax.set_ylabel(r'$\rho(R^+)\sigma^3$')
    ax.set_title('(h) Contact Density')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 0.5])
    ax.grid(True, alpha=0.3)

    # (i) Parameter Space Trajectories
    ax = axes[2, 2]

    A_grid = np.linspace(0.5, 2.0, 100)
    B_grid = np.linspace(-2.0, 0.5, 100)
    A_mesh, B_mesh = np.meshgrid(A_grid, B_grid)
    C_mesh = 8*A_mesh + 2*B_mesh - 9

    levels = np.linspace(-4, 4, 17)
    cmap = plt.cm.RdYlBu_r
    cf = ax.contourf(A_mesh, B_mesh, C_mesh, levels=levels, cmap=cmap, alpha=0.8)
    plt.colorbar(cf, ax=ax, label='C')

    A_py = np.linspace(0.5, 2.0, 50)
    B_py = (9 - 8*A_py) / 2
    ax.plot(A_py, B_py, 'k--', lw=2, label='PY line')

    ax.plot(A1, B1, color=c1, lw=2, label='Approach 1')
    ax.plot(A2, B2, color=c2, lw=2, label='Approach 2')
    ax.plot(A3, B3, color=c3, lw=2, label='Approach 3')
    ax.plot(A4, B4, color=c4, lw=2, label='Approach 4')

    ax.plot(1.0, 0.0, 'o', color='orange', ms=10, mew=2, mfc='orange', label='Lutsko')
    ax.plot(1.3, -1.0, '*', color='brown', ms=12, mew=2, label='Gül et al.')
    ax.plot(1.5, 0.0, 's', color='gray', ms=8, mew=2, mfc='gray', label='Rosenfeld')

    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_title('(i) Parameter Space Trajectories')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim([0.5, 2.0])
    ax.set_ylim([-2.0, 0.5])
    ax.grid(True, alpha=0.3)

    plt.suptitle('Comparison of Four Training Approaches for A(eta), B(eta)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare four training approaches for A(eta), B(eta)')
    parser.add_argument('--fast', action='store_true',
                        help='Quick run with 100 iterations (default: 500)')
    parser.add_argument('--n-iter', type=int, default=None,
                        help='Override iteration count (default: 100 if --fast, 500 otherwise)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 5e-3 if --fast, 3e-3 otherwise)')
    args = parser.parse_args()

    if args.n_iter is not None:
        n_iter = args.n_iter
    else:
        n_iter = 100 if args.fast else 500

    if args.lr is not None:
        lr = args.lr
    else:
        lr = 5e-3 if args.fast else 3e-3

    mode_label = f"FAST ({n_iter} iters)" if args.fast else f"FULL ({n_iter} iters)"
    print("="*70)
    print(f"FOUR APPROACHES COMPARISON  [{mode_label}]")
    print("="*70)

    eta_values = jnp.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])

    print("\n[Approach 1: CS EOS]")
    net1, hist1 = train_approach(eta_values, 42, loss_type=1, n_iter=n_iter, lr=lr)

    print("\n[Approach 2: delta_mu, delta_chi]")
    net2, hist2 = train_approach(eta_values, 43, loss_type=2, n_iter=n_iter, lr=lr)

    print("\n[Approach 3: Contact]")
    net3, hist3 = train_approach(eta_values, 44, loss_type=3, n_iter=n_iter, lr=lr)

    print("\n[Approach 4: Combined]")
    net4, hist4 = train_approach(eta_values, 45, loss_type=4, n_iter=n_iter, lr=lr)

    results = {
        'approach1': (net1, hist1),
        'approach2': (net2, hist2),
        'approach3': (net3, hist3),
        'approach4': (net4, hist4),
    }

    plot_comparison(results, 'outputs/four_approaches_comparison.png')

    print("\n" + "="*70)
    print("SUMMARY: Learned Parameters at eta = 0.3")
    print("="*70)
    for name, (net, _) in results.items():
        A, B = net.from_eta(0.3)
        C = 8*float(A) + 2*float(B) - 9
        print(f"{name}: A={float(A):.3f}, B={float(B):.3f}, C={C:.3f}")

    print("\nReference values:")
    print("  Rosenfeld: A=1.5, B=0.0, C=3.0")
    print("  Lutsko:    A=1.0, B=0.0, C=-1.0")
    print("  Gul et al: A=1.3, B=-1.0, C=-0.6")

    print("\nDone!")


if __name__ == "__main__":
    main()
