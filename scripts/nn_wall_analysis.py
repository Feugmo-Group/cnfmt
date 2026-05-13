#!/usr/bin/env python
"""
Analysis of two-phase training: bulk EOS preservation, parameter curves,
convergence, and finite-difference sensitivity.

Produces figures for the paper:
  1. Bulk EOS error before/after Phase 2
  2. A(eta), B(eta), C(eta) curves before/after Phase 2
  3. Wall loss convergence curve
  4. Finite-difference step sensitivity

Usage:
    python scripts/nn_wall_analysis.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_figure_style import apply_paper_style
apply_paper_style()

from solvers.fmt_1d_wbii_tensor import WallSolver, esFMT_Tensor
from neural.network import ConditionalNetwork
from core.thermodynamics import BulkThermodynamics as BT

PI = np.pi


def train_phase1(n_iter=1500, lr=3e-3, seed=42):
    """Phase 1: Bulk EOS training."""
    print("Phase 1: Bulk EOS training...")
    key = jax.random.PRNGKey(seed)
    net = ConditionalNetwork(key, n_features=5, hidden_dim=32, n_hidden=2,
                             A_bounds=(0.8, 1.5), B_bounds=(-1.5, 0.0))

    schedule = optax.cosine_decay_schedule(lr, n_iter)
    optimizer = optax.adamw(schedule)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    eta_train = jnp.linspace(0.05, 0.50, 20)

    def single_loss(net, eta):
        A, B = net.from_eta(eta)
        Z_pred = BT.Z_lutsko(eta, A, B)
        mu_pred = BT.mu_ex_bulk_lutsko(eta, A, B)
        Z_target = BT.Z_CS(eta)
        mu_target = BT.mu_ex_CS(eta)
        return (Z_pred - Z_target)**2 + 0.5 * (mu_pred - mu_target)**2

    @eqx.filter_value_and_grad
    def loss_fn(net):
        losses = jax.vmap(lambda eta: single_loss(net, eta))(eta_train)
        return jnp.mean(losses)

    @eqx.filter_jit
    def step(net, opt_state):
        loss, grads = loss_fn(net)
        updates, new_opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(net, eqx.is_array))
        new_net = eqx.apply_updates(net, updates)
        return new_net, new_opt_state, loss

    print("  Compiling JIT...")
    for i in range(n_iter):
        net, opt_state, loss = step(net, opt_state)
        if i % 500 == 0:
            print(f"  Iter {i}: loss={float(loss):.4e}")

    print(f"  Phase 1 final: loss={float(loss):.4e}")
    return net


def train_phase2_with_history(net, solver, n_iter=20, lr=5e-4, dA=0.01, dB=0.01):
    """Phase 2 with full loss history for convergence plotting."""
    print(f"\nPhase 2: Wall fine-tuning (dA={dA}, dB={dB})...")

    eta_targets = [0.367, 0.393, 0.449]
    md_contacts = {0.367: 5.36, 0.393: 6.15, 0.449: 8.34}

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    def get_solver_contact(A_val, B_val, eta):
        func = esFMT_Tensor(A=A_val, B=B_val)
        result = solver.solve(eta, func, max_iter=4000, tol=1e-7, verbose=False)
        return float(result['contact'])

    history = []  # (iteration, wall_loss, A, B, C)

    for iteration in range(n_iter):
        total_loss = 0.0
        grad_accum = {'dL_dA': {}, 'dL_dB': {}}

        for eta in eta_targets:
            A, B = net.from_eta(eta)
            A_val, B_val = float(A), float(B)
            md_contact = md_contacts[eta]

            contact_0 = get_solver_contact(A_val, B_val, eta)
            loss_0 = ((contact_0 - md_contact) / md_contact) ** 2
            total_loss += loss_0

            contact_pA = get_solver_contact(min(A_val + dA, 1.5), B_val, eta)
            loss_pA = ((contact_pA - md_contact) / md_contact) ** 2
            dL_dA = (loss_pA - loss_0) / dA

            contact_pB = get_solver_contact(A_val, min(B_val + dB, 0.0), eta)
            loss_pB = ((contact_pB - md_contact) / md_contact) ** 2
            dL_dB = (loss_pB - loss_0) / dB

            grad_accum['dL_dA'][eta] = dL_dA
            grad_accum['dL_dB'][eta] = dL_dB

        total_loss /= len(eta_targets)

        @eqx.filter_value_and_grad
        def surrogate_loss(net):
            total = 0.0
            for eta in eta_targets:
                A, B = net.from_eta(eta)
                total += grad_accum['dL_dA'][eta] * A + grad_accum['dL_dB'][eta] * B
            return total / len(eta_targets)

        _, grads = surrogate_loss(net)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(net, eqx.is_array))
        net = eqx.apply_updates(net, updates)

        A, B = net.from_eta(0.367)
        A_f, B_f = float(A), float(B)
        C_f = 8*A_f + 2*B_f - 9
        history.append((iteration, total_loss, A_f, B_f, C_f))

        if iteration % 5 == 0 or iteration == n_iter - 1:
            print(f"  Iter {iteration}: wall_loss={total_loss:.4e}, "
                  f"A={A_f:.4f}, B={B_f:.4f}, C={C_f:.4f}")

    print(f"  Phase 2 final: wall_loss={total_loss:.4e}")
    return net, history


def compute_bulk_eos_errors(net, label=""):
    """Compute Z and mu_ex errors vs Carnahan-Starling at benchmark etas."""
    etas = np.linspace(0.05, 0.50, 50)
    errors = {'eta': [], 'Z_pred': [], 'Z_cs': [], 'Z_err_pct': [],
              'mu_pred': [], 'mu_cs': [], 'mu_err_pct': []}

    for eta in etas:
        A, B = net.from_eta(eta)
        Z_pred = float(BT.Z_lutsko(eta, A, B))
        Z_cs = float(BT.Z_CS(eta))
        mu_pred = float(BT.mu_ex_bulk_lutsko(eta, A, B))
        mu_cs = float(BT.mu_ex_CS(eta))

        errors['eta'].append(eta)
        errors['Z_pred'].append(Z_pred)
        errors['Z_cs'].append(Z_cs)
        errors['Z_err_pct'].append(100 * (Z_pred - Z_cs) / Z_cs)
        errors['mu_pred'].append(mu_pred)
        errors['mu_cs'].append(mu_cs)
        errors['mu_err_pct'].append(100 * (mu_pred - mu_cs) / mu_cs)

    # Print table at benchmark etas
    print(f"\nBulk EOS errors ({label}):")
    print(f"  {'eta':>6s}  {'Z_pred':>8s}  {'Z_CS':>8s}  {'Z_err%':>8s}  "
          f"{'mu_pred':>8s}  {'mu_CS':>8s}  {'mu_err%':>8s}")
    print("  " + "-" * 60)
    for eta_b in [0.367, 0.393, 0.449, 0.492]:
        A, B = net.from_eta(eta_b)
        Z_pred = float(BT.Z_lutsko(eta_b, A, B))
        Z_cs = float(BT.Z_CS(eta_b))
        mu_pred = float(BT.mu_ex_bulk_lutsko(eta_b, A, B))
        mu_cs = float(BT.mu_ex_CS(eta_b))
        Z_err = 100 * (Z_pred - Z_cs) / Z_cs
        mu_err = 100 * (mu_pred - mu_cs) / mu_cs
        print(f"  {eta_b:6.3f}  {Z_pred:8.4f}  {Z_cs:8.4f}  {Z_err:8.4f}  "
              f"{mu_pred:8.4f}  {mu_cs:8.4f}  {mu_err:8.4f}")

    return errors


def get_parameter_curves(net, label=""):
    """Get A(eta), B(eta), C(eta) over dense eta grid."""
    etas = np.linspace(0.05, 0.50, 100)
    As, Bs, Cs = [], [], []
    for eta in etas:
        A, B = net.from_eta(eta)
        A, B = float(A), float(B)
        As.append(A)
        Bs.append(B)
        Cs.append(8*A + 2*B - 9)
    return etas, np.array(As), np.array(Bs), np.array(Cs)


def main():
    print("=" * 60)
    print("TWO-PHASE TRAINING ANALYSIS")
    print("=" * 60)

    out = Path('outputs')
    out.mkdir(exist_ok=True)
    solver = WallSolver(nz=1024, Lz=8.0, R=0.5)

    # ===== Phase 1 =====
    net_phase1 = train_phase1()

    # Bulk EOS errors after Phase 1
    eos_phase1 = compute_bulk_eos_errors(net_phase1, "Phase 1")

    # Parameter curves after Phase 1
    etas1, A1, B1, C1 = get_parameter_curves(net_phase1, "Phase 1")

    # ===== Phase 2 (default dA=dB=0.01) =====
    net_phase2, history_default = train_phase2_with_history(
        net_phase1, solver, n_iter=20, lr=5e-4, dA=0.01, dB=0.01)

    # Bulk EOS errors after Phase 2
    eos_phase2 = compute_bulk_eos_errors(net_phase2, "Phase 2")

    # Parameter curves after Phase 2
    etas2, A2, B2, C2 = get_parameter_curves(net_phase2, "Phase 2")

    # ===== Finite-difference sensitivity (retrain from Phase 1) =====
    print("\n" + "=" * 60)
    print("FINITE-DIFFERENCE SENSITIVITY")
    print("=" * 60)

    # dA=dB=0.005
    net_small, history_small = train_phase2_with_history(
        net_phase1, solver, n_iter=20, lr=5e-4, dA=0.005, dB=0.005)
    eos_small = compute_bulk_eos_errors(net_small, "FD step=0.005")

    # dA=dB=0.02
    net_large, history_large = train_phase2_with_history(
        net_phase1, solver, n_iter=20, lr=5e-4, dA=0.02, dB=0.02)
    eos_large = compute_bulk_eos_errors(net_large, "FD step=0.02")

    # Print FD sensitivity summary
    print("\nFD Sensitivity: Final contact densities at eta=0.367")
    for label, net_fd in [("dA=dB=0.005", net_small),
                           ("dA=dB=0.01", net_phase2),
                           ("dA=dB=0.02", net_large)]:
        A, B = net_fd.from_eta(0.367)
        func = esFMT_Tensor(A=float(A), B=float(B))
        r = solver.solve(0.367, func, max_iter=4000, tol=1e-7, verbose=False)
        print(f"  {label}: A={float(A):.4f}, B={float(B):.4f}, "
              f"contact={float(r['contact']):.3f}")

    # ===== FIGURE 1: Bulk EOS preservation =====
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(eos_phase1['eta'], eos_phase1['Z_err_pct'], 'b-', lw=2,
            label='Phase 1 (bulk)')
    ax.plot(eos_phase2['eta'], eos_phase2['Z_err_pct'], 'r-', lw=2,
            label='Phase 2 (wall)')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$Z$ error (\%)')
    ax.set_title('Compressibility factor')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(eos_phase1['eta'], eos_phase1['mu_err_pct'], 'b-', lw=2,
            label='Phase 1 (bulk)')
    ax.plot(eos_phase2['eta'], eos_phase2['mu_err_pct'], 'r-', lw=2,
            label='Phase 2 (wall)')
    ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\mu_{\mathrm{ex}}$ error (\%)')
    ax.set_title('Excess chemical potential')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Bulk EOS preservation after wall fine-tuning', fontsize=12)
    plt.tight_layout()
    plt.savefig(out / 'bulk_eos_preservation.png', dpi=200)
    print(f"\nSaved: {out / 'bulk_eos_preservation.png'}")
    plt.close()

    # ===== FIGURE 2: Parameter curves =====
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ax = axes[0]
    ax.plot(etas1, A1, 'b-', lw=2, label='Phase 1')
    ax.plot(etas2, A2, 'r-', lw=2, label='Phase 2')
    ax.axhline(1.3, color='green', ls=':', lw=1, alpha=0.7, label='Gül ($A=1.3$)')
    ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.7, label='Lutsko ($A=1$)')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$A(\eta)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(etas1, B1, 'b-', lw=2, label='Phase 1')
    ax.plot(etas2, B2, 'r-', lw=2, label='Phase 2')
    ax.axhline(-1.0, color='green', ls=':', lw=1, alpha=0.7, label='Gül ($B=-1$)')
    ax.axhline(0.0, color='gray', ls=':', lw=1, alpha=0.7, label='Lutsko ($B=0$)')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$B(\eta)$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(etas1, C1, 'b-', lw=2, label='Phase 1')
    ax.plot(etas2, C2, 'r-', lw=2, label='Phase 2')
    ax.axhline(-0.6, color='green', ls=':', lw=1, alpha=0.7, label='Gül ($C=-0.6$)')
    ax.axhline(3.0, color='gray', ls=':', lw=1, alpha=0.5, label='PY ($C=3$)')
    ax.axhline(-3.0, color='orange', ls=':', lw=1, alpha=0.5, label='CS ($C=-3$)')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$C(\eta) = 8A + 2B - 9$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle(r'Learned $A(\eta)$, $B(\eta)$, $C(\eta)$: Phase 1 vs Phase 2',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out / 'parameter_curves.png', dpi=200)
    print(f"Saved: {out / 'parameter_curves.png'}")
    plt.close()

    # ===== FIGURE 3: Convergence + FD sensitivity =====
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Convergence
    ax = axes[0]
    iters_d = [h[0] for h in history_default]
    loss_d = [h[1] for h in history_default]
    ax.semilogy(iters_d, loss_d, 'r-o', ms=4, lw=2, label=r'$\delta=0.01$')
    loss_s = [h[1] for h in history_small]
    ax.semilogy(iters_d, loss_s, 'b-s', ms=4, lw=1.5, label=r'$\delta=0.005$')
    loss_l = [h[1] for h in history_large]
    ax.semilogy(iters_d, loss_l, 'g-^', ms=4, lw=1.5, label=r'$\delta=0.02$')
    ax.set_xlabel('Phase 2 iteration')
    ax.set_ylabel('Wall contact loss')
    ax.set_title('Convergence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # FD sensitivity: parameter trajectories
    ax = axes[1]
    C_d = [h[4] for h in history_default]
    C_s = [h[4] for h in history_small]
    C_l = [h[4] for h in history_large]
    ax.plot(iters_d, C_d, 'r-o', ms=4, lw=2, label=r'$\delta=0.01$')
    ax.plot(iters_d, C_s, 'b-s', ms=4, lw=1.5, label=r'$\delta=0.005$')
    ax.plot(iters_d, C_l, 'g-^', ms=4, lw=1.5, label=r'$\delta=0.02$')
    ax.axhline(-0.6, color='green', ls=':', lw=1, alpha=0.5, label='Gül')
    ax.set_xlabel('Phase 2 iteration')
    ax.set_ylabel(r'$C(\eta=0.367)$')
    ax.set_title('FD step sensitivity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Wall fine-tuning convergence and FD step sensitivity', fontsize=12)
    plt.tight_layout()
    plt.savefig(out / 'convergence_sensitivity.png', dpi=200)
    print(f"Saved: {out / 'convergence_sensitivity.png'}")
    plt.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
