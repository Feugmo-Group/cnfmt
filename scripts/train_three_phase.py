#!/usr/bin/env python
"""
Three-phase training: bulk EOS → test-particle sum rules → wall contacts.

Tests whether the simulation-free sum-rule step provides useful prior
information that reduces the wall fine-tuning effort.

Compares:
  - Two-phase: bulk → wall (20 iters)
  - Three-phase: bulk → TP (15 iters, 32^3) → wall (20 iters)

Usage:
    python scripts/train_three_phase.py
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

from core.grid import Grid
from core.thermodynamics import BulkThermodynamics as BT
from neural.network import ConditionalNetwork
from solvers.test_particle import TestParticleCalculator
from solvers.fmt_1d_wbii_tensor import WallSolver, esFMT_Tensor
from solvers.wall_profile import MC_WALL_DATA

PI = np.pi


def train_phase1_bulk(n_iter=1500, lr=3e-3, seed=42):
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


def train_phase_tp(net, n_iter=15, lr=1e-4, grid_size=32):
    """Test-particle sum-rule phase."""
    print(f"\nTP Phase: Sum-rule training ({grid_size}^3, {n_iter} iters)...")

    grid = Grid((grid_size, grid_size, grid_size), length=6.0)
    base_calc = TestParticleCalculator(grid, sigma=1.0, A=1.0, B=0.0)

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    eta_train = [0.2, 0.3, 0.367, 0.393]
    dA, dB = 0.02, 0.02

    history = []

    for iteration in range(n_iter):
        total_loss = 0.0
        grad_accum = {}

        for eta in eta_train:
            A, B = net.from_eta(eta)
            A_val, B_val = float(A), float(B)

            calc = base_calc.with_parameters(A_val, B_val)
            result = calc.compute(eta, n_steps=200, lr=5e-4, verbose=False)
            loss_0 = result['delta_mu']**2 + result['delta_chi']**2
            total_loss += loss_0

            calc_pA = base_calc.with_parameters(min(A_val + dA, 1.5), B_val)
            res_pA = calc_pA.compute(eta, n_steps=200, lr=5e-4, verbose=False)
            loss_pA = res_pA['delta_mu']**2 + res_pA['delta_chi']**2
            dL_dA = (loss_pA - loss_0) / dA

            calc_pB = base_calc.with_parameters(A_val, min(B_val + dB, 0.0))
            res_pB = calc_pB.compute(eta, n_steps=200, lr=5e-4, verbose=False)
            loss_pB = res_pB['delta_mu']**2 + res_pB['delta_chi']**2
            dL_dB = (loss_pB - loss_0) / dB

            grad_accum[eta] = (dL_dA, dL_dB)

        total_loss /= len(eta_train)
        A, B = net.from_eta(0.367)
        C = 8*float(A) + 2*float(B) - 9
        history.append((iteration, total_loss, float(A), float(B), C))

        @eqx.filter_value_and_grad
        def surrogate_loss(net):
            total = 0.0
            for eta in eta_train:
                A, B = net.from_eta(eta)
                dL_dA, dL_dB = grad_accum[eta]
                total += dL_dA * A + dL_dB * B
            return total / len(eta_train)

        _, grads = surrogate_loss(net)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(net, eqx.is_array))
        net = eqx.apply_updates(net, updates)

        if iteration % 5 == 0 or iteration == n_iter - 1:
            print(f"  Iter {iteration}: SR_loss={total_loss:.4e}, "
                  f"A={float(A):.4f}, B={float(B):.4f}, C={C:.4f}")

    print(f"  TP Phase final: SR_loss={total_loss:.4e}")
    return net, history


def train_phase_wall(net, solver, n_iter=20, lr=5e-4, label="Wall"):
    """Wall contact density fine-tuning phase."""
    print(f"\n{label} Phase: Wall fine-tuning ({n_iter} iters)...")

    eta_targets = [0.367, 0.393, 0.449]
    md_contacts = {0.367: 5.36, 0.393: 6.15, 0.449: 8.34}

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))
    dA, dB = 0.01, 0.01

    def get_solver_contact(A_val, B_val, eta):
        func = esFMT_Tensor(A=A_val, B=B_val)
        result = solver.solve(eta, func, max_iter=4000, tol=1e-7, verbose=False)
        return float(result['contact'])

    history = []

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

    print(f"  {label} Phase final: wall_loss={total_loss:.4e}")
    return net, history


def compute_contacts(net, solver, label=""):
    """Compute and print contact densities."""
    eta_values = [0.367, 0.393, 0.449, 0.492]
    md_contacts = {0.367: 5.36, 0.393: 6.15, 0.449: 8.34, 0.492: 10.65}
    contacts = {}

    print(f"\nContact densities ({label}):")
    print(f"  {'eta':>6s}  {'MD':>8s}  {'NN':>8s}  {'err%':>8s}  "
          f"{'A':>7s}  {'B':>7s}  {'C':>7s}")
    print("  " + "-" * 55)

    for eta in eta_values:
        A, B = net.from_eta(eta)
        A_f, B_f = float(A), float(B)
        C_f = 8*A_f + 2*B_f - 9
        func = esFMT_Tensor(A=A_f, B=B_f)
        result = solver.solve(eta, func, max_iter=4000, tol=1e-7, verbose=False)
        contact = float(result['contact'])
        md = md_contacts[eta]
        err = 100 * (contact - md) / md
        contacts[eta] = contact
        print(f"  {eta:6.3f}  {md:8.2f}  {contact:8.2f}  {err:+8.1f}  "
              f"{A_f:7.4f}  {B_f:7.4f}  {C_f:7.4f}")

    return contacts


def main():
    print("=" * 60)
    print("THREE-PHASE TRAINING COMPARISON")
    print("=" * 60)

    out = Path('outputs')
    out.mkdir(exist_ok=True)
    solver = WallSolver(nz=1024, Lz=8.0, R=0.5)

    # ===== Route A: Two-phase (bulk → wall) =====
    print("\n" + "=" * 60)
    print("ROUTE A: TWO-PHASE (bulk → wall)")
    print("=" * 60)
    net_bulk = train_phase1_bulk()
    contacts_bulk = compute_contacts(net_bulk, solver, "After Phase 1 (bulk)")

    net_2phase, hist_2phase = train_phase_wall(
        net_bulk, solver, n_iter=20, lr=5e-4, label="2-phase Wall")
    contacts_2phase = compute_contacts(net_2phase, solver, "After 2-phase")

    # ===== Route B: Three-phase (bulk → TP → wall) =====
    print("\n" + "=" * 60)
    print("ROUTE B: THREE-PHASE (bulk → TP → wall)")
    print("=" * 60)
    # Reuse same Phase 1 network
    net_tp, hist_tp = train_phase_tp(net_bulk, n_iter=15, lr=1e-4, grid_size=32)
    contacts_tp = compute_contacts(net_tp, solver, "After TP phase")

    net_3phase, hist_3phase = train_phase_wall(
        net_tp, solver, n_iter=20, lr=5e-4, label="3-phase Wall")
    contacts_3phase = compute_contacts(net_3phase, solver, "After 3-phase")

    # ===== Comparison figure =====
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Wall loss convergence comparison
    ax = axes[0]
    iters_2 = [h[0] for h in hist_2phase]
    loss_2 = [h[1] for h in hist_2phase]
    iters_3 = [h[0] for h in hist_3phase]
    loss_3 = [h[1] for h in hist_3phase]
    ax.semilogy(iters_2, loss_2, 'b-o', ms=4, lw=2, label='2-phase (bulk→wall)')
    ax.semilogy(iters_3, loss_3, 'r-s', ms=4, lw=2, label='3-phase (bulk→TP→wall)')
    ax.set_xlabel('Wall fine-tuning iteration')
    ax.set_ylabel('Wall contact loss')
    ax.set_title('Convergence comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # C trajectory comparison
    ax = axes[1]
    C_2 = [h[4] for h in hist_2phase]
    C_3 = [h[4] for h in hist_3phase]
    ax.plot(iters_2, C_2, 'b-o', ms=4, lw=2, label='2-phase')
    ax.plot(iters_3, C_3, 'r-s', ms=4, lw=2, label='3-phase')
    ax.axhline(-0.6, color='green', ls=':', lw=1, alpha=0.7, label='Gül ($C=-0.6$)')
    ax.set_xlabel('Wall fine-tuning iteration')
    ax.set_ylabel(r'$C(\eta=0.367)$')
    ax.set_title(r'$C$ parameter trajectory')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Two-phase vs three-phase training', fontsize=12)
    plt.tight_layout()
    plt.savefig(out / 'three_phase_comparison.png', dpi=200)
    print(f"\nSaved: {out / 'three_phase_comparison.png'}")
    plt.close()

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY: CONTACT DENSITIES")
    print("=" * 60)
    eta_values = [0.367, 0.393, 0.449, 0.492]
    md_contacts = {0.367: 5.36, 0.393: 6.15, 0.449: 8.34, 0.492: 10.65}
    print(f"  {'eta':>6s}  {'MD':>8s}  {'Bulk':>8s}  {'2-phase':>8s}  {'3-phase':>8s}")
    print("  " + "-" * 45)
    for eta in eta_values:
        md = md_contacts[eta]
        c_b = contacts_bulk.get(eta, 0)
        c_2 = contacts_2phase.get(eta, 0)
        c_3 = contacts_3phase.get(eta, 0)
        print(f"  {eta:6.3f}  {md:8.2f}  {c_b:8.2f}  {c_2:8.2f}  {c_3:8.2f}")


if __name__ == '__main__':
    main()
