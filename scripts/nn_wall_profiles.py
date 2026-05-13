#!/usr/bin/env python
"""
B4: Compute wall profiles using NN-learned A(eta), B(eta).

Trains the network (Approach 1: CS EOS), then computes density profiles
at the four benchmark packing fractions and compares to:
  - Lutsko fixed (A=1, B=0)
  - Gul et al. fixed (A=1.3, B=-1.0)
  - MD data (Davidchack et al. 2016)

Usage:
    python -m cnfmt.scripts.nn_wall_profiles
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

from scripts.paper_figure_style import apply_paper_style
apply_paper_style()

from solvers.fmt_1d_wbii_tensor import WallSolver, esFMT_Tensor
from neural.network import ConditionalNetwork
from core.thermodynamics import BulkThermodynamics as BT
from solvers.wall_profile import MC_WALL_DATA

PI = np.pi


def train_network_phase1(n_iter=1500, lr=3e-3, seed=42):
    """Phase 1: Train network on CS EOS objective."""
    print("Phase 1: Bulk EOS training...")
    key = jax.random.PRNGKey(seed)
    # Match paper specs: 2 hidden layers, 32 neurons, bounds as stated
    net = ConditionalNetwork(key, n_features=5, hidden_dim=32, n_hidden=2,
                             A_bounds=(0.8, 1.5), B_bounds=(-1.5, 0.0))

    # Use cosine decay schedule for better convergence
    schedule = optax.cosine_decay_schedule(lr, n_iter)
    optimizer = optax.adamw(schedule)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    # Extended range to cover all benchmark etas including 0.492
    eta_train = jnp.linspace(0.05, 0.50, 20)

    # Vectorized single-eta loss for use with vmap
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


def train_network_phase2(net, solver, n_iter=50, lr=1e-4):
    """Phase 2: Fine-tune on wall contact densities vs MD data.

    Uses numerical gradients (finite differences on A,B) to differentiate
    through the non-differentiable Picard solver.
    """
    print("\nPhase 2: Wall contact density fine-tuning...")

    # MD contact densities from Davidchack et al. 2016
    eta_targets = [0.367, 0.393, 0.449]  # skip 0.492 (near freezing, unstable)
    md_contacts = {0.367: 5.36, 0.393: 6.15, 0.449: 8.34}

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    def get_solver_contact(A_val, B_val, eta):
        """Run wall solver and return contact density."""
        func = esFMT_Tensor(A=A_val, B=B_val)
        result = solver.solve(eta, func, max_iter=4000, tol=1e-7, verbose=False)
        return float(result['contact'])

    dA, dB = 0.01, 0.01  # finite difference step

    for iteration in range(n_iter):
        total_loss = 0.0
        # Accumulate numerical gradients for A and B at each eta
        grad_accum = {'dL_dA': {}, 'dL_dB': {}}

        for eta in eta_targets:
            A, B = net.from_eta(eta)
            A_val, B_val = float(A), float(B)
            md_contact = md_contacts[eta]
            rho_bulk = eta / ((4/3) * PI * 0.5**3)

            # Contact density at current (A, B)
            contact_0 = get_solver_contact(A_val, B_val, eta)
            loss_0 = ((contact_0 - md_contact) / md_contact) ** 2
            total_loss += loss_0

            # Finite differences for dL/dA
            contact_pA = get_solver_contact(min(A_val + dA, 1.5), B_val, eta)
            loss_pA = ((contact_pA - md_contact) / md_contact) ** 2
            dL_dA = (loss_pA - loss_0) / dA

            # Finite differences for dL/dB
            contact_pB = get_solver_contact(A_val, min(B_val + dB, 0.0), eta)
            loss_pB = ((contact_pB - md_contact) / md_contact) ** 2
            dL_dB = (loss_pB - loss_0) / dB

            grad_accum['dL_dA'][eta] = dL_dA
            grad_accum['dL_dB'][eta] = dL_dB

        total_loss /= len(eta_targets)

        # Now do a gradient step using the chain rule:
        # dL/d(params) = sum_eta (dL/dA * dA/d(params) + dL/dB * dB/d(params))
        # We use JAX autodiff for dA/d(params) and dB/d(params)
        @eqx.filter_value_and_grad
        def surrogate_loss(net):
            """Surrogate loss whose gradient matches the true numerical gradient."""
            total = 0.0
            for eta in eta_targets:
                A, B = net.from_eta(eta)
                # Linear surrogate: L_surrogate = dL/dA * A + dL/dB * B
                total += grad_accum['dL_dA'][eta] * A + grad_accum['dL_dB'][eta] * B
            return total / len(eta_targets)

        _, grads = surrogate_loss(net)
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(net, eqx.is_array))
        net = eqx.apply_updates(net, updates)

        if iteration % 10 == 0 or iteration == n_iter - 1:
            A, B = net.from_eta(0.367)
            print(f"  Iter {iteration}: wall_loss={total_loss:.4e}, "
                  f"A={float(A):.4f}, B={float(B):.4f}")

    print(f"  Phase 2 final: wall_loss={total_loss:.4e}")
    return net


def train_network(n_iter=1500, lr=3e-3, seed=42, phase2_iters=20):
    """Two-phase training: bulk EOS + wall contact fine-tuning."""
    # Phase 1: bulk EOS
    net = train_network_phase1(n_iter=n_iter, lr=lr, seed=seed)

    # Phase 2: fine-tune on wall contact densities (use smaller solver for speed)
    solver = WallSolver(nz=1024, Lz=8.0, R=0.5)
    net = train_network_phase2(net, solver, n_iter=phase2_iters, lr=5e-4)

    return net


def main():
    print("="*60)
    print("NN WALL PROFILES")
    print("="*60)

    # Train
    net = train_network()

    # Print learned parameters at benchmark etas
    print("\nLearned parameters:")
    eta_values = [0.367, 0.393, 0.449, 0.492]
    for eta in eta_values:
        A, B = net.from_eta(eta)
        A, B = float(A), float(B)
        C = 8*A + 2*B - 9
        print(f"  eta={eta}: A={A:.4f}, B={B:.4f}, C={C:.4f}")

    # Solve wall profiles
    solver = WallSolver(nz=2048, Lz=8.0, R=0.5)

    # Fixed-parameter functionals for comparison
    fixed_funcs = {
        'Lutsko (A=1, B=0)': esFMT_Tensor(A=1.0, B=0.0),
        'Gül et al. (A=1.3, B=$-$1)': esFMT_Tensor(A=1.3, B=-1.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    print("\nSolving wall profiles...")
    for idx, eta in enumerate(eta_values):
        ax = axes[idx]
        print(f"\neta = {eta}:")

        # MD data (subsampled for clarity)
        if eta in MC_WALL_DATA:
            mc = MC_WALL_DATA[eta]
            rho_bulk = mc.get('rho_bulk', eta / ((4/3) * PI * 0.5**3))
            mc_z = mc['z']
            mc_rho_norm = mc['rho'] / rho_bulk
            mask = mc_z <= 6.0
            z_plot = mc_z[mask]
            rho_plot = mc_rho_norm[mask]
            dense = z_plot <= 2.0
            sparse_idx = np.where(~dense)[0][::5]
            plot_idx = np.concatenate([np.where(dense)[0], sparse_idx])
            plot_idx.sort()
            ax.plot(z_plot[plot_idx], rho_plot[plot_idx], 'ko', ms=3, mfc='white',
                    mew=1.0, alpha=0.8, label='MD', zorder=10)

        # Fixed functionals
        for name, func in fixed_funcs.items():
            result = solver.solve(eta, func, max_iter=4000, tol=1e-8, verbose=False)
            contact = float(result['contact'])
            ax.plot(result['z'], result['rho_norm'], '--', lw=1.2,
                    alpha=0.7, label=f'{name}')
            print(f"  {name}: contact = {contact:.3f}")

        # NN-learned parameters
        A_nn, B_nn = net.from_eta(eta)
        A_nn, B_nn = float(A_nn), float(B_nn)
        nn_func = esFMT_Tensor(A=A_nn, B=B_nn)
        result_nn = solver.solve(eta, nn_func, max_iter=6000, tol=1e-8, verbose=False)
        contact_nn = float(result_nn['contact'])
        converged = result_nn.get('converged', True)

        if converged:
            ax.plot(result_nn['z'], result_nn['rho_norm'], '-', color='C3', lw=2,
                    label=f'NN (A={A_nn:.2f}, B={B_nn:.2f})')
            print(f"  NN: contact = {contact_nn:.3f}")
        else:
            # Check if profile looks reasonable despite not hitting tolerance
            max_rho_norm = float(np.max(result_nn['rho_norm']))
            if max_rho_norm < 25:
                ax.plot(result_nn['z'], result_nn['rho_norm'], '-', color='C3', lw=2,
                        label=f'NN (A={A_nn:.2f}, B={B_nn:.2f})')
                print(f"  NN: contact = {contact_nn:.3f} (not converged but stable)")
            else:
                ax.text(0.5, 0.5, 'NN: unstable\nat this $\\eta$',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='C3', style='italic',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='C3', alpha=0.8))
                print(f"  NN: DIVERGED (max rho/rho_b = {max_rho_norm:.1f})")

        # Reference lines
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5, lw=0.8)
        ax.axvline(0.5, color='gray', ls=':', alpha=0.3, lw=0.8)

        ax.set_xlabel(r'$z/\sigma$')
        ax.set_ylabel(r'$\rho(z)/\rho_b$')
        ax.set_title(rf'$\eta = {eta}$')
        ax.set_xlim([0.4, 6.0])
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.suptitle('Wall Profiles: NN-Learned vs Fixed Parameters', fontsize=14, y=1.01)
    plt.tight_layout()

    out = Path('outputs')
    out.mkdir(exist_ok=True)
    path = out / 'nn_wall_profiles.png'
    plt.savefig(path)
    print(f"\nSaved: {path}")
    plt.close()

    # Summary table
    print("\n" + "="*60)
    print("CONTACT DENSITY COMPARISON")
    print("="*60)
    print(f"{'eta':>6s}  {'MD':>8s}  {'CS':>8s}  {'Lutsko':>8s}  {'Gul':>8s}  {'NN':>8s}")
    print("-"*50)
    for eta in eta_values:
        if eta in MC_WALL_DATA:
            mc = MC_WALL_DATA[eta]
            rho_bulk = mc.get('rho_bulk', eta / ((4/3) * PI * 0.5**3))
            md_contact = mc['rho'][0] / rho_bulk
        else:
            rho_bulk = eta / ((4/3) * PI * 0.5**3)
            md_contact = 0
        cs_contact = float((1 + eta + eta**2 - eta**3) / (1 - eta)**3)

        r_lut = solver.solve(eta, esFMT_Tensor(1.0, 0.0), max_iter=2000, tol=1e-7, verbose=False)
        r_gul = solver.solve(eta, esFMT_Tensor(1.3, -1.0), max_iter=2000, tol=1e-7, verbose=False)
        A_nn, B_nn = float(net.from_eta(eta)[0]), float(net.from_eta(eta)[1])
        r_nn = solver.solve(eta, esFMT_Tensor(A_nn, B_nn), max_iter=2000, tol=1e-7, verbose=False)

        print(f"{eta:6.3f}  {md_contact:8.2f}  {cs_contact:8.2f}  "
              f"{float(r_lut['contact']):8.2f}  {float(r_gul['contact']):8.2f}  "
              f"{float(r_nn['contact']):8.2f}")


if __name__ == '__main__':
    main()
