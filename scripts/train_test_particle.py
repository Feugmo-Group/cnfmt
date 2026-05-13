#!/usr/bin/env python
"""
Approach 5: Train A(eta), B(eta) on test-particle sum-rule residuals.

Minimizes the Gül et al. (2024) sum-rule deviations:
    L = δ_μ² + δ_χ²
where δ_μ and δ_χ are computed via the full 3D test-particle DFT.

Unlike bulk EOS training (which depends only on C = 8A + 2B - 9),
test-particle sum rules depend on A and B individually through the
tensor contributions to the 3D density profile around the test sphere.

Usage:
    python -m cnfmt.scripts.train_test_particle
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


def train_phase1_bulk(n_iter=1000, lr=3e-3, seed=42):
    """Phase 1: Quick bulk EOS pre-training to get near the right C."""
    print("Phase 1: Bulk EOS pre-training...")
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


def train_phase2_test_particle(net, n_iter=30, lr=5e-5, grid_size=32):
    """Phase 2: Fine-tune on test-particle sum-rule residuals.

    Uses numerical gradients via finite differences on (A, B) since
    the 3D DFT minimizer is too expensive for full backpropagation.
    """
    print(f"\nPhase 2: Test-particle sum-rule training ({grid_size}^3 grid)...")

    # Create 3D grid for test particle calculations
    grid = Grid((grid_size, grid_size, grid_size), length=6.0)
    base_calc = TestParticleCalculator(grid, sigma=1.0, A=1.0, B=0.0)

    optimizer = optax.adamw(lr)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    # Training packing fractions (moderate range where solver is stable)
    eta_train = [0.2, 0.3, 0.367, 0.393]
    dA, dB = 0.02, 0.02  # finite difference step

    history = []

    for iteration in range(n_iter):
        total_loss = 0.0
        grad_accum = {}

        for eta in eta_train:
            A, B = net.from_eta(eta)
            A_val, B_val = float(A), float(B)

            # Compute sum-rule deviations at current (A, B)
            calc = base_calc.with_parameters(A_val, B_val)
            result = calc.compute(eta, n_steps=300, lr=5e-4, verbose=False)
            delta_mu = result['delta_mu']
            delta_chi = result['delta_chi']
            loss_0 = delta_mu**2 + delta_chi**2
            total_loss += loss_0

            # Finite difference: dL/dA
            calc_pA = base_calc.with_parameters(min(A_val + dA, 1.5), B_val)
            res_pA = calc_pA.compute(eta, n_steps=300, lr=5e-4, verbose=False)
            loss_pA = res_pA['delta_mu']**2 + res_pA['delta_chi']**2
            dL_dA = (loss_pA - loss_0) / dA

            # Finite difference: dL/dB
            calc_pB = base_calc.with_parameters(A_val, min(B_val + dB, 0.0))
            res_pB = calc_pB.compute(eta, n_steps=300, lr=5e-4, verbose=False)
            loss_pB = res_pB['delta_mu']**2 + res_pB['delta_chi']**2
            dL_dB = (loss_pB - loss_0) / dB

            grad_accum[eta] = (dL_dA, dL_dB)

        total_loss /= len(eta_train)
        history.append(total_loss)

        # Surrogate loss for backpropagation through the network
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
            A, B = net.from_eta(0.367)
            C = 8*float(A) + 2*float(B) - 9
            print(f"  Iter {iteration}: sum_rule_loss={total_loss:.4e}, "
                  f"A={float(A):.4f}, B={float(B):.4f}, C={C:.4f}")

    print(f"  Phase 2 final: sum_rule_loss={total_loss:.4e}")
    return net, history


def main():
    print("="*60)
    print("APPROACH 5: TEST-PARTICLE SUM-RULE TRAINING")
    print("="*60)

    # Phase 1: Bulk EOS pre-training
    net = train_phase1_bulk(n_iter=1000)

    # Print Phase 1 parameters
    print("\nPhase 1 learned parameters:")
    eta_values = [0.367, 0.393, 0.449, 0.492]
    for eta in eta_values:
        A, B = net.from_eta(eta)
        A, B = float(A), float(B)
        C = 8*A + 2*B - 9
        print(f"  eta={eta}: A={A:.4f}, B={B:.4f}, C={C:.4f}")

    # Phase 2: Test-particle fine-tuning (48^3 grid for better accuracy)
    net, tp_history = train_phase2_test_particle(net, n_iter=50, lr=1e-4, grid_size=64)

    # Print Phase 2 parameters
    print("\nPhase 2 learned parameters:")
    for eta in eta_values:
        A, B = net.from_eta(eta)
        A, B = float(A), float(B)
        C = 8*A + 2*B - 9
        print(f"  eta={eta}: A={A:.4f}, B={B:.4f}, C={C:.4f}")

    # Compute wall profiles to evaluate improvement
    print("\nComputing wall profiles with test-particle-trained parameters...")
    solver = WallSolver(nz=2048, Lz=8.0, R=0.5)

    print(f"\n{'eta':>6s}  {'MD':>8s}  {'Lutsko':>8s}  {'Gül':>8s}  {'NN-bulk':>8s}  {'NN-TP':>8s}")
    print("-"*55)

    for eta in eta_values:
        # MD data
        if eta in MC_WALL_DATA:
            mc = MC_WALL_DATA[eta]
            rho_bulk = mc.get('rho_bulk', eta / ((4/3) * PI * 0.5**3))
            md_contact = mc['rho'][0] / rho_bulk
        else:
            md_contact = 0

        # Fixed functionals
        r_lut = solver.solve(eta, esFMT_Tensor(1.0, 0.0), max_iter=3000, tol=1e-7, verbose=False)
        r_gul = solver.solve(eta, esFMT_Tensor(1.3, -1.0), max_iter=3000, tol=1e-7, verbose=False)

        # NN-trained parameters
        A_nn, B_nn = float(net.from_eta(eta)[0]), float(net.from_eta(eta)[1])
        r_nn = solver.solve(eta, esFMT_Tensor(A_nn, B_nn), max_iter=4000, tol=1e-7, verbose=False)

        print(f"{eta:6.3f}  {md_contact:8.2f}  {float(r_lut['contact']):8.2f}  "
              f"{float(r_gul['contact']):8.2f}  {'---':>8s}  {float(r_nn['contact']):8.2f}")

    # Save results
    out = Path('outputs')
    out.mkdir(exist_ok=True)
    print(f"\nDone! Results printed above.")


if __name__ == '__main__':
    main()
