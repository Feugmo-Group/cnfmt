#!/usr/bin/env python
"""
Train Nonlocal Neural Functional
=================================

Curriculum training of the nonlocal Lutsko functional with spatially
varying A(r), B(r).  Uses only physics constraints — no simulation data.

Phases:
  1. Bulk thermodynamics (Z, μ, χ → Carnahan-Starling)
  2. Contact sum rule via 1D Picard wall profiles
  3. OZ consistency (c₂ → g(r) constraints + PY reference)
  4. Noether invariance + cosine-annealed fine-tune

Outputs (to --output-dir):
  - Checkpoints at each phase boundary
  - Loss curve plot
  - Bulk parameter table
  - Wall profile comparison vs MD data

Usage:
    python -m scripts.train_nonlocal                  # defaults
    python -m scripts.train_nonlocal --quick           # fast test run
    python -m scripts.train_nonlocal --output-dir runs/exp1
    python -m scripts.train_nonlocal --resume phase2   # resume from checkpoint
"""

import argparse
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import equinox as eqx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.grid import Grid
from core.weights import FMTKernels
from core.densities import WeightedDensityCalculator
from core.thermodynamics import BulkThermodynamics as BT
from nonlocal_ext.functional import NonlocalLutskoFunctional
from nonlocal_ext.kernels import LearnableKernel
from neural.network import NonlocalConditionalNetwork
from solvers.fmt_1d_wbii_tensor import WallSolver, esFMT_Tensor
from solvers.wall_profile import get_mc_data
from training.curriculum import CurriculumTrainer, CurriculumConfig


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train nonlocal neural functional via curriculum."
    )
    p.add_argument("--output-dir", type=str, default="outputs/nonlocal",
                   help="Output directory (default: outputs/nonlocal)")
    p.add_argument("--quick", action="store_true",
                   help="Quick test run (few epochs, small grid)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint name (e.g. 'phase1', 'phase2')")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--grid-size", type=int, default=32,
                   help="3D grid size per dimension (default: 32)")
    p.add_argument("--box-length", type=float, default=10.0,
                   help="Box length in σ (default: 10.0)")
    p.add_argument("--skip-oz", action="store_true",
                   help="Skip Phase 3 (OZ) — useful for fast iteration")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# FUNCTIONAL SETUP
# ═════════════════════════════════════════════════════════════════════

def create_functional(grid, seed=42):
    """Create a fresh NonlocalLutskoFunctional."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    kernels = FMTKernels(grid, R=0.5)
    calculator = WeightedDensityCalculator(kernels)
    network = NonlocalConditionalNetwork(k1)
    kernel = LearnableKernel(k2)

    return NonlocalLutskoFunctional(network, kernel, calculator, grid)


# ═════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════

def validate_bulk(functional, outdir):
    """Validate bulk thermodynamics and save table."""
    eta_vals = np.linspace(0.05, 0.50, 20)
    print("\n  Bulk Thermodynamics Validation")
    print("  " + "-" * 60)
    print(f"  {'η':>6s}  {'A':>7s}  {'B':>7s}  {'C':>8s}  "
          f"{'Z_NN':>8s}  {'Z_CS':>8s}  {'err%':>7s}")
    print("  " + "-" * 60)

    rows = []
    for eta in eta_vals:
        A, B = functional.bulk_parameters(eta)
        A_f, B_f = float(A), float(B)
        C = 8 * A_f + 2 * B_f - 9
        Z_nn = float(BT.Z_lutsko(eta, A, B))
        Z_cs = float(BT.Z_CS(eta))
        err = 100 * (Z_nn / Z_cs - 1)
        rows.append((eta, A_f, B_f, C, Z_nn, Z_cs, err))
        print(f"  {eta:6.3f}  {A_f:7.4f}  {B_f:7.4f}  {C:+8.4f}  "
              f"{Z_nn:8.4f}  {Z_cs:8.4f}  {err:+7.3f}")

    # Save to file
    with open(outdir / "bulk_parameters.txt", "w") as f:
        f.write(f"{'eta':>8s}  {'A':>10s}  {'B':>10s}  {'C':>10s}  "
                f"{'Z_NN':>10s}  {'Z_CS':>10s}  {'err%':>10s}\n")
        for row in rows:
            f.write(f"{row[0]:8.4f}  {row[1]:10.6f}  {row[2]:10.6f}  "
                    f"{row[3]:+10.6f}  {row[4]:10.6f}  {row[5]:10.6f}  "
                    f"{row[6]:+10.4f}\n")
    print(f"\n  Saved: {outdir / 'bulk_parameters.txt'}")

    return rows


def validate_wall_profiles(functional, outdir):
    """Solve wall profiles and compare to MD data."""
    eta_targets = [0.367, 0.393, 0.449, 0.492]
    solver = WallSolver(nz=2048, Lz=8.0, R=0.5)

    print("\n  Wall Profile Validation")
    print("  " + "-" * 55)
    print(f"  {'η':>6s}  {'contact_NN':>12s}  {'contact_CS':>12s}  "
          f"{'err%':>7s}  {'conv':>5s}")
    print("  " + "-" * 55)

    results = []
    for eta in eta_targets:
        A, B = functional.bulk_parameters(eta)
        fmt = esFMT_Tensor(A=float(A), B=float(B))
        res = solver.solve(eta, fmt, max_iter=6000, tol=1e-8, verbose=False)
        err = 100 * (res["contact"] / res["contact_CS"] - 1)
        results.append(res)
        conv = "✓" if res["converged"] else "✗"
        print(f"  {eta:6.3f}  {res['contact']:12.4f}  "
              f"{res['contact_CS']:12.4f}  {err:+7.2f}  {conv:>5s}")

    # Plot profiles vs MD
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (eta, res) in enumerate(zip(eta_targets, results)):
        ax = axes[i]
        z = np.array(res["z"])
        rho_norm = np.array(res["rho_norm"])
        ax.plot(z, rho_norm, "b-", lw=1.5, label="Nonlocal NN")

        # MD reference
        mc = get_mc_data(eta)
        if mc is not None:
            z_mc = np.array(mc["z"])
            rho_mc = np.array(mc["rho"]) / mc["rho_bulk"]
            ax.plot(z_mc, rho_mc, "ko", ms=2, alpha=0.5, label="MD data")

        # CS contact
        ax.axhline(res["contact_CS"], color="g", ls=":", lw=1,
                    alpha=0.6, label=f"CS contact={res['contact_CS']:.2f}")

        ax.set_xlim(0, 5)
        ax.set_xlabel(r"$z / \sigma$")
        ax.set_ylabel(r"$\rho(z) / \rho_b$")
        ax.set_title(f"$\\eta = {eta}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Wall Density Profiles: Nonlocal Neural Functional", fontsize=13)
    plt.tight_layout()
    figpath = outdir / "wall_profiles.png"
    plt.savefig(figpath, dpi=200)
    plt.close()
    print(f"\n  Saved: {figpath}")

    return results


def plot_loss_curve(trainer, outdir):
    """Plot loss history by phase."""
    by_phase = trainer.get_loss_by_phase()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
    labels = {1: "Phase 1: Bulk", 2: "Phase 2: Contact",
              3: "Phase 3: OZ", 4: "Phase 4: Noether"}

    offset = 0
    for phase in sorted(by_phase.keys()):
        losses = by_phase[phase]
        epochs = np.arange(offset, offset + len(losses))
        ax.semilogy(epochs, losses, color=colors.get(phase, "gray"),
                    lw=1.5, label=labels.get(phase, f"Phase {phase}"))
        offset += len(losses)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Curriculum Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    figpath = outdir / "loss_curve.png"
    plt.savefig(figpath, dpi=200)
    plt.close()
    print(f"  Saved: {figpath}")


def plot_AB_profiles(functional, outdir):
    """Plot A(η) and B(η) bulk parameter curves."""
    etas = np.linspace(0.01, 0.52, 100)
    As, Bs, Cs = [], [], []
    for eta in etas:
        A, B = functional.bulk_parameters(eta)
        As.append(float(A))
        Bs.append(float(B))
        Cs.append(8 * float(A) + 2 * float(B) - 9)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(etas, As, "b-", lw=2)
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel(r"$A(\eta)$")
    axes[0].set_title("Learned A parameter")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(etas, Bs, "r-", lw=2)
    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel(r"$B(\eta)$")
    axes[1].set_title("Learned B parameter")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(etas, Cs, "k-", lw=2)
    axes[2].axhline(-3, color="g", ls=":", label="CS (C=-3)")
    axes[2].axhline(0, color="b", ls=":", label="PY (C=0)")
    axes[2].axhline(-0.6, color="orange", ls=":", label="Gül (C=-0.6)")
    axes[2].set_xlabel(r"$\eta$")
    axes[2].set_ylabel(r"$C(\eta) = 8A + 2B - 9$")
    axes[2].set_title("Constraint parameter")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2)

    plt.suptitle("Learned Bulk Parameters", fontsize=13)
    plt.tight_layout()
    figpath = outdir / "AB_parameters.png"
    plt.savefig(figpath, dpi=200)
    plt.close()
    print(f"  Saved: {figpath}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(outdir / "checkpoints")

    print("=" * 65)
    print("  NONLOCAL NEURAL FUNCTIONAL TRAINING")
    print("  Physics-constrained, no simulation data")
    print("=" * 65)
    print(f"  Output:    {outdir}")
    print(f"  Grid:      {args.grid_size}³, L={args.box_length}σ")
    print(f"  Seed:      {args.seed}")
    print(f"  Quick:     {args.quick}")
    print(f"  Skip OZ:   {args.skip_oz}")
    if args.resume:
        print(f"  Resume:    {args.resume}")

    # ── Grid and functional ──
    grid = Grid((args.grid_size,) * 3, args.box_length)
    functional = create_functional(grid, args.seed)

    # ── Curriculum config ──
    if args.quick:
        cur_config = CurriculumConfig(
            n_epochs_phase1=30,
            n_epochs_phase2=20,
            n_epochs_phase3=0 if args.skip_oz else 10,
            n_epochs_phase4=10,
            n_eta_train=10,
            eta_wall_profiles=[0.2, 0.3, 0.4],
            oz_eta_values=[0.2, 0.3],
            solver_max_iter=3000,
            log_every=5,
            save_every=50,
            checkpoint_dir=ckpt_dir,
        )
    else:
        cur_config = CurriculumConfig(
            n_epochs_phase1=200,
            n_epochs_phase2=300,
            n_epochs_phase3=0 if args.skip_oz else 300,
            n_epochs_phase4=200,
            n_eta_train=30,
            eta_wall_profiles=[0.1, 0.2, 0.3, 0.367, 0.393, 0.449],
            oz_eta_values=[0.1, 0.2, 0.3, 0.4],
            solver_max_iter=5000,
            log_every=10,
            save_every=50,
            checkpoint_dir=ckpt_dir,
        )

    # ── Trainer ──
    trainer = CurriculumTrainer(functional, grid, cur_config)

    # ── Resume ──
    start_phase = 1
    if args.resume:
        trainer.load_checkpoint(args.resume)
        phase_map = {"phase1": 2, "phase2": 3, "phase3": 4}
        start_phase = phase_map.get(args.resume, 1)
        print(f"  Resuming from phase {start_phase}")

    # ── Train ──
    t0 = time.time()
    trained_functional = trainer.train(start_phase=start_phase, verbose=True)
    elapsed = time.time() - t0

    # ── Validation ──
    print("\n" + "=" * 65)
    print("  VALIDATION")
    print("=" * 65)

    validate_bulk(trained_functional, outdir)
    validate_wall_profiles(trained_functional, outdir)

    # ── Plots ──
    print("\n  Generating plots...")
    plot_loss_curve(trainer, outdir)
    plot_AB_profiles(trained_functional, outdir)

    # ── Summary ──
    trainer.report()

    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  All outputs saved to: {outdir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
