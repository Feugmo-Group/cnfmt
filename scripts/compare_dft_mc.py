#!/usr/bin/env python
"""
DFT vs MC Comparison Script
============================

Compares the nonlocal DFT functional (trained or untrained) against Monte Carlo
simulation data across four comparisons:

  1. Bulk g(r): DFT c₂ via OZ  vs  NVT MC g(r)
  2. μ_ex: Widom insertion  vs  DFT c₁_bulk
  3. Wall profiles: precomputed/MD data  vs  DFT Picard
  4. Summary table: Z, μ_ex, g(σ⁺) for DFT vs CS vs MC

Usage
-----
    python -m scripts.compare_dft_mc
    python -m scripts.compare_dft_mc --checkpoint outputs/run1/checkpoints/nonlocal_phase2.eqx
    python -m scripts.compare_dft_mc --mc-data outputs/mc_walls --output-dir outputs/comparison_dft_mc
    python -m scripts.compare_dft_mc --quick
"""

import argparse
import sys
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
from correlations.direct import compute_c2_fourier
from correlations.ornstein_zernike import solve_oz_fourier, compute_g_radial
from solvers.fmt_1d_wbii_tensor import WallSolver, esFMT_Tensor
from solvers.wall_profile import get_mc_data
from mc.core.state import MCState
from mc.ensembles.nvt import NVTSampler, NVTState, mc_sweep, run_nvt
from mc.observables.rdf import RDFAccumulator
from mc.observables.widom import WidomAccumulator


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare nonlocal DFT functional against MC data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained .eqx checkpoint. "
             "If omitted, a fresh (untrained) functional is used.",
    )
    p.add_argument(
        "--output-dir", type=str, default="outputs/comparison_dft_mc",
        help="Directory for plots and results (default: outputs/comparison_dft_mc)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick mode: skip MC runs, use only DFT and any precomputed data.",
    )
    p.add_argument(
        "--mc-data", type=str, default=None,
        help="Path to directory containing precomputed MC data as .npz files. "
             "Wall profiles: keys 'z', 'rho', 'rho_bulk'. "
             "g(r): keys 'r', 'gr'. "
             "File naming: wall_eta{eta:.3f}.npz, gr_eta{eta:.3f}.npz",
    )
    p.add_argument(
        "--grid-size", type=int, default=32,
        help="3D grid points per dimension (default: 32)",
    )
    p.add_argument(
        "--box-length", type=float, default=10.0,
        help="Cubic box side length in sigma (default: 10.0)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--n-mc-particles", type=int, default=256,
        help="Number of particles for MC runs (default: 256)",
    )
    p.add_argument(
        "--n-mc-equil", type=int, default=2000,
        help="Number of equilibration sweeps (default: 2000)",
    )
    p.add_argument(
        "--n-mc-prod", type=int, default=10000,
        help="Number of production sweeps (default: 10000)",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# FUNCTIONAL SETUP
# ═════════════════════════════════════════════════════════════════════

def create_functional(grid, seed=42):
    """Create a fresh NonlocalLutskoFunctional."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    kernels = FMTKernels(grid, R=0.5)
    calc = WeightedDensityCalculator(kernels)
    network = NonlocalConditionalNetwork(k1)
    kernel = LearnableKernel(k2)
    return NonlocalLutskoFunctional(network, kernel, calc, grid)


def load_functional(checkpoint_path, grid, seed=42):
    """Load a trained functional from an Equinox checkpoint."""
    template = create_functional(grid, seed)
    return eqx.tree_deserialise_leaves(checkpoint_path, template)


# ═════════════════════════════════════════════════════════════════════
# HELPER: radial binning of a 3D Fourier-space field
# ═════════════════════════════════════════════════════════════════════

def _radial_bin_fourier(field_3d, grid, n_bins=150):
    """Radially average a 3D Fourier-space field by |k|.

    Returns (k_centers, f_radial, counts).
    """
    k_abs = np.array(grid.k_abs)
    k_max = float(np.pi / grid.dx)
    dk = k_max / n_bins
    k_centers = (np.arange(n_bins) + 0.5) * dk

    flat_k = k_abs.ravel()
    flat_f = np.array(jnp.real(field_3d)).ravel()
    bin_idx = np.floor(flat_k / dk).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    f_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    np.add.at(f_sum, bin_idx, flat_f)
    np.add.at(counts, bin_idx, 1.0)
    f_radial = np.where(counts > 0, f_sum / counts, 0.0)

    return k_centers, f_radial, counts


# ═════════════════════════════════════════════════════════════════════
# MC UTILITIES
# ═════════════════════════════════════════════════════════════════════

def _mc_box_length_for_eta(eta, n_particles, sigma=1.0):
    """Compute box length such that N particles at packing fraction eta."""
    V = n_particles * np.pi * sigma**3 / (6.0 * eta)
    return float(V ** (1.0 / 3.0))


def run_nvt_with_accumulators(eta, n_particles, n_equil, n_prod, seed,
                               collect_widom=True, rdf_n_bins=200,
                               verbose=False):
    """Run NVT MC and return (rdf_acc, widom_acc, final_state).

    Returns None for rdf_acc / widom_acc if the run fails.
    """
    box_length = _mc_box_length_for_eta(eta, n_particles)
    rho_bulk = n_particles / box_length**3
    r_max = min(box_length / 2.0, 5.0)

    print(f"    NVT MC: eta={eta:.3f}, N={n_particles}, "
          f"L={box_length:.3f}, rho={rho_bulk:.4f}")

    try:
        mc_state = MCState.from_fcc(N=n_particles, box_length=box_length)
        sampler = NVTSampler(mc_state, seed=seed)

        # Equilibration
        t0 = time.time()
        sampler.run(n_equil=n_equil, n_prod=0, verbose=verbose)
        print(f"    Equilibration done ({time.time()-t0:.1f}s)")

        # Production: accumulate g(r) and Widom per sweep
        rdf_acc = RDFAccumulator(n_bins=rdf_n_bins, r_max=r_max)
        widom_acc = WidomAccumulator(n_test=500, seed=seed + 1) if collect_widom else None

        # Re-use the internal NVT state after equilibration
        nvt_state = sampler.nvt_state

        # Run production sweep-by-sweep to accumulate observables
        t1 = time.time()
        SAMPLE_EVERY = 10  # accumulate every 10 sweeps
        max_disp = 0.2
        n_sampled = 0

        for sweep_idx in range(n_prod):
            # One sweep
            nvt_state, n_acc = mc_sweep(nvt_state, n_particles, max_disp)

            # Adjust displacement every 100 sweeps
            if (sweep_idx + 1) % 100 == 0:
                rate = nvt_state.n_accepted / max(nvt_state.n_total, 1)
                if rate > 0.40:
                    max_disp = min(max_disp * 1.05, box_length / 2.0)
                elif rate < 0.30:
                    max_disp = max(max_disp * 0.95, 0.01)

            # Sample observables
            if (sweep_idx + 1) % SAMPLE_EVERY == 0:
                positions_np = np.array(nvt_state.mc_state.positions)
                rdf_acc.update(positions_np, box_length)
                if widom_acc is not None:
                    widom_acc.update(positions_np, box_length)
                n_sampled += 1

        elapsed = time.time() - t1
        print(f"    Production done ({elapsed:.1f}s, {n_sampled} samples)")

        return rdf_acc, widom_acc, nvt_state.mc_state

    except Exception as exc:
        print(f"    WARNING: MC run failed: {exc}")
        return None, None, None


# ═════════════════════════════════════════════════════════════════════
# COMPARISON 1: Bulk g(r) — DFT c₂ vs MC g(r)
# ═════════════════════════════════════════════════════════════════════

def comparison1_bulk_gr(functional, grid, outdir, args):
    """Compare DFT g(r) from OZ equation with MC g(r) from NVT simulation."""
    eta_vals = [0.1, 0.2, 0.3, 0.367, 0.4]

    print("\n" + "=" * 68)
    print("  COMPARISON 1: Bulk g(r) — DFT c₂ via OZ  vs  MC NVT")
    print("=" * 68)

    n_cols = len(eta_vals)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes = [axes]

    results = {}

    for idx, eta in enumerate(eta_vals):
        ax = axes[idx]
        rho_bulk = 6.0 * eta / np.pi

        print(f"\n  eta = {eta:.3f}, rho = {rho_bulk:.4f}")

        # ── DFT g(r) via c₂(k) → OZ → g(r) ──────────────────────
        try:
            t0 = time.time()
            c2_k = compute_c2_fourier(functional, rho_bulk, grid)
            r_dft, g_dft = compute_g_radial(c2_k, rho_bulk, grid, n_bins=200)
            r_dft = np.array(r_dft)
            g_dft = np.array(g_dft)
            print(f"    DFT g(r) computed in {time.time()-t0:.1f}s")

            # Contact value from DFT g(r): first point at r > 1
            mask_contact = r_dft > 1.0
            g_contact_dft = float(g_dft[mask_contact][0]) if mask_contact.any() else np.nan

            ax.plot(r_dft, g_dft, "b-", lw=1.8, label="DFT (OZ)", zorder=3)
            results[eta] = {"g_contact_dft": g_contact_dft}
        except Exception as exc:
            print(f"    WARNING: DFT g(r) failed: {exc}")
            results[eta] = {"g_contact_dft": np.nan}

        # ── Precomputed MC g(r) from --mc-data dir ─────────────────
        mc_gr_loaded = False
        if args.mc_data:
            mc_data_dir = Path(args.mc_data)
            gr_file = mc_data_dir / f"gr_eta{eta:.3f}.npz"
            if gr_file.exists():
                try:
                    data = np.load(gr_file)
                    r_mc = data["r"]
                    g_mc = data["gr"]
                    ax.plot(r_mc, g_mc, "ko", ms=2, alpha=0.5,
                            label="MC (precomputed)", zorder=5)
                    results[eta]["g_contact_mc"] = float(np.interp(1.0, r_mc, g_mc))
                    mc_gr_loaded = True
                    print(f"    Loaded precomputed MC g(r) from {gr_file.name}")
                except Exception as exc:
                    print(f"    WARNING: Could not load {gr_file}: {exc}")

        # ── Live NVT MC g(r) (skip when --quick) ───────────────────
        if not args.quick and not mc_gr_loaded:
            rdf_acc, _, _ = run_nvt_with_accumulators(
                eta, args.n_mc_particles, args.n_mc_equil, args.n_mc_prod,
                seed=args.seed, collect_widom=False,
            )
            if rdf_acc is not None and rdf_acc.n_frames > 0:
                r_mc, g_mc = rdf_acc.get()
                ax.plot(r_mc, g_mc, "r--", lw=1.4, label="MC (NVT)", zorder=4)
                mask_mc = r_mc > 1.0
                g_contact_mc = float(g_mc[mask_mc][0]) if mask_mc.any() else np.nan
                results[eta]["g_contact_mc"] = g_contact_mc

                # Save for reuse
                np.savez(outdir / f"gr_eta{eta:.3f}.npz", r=r_mc, gr=g_mc)

        # Decoration
        ax.axvline(1.0, color="gray", ls=":", lw=0.8, alpha=0.6)
        ax.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.4)
        ax.set_xlabel(r"$r / \sigma$")
        ax.set_ylabel(r"$g(r)$")
        ax.set_title(rf"$\eta = {eta}$")
        ax.set_xlim(0.5, 4.0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Comparison 1: Bulk g(r) — DFT (OZ) vs Monte Carlo", fontsize=13)
    fig.tight_layout()
    figpath = outdir / "comp1_bulk_gr.png"
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {figpath}")

    return results


# ═════════════════════════════════════════════════════════════════════
# COMPARISON 2: μ_ex — Widom insertion vs DFT c₁
# ═════════════════════════════════════════════════════════════════════

def comparison2_mu_ex(functional, outdir, args):
    """Compare βμ_ex from Widom insertion (MC) and DFT c₁_bulk."""
    eta_vals = [0.1, 0.2, 0.3, 0.4, 0.45]

    print("\n" + "=" * 68)
    print("  COMPARISON 2: μ_ex — Widom insertion vs DFT c₁")
    print("=" * 68)

    mu_dft = []
    mu_cs  = []
    mu_mc  = []
    mu_mc_std = []
    mc_available = []

    for eta in eta_vals:
        # DFT: βμ_ex = -c₁_bulk
        try:
            c1_b = float(functional.compute_c1_bulk(eta))
            mu_dft_val = -c1_b
        except Exception as exc:
            print(f"  WARNING: DFT c1_bulk failed for eta={eta}: {exc}")
            mu_dft_val = np.nan
        mu_dft.append(mu_dft_val)

        # CS reference
        mu_cs.append(float(BT.mu_ex_CS(eta)))

        print(f"  eta={eta:.2f}  μ_DFT={mu_dft_val:.4f}  μ_CS={float(BT.mu_ex_CS(eta)):.4f}")

    # MC Widom (skip when --quick)
    mc_data = {}
    if not args.quick:
        for eta in eta_vals:
            if eta >= 0.45:
                # Widom breaks down at very high density — try anyway
                pass
            rdf_acc, widom_acc, _ = run_nvt_with_accumulators(
                eta, args.n_mc_particles, args.n_mc_equil, args.n_mc_prod,
                seed=args.seed, collect_widom=True,
            )
            if widom_acc is not None and widom_acc.n_frames > 0:
                mean_mu, std_mu = widom_acc.get()
                mc_data[eta] = (mean_mu, std_mu)
                print(f"    eta={eta:.2f}  μ_MC={mean_mu:.4f} ± {std_mu:.4f}")
            else:
                mc_data[eta] = (np.nan, np.nan)

    # Build arrays
    for eta in eta_vals:
        if eta in mc_data:
            mu_mc.append(mc_data[eta][0])
            mu_mc_std.append(mc_data[eta][1])
            mc_available.append(np.isfinite(mc_data[eta][0]))
        else:
            mu_mc.append(np.nan)
            mu_mc_std.append(np.nan)
            mc_available.append(False)

    mu_dft = np.array(mu_dft)
    mu_cs  = np.array(mu_cs)
    mu_mc  = np.array(mu_mc)
    mu_mc_std = np.array(mu_mc_std)
    eta_arr = np.array(eta_vals)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(eta_arr, mu_cs, "k-", lw=2.5, label="CS (exact)", zorder=5)
    ax.plot(eta_arr, mu_dft, "b^-", ms=7, lw=1.8, label="DFT (-c₁)", zorder=4)

    if any(mc_available):
        finite = np.isfinite(mu_mc)
        ax.errorbar(
            eta_arr[finite], mu_mc[finite], yerr=mu_mc_std[finite],
            fmt="ro", ms=6, lw=1.5, capsize=4,
            label="MC (Widom)", zorder=6,
        )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\beta\mu_{\rm ex}$")
    ax.set_title(r"Comparison 2: Excess Chemical Potential")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    figpath = outdir / "comp2_mu_ex.png"
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {figpath}")

    # Print comparison table
    print(f"\n  {'eta':>6s}  {'μ_DFT':>9s}  {'μ_CS':>9s}  {'μ_MC':>9s}  "
          f"{'DFT-CS%':>8s}  {'DFT-MC%':>8s}")
    print("  " + "-" * 60)
    for i, eta in enumerate(eta_vals):
        dft_cs = 100*(mu_dft[i]/mu_cs[i] - 1.0) if abs(mu_cs[i]) > 1e-10 else np.nan
        dft_mc = (100*(mu_dft[i]/mu_mc[i] - 1.0)
                  if (np.isfinite(mu_mc[i]) and abs(mu_mc[i]) > 1e-10) else np.nan)
        print(f"  {eta:6.3f}  {mu_dft[i]:9.4f}  {mu_cs[i]:9.4f}  "
              f"{mu_mc[i]:9.4f}  {dft_cs:+8.3f}  {dft_mc:+8.3f}")

    return {
        "eta": eta_vals,
        "mu_dft": mu_dft.tolist(),
        "mu_cs":  mu_cs.tolist(),
        "mu_mc":  mu_mc.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════
# COMPARISON 3: Wall profiles — existing MD/MC data vs DFT Picard
# ═════════════════════════════════════════════════════════════════════

def comparison3_wall_profiles(functional, outdir, args):
    """Compare DFT Picard wall profiles against MC/MD reference data."""
    eta_targets = [0.367, 0.393, 0.449, 0.492]

    print("\n" + "=" * 68)
    print("  COMPARISON 3: Wall Profiles — MC/MD data vs DFT Picard")
    print("=" * 68)

    solver = WallSolver(nz=2048, Lz=8.0, R=0.5)
    n_panels = len(eta_targets)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    results = {}

    for i, eta in enumerate(eta_targets):
        ax = axes_flat[i]

        # ── DFT wall profile ────────────────────────────────────────
        A, B = functional.bulk_parameters(eta)
        fmt = esFMT_Tensor(A=float(A), B=float(B))
        try:
            res = solver.solve(eta, fmt, max_iter=6000, tol=1e-8, verbose=False)
            z_dft = np.array(res["z"])
            rho_norm_dft = np.array(res["rho_norm"])
            contact_dft = float(res["contact"])
            converged = res["converged"]
            ax.plot(z_dft, rho_norm_dft, "b-", lw=1.8, label="DFT (Picard)", zorder=3)
        except Exception as exc:
            print(f"  WARNING: DFT wall solver failed for eta={eta}: {exc}")
            contact_dft = np.nan
            converged = False

        results[eta] = {
            "contact_dft": contact_dft,
            "contact_cs": float(BT.Z_CS(eta)),
            "converged": converged,
        }

        # CS contact theorem line
        Z_cs = float(BT.Z_CS(eta))
        ax.axhline(Z_cs, color="gray", ls=":", lw=1.0, alpha=0.6,
                   label=f"CS contact = {Z_cs:.2f}")

        # ── Precomputed MC wall data (from --mc-data dir) ───────────
        loaded_mc = False
        if args.mc_data:
            mc_data_dir = Path(args.mc_data)
            wall_file = mc_data_dir / f"wall_eta{eta:.3f}.npz"
            if wall_file.exists():
                try:
                    data = np.load(wall_file)
                    z_mc = data["z"]
                    rho_mc = data["rho"] / float(data["rho_bulk"])
                    ax.plot(z_mc, rho_mc, "ko", ms=2, alpha=0.45,
                            label="MC (precomputed)", zorder=5)
                    results[eta]["contact_mc"] = float(data["rho"][np.argmin(
                        np.abs(z_mc - 0.5))] / float(data["rho_bulk"]))
                    loaded_mc = True
                    print(f"  eta={eta:.3f}: loaded MC wall data from {wall_file.name}")
                except Exception as exc:
                    print(f"  WARNING: Could not load {wall_file}: {exc}")

        # ── Built-in MD data (Davidchack 2016) ─────────────────────
        if not loaded_mc:
            mc = get_mc_data(eta)
            if mc is not None:
                z_mc = np.array(mc["z"])
                rho_mc = np.array(mc["rho"]) / mc["rho_bulk"]
                ax.plot(z_mc, rho_mc, "ko", ms=2, alpha=0.45,
                        label="MD (Davidchack 2016)", zorder=5)
                contact_idx = np.argmin(np.abs(z_mc - 0.5))
                results[eta]["contact_mc"] = float(rho_mc[contact_idx])
                loaded_mc = True

        if not loaded_mc:
            results[eta]["contact_mc"] = np.nan
            if args.quick:
                print(f"  eta={eta:.3f}: no MC data found (--quick mode, skipping live MC)")
            else:
                print(f"  eta={eta:.3f}: no MC data found")

        conv_str = "Y" if converged else "N"
        print(f"  eta={eta:.3f}  contact_DFT={contact_dft:.3f}  "
              f"contact_CS={Z_cs:.3f}  conv={conv_str}")

        ax.set_xlim(0, 5)
        ax.set_xlabel(r"$z / \sigma$")
        ax.set_ylabel(r"$\rho(z) / \rho_b$")
        ax.set_title(rf"$\eta = {eta}$"
                     + (" [converged]" if converged else " [NOT converged]"))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Comparison 3: Wall Density Profiles — DFT vs MC/MD", fontsize=13)
    fig.tight_layout()
    figpath = outdir / "comp3_wall_profiles.png"
    fig.savefig(figpath, dpi=200)
    plt.close(fig)
    print(f"\n  Saved: {figpath}")

    return results


# ═════════════════════════════════════════════════════════════════════
# COMPARISON 4: Summary table
# ═════════════════════════════════════════════════════════════════════

def comparison4_summary_table(functional, wall_results, gr_results, mu_results,
                               outdir, args):
    """Print and save summary table: Z, μ_ex, g(σ⁺) for DFT vs CS vs MC."""
    eta_vals = [0.1, 0.2, 0.3, 0.367, 0.4]

    print("\n" + "=" * 80)
    print("  COMPARISON 4: Summary Table  [DFT vs CS vs MC]")
    print("=" * 80)

    hdr = (f"  {'eta':>6s}  "
           f"{'Z_DFT':>8s}  {'Z_CS':>8s}  {'Z_MC':>8s}  "
           f"{'μ_DFT':>8s}  {'μ_CS':>8s}  {'μ_MC':>8s}  "
           f"{'g(σ+)_DFT':>10s}  {'g(σ+)_MC':>10s}")
    print(hdr)
    print("  " + "-" * 78)

    rows = []
    for eta in eta_vals:
        rho_bulk = 6.0 * eta / np.pi

        # Z from DFT (Lutsko with bulk A, B)
        try:
            A, B = functional.bulk_parameters(eta)
            Z_dft = float(BT.Z_lutsko(eta, A, B))
        except Exception:
            Z_dft = np.nan

        Z_cs = float(BT.Z_CS(eta))

        # Z_MC: use Z_CS as best available reference (contact theorem gives
        # g(σ+) = Z, so Z_MC ≈ g(σ+)_MC from wall profiles if available)
        Z_mc = np.nan
        if eta in wall_results and "contact_mc" in wall_results[eta]:
            Z_mc = wall_results[eta].get("contact_mc", np.nan)

        # μ_ex
        mu_dft_val = np.nan
        mu_cs_val  = float(BT.mu_ex_CS(eta))
        mu_mc_val  = np.nan

        if mu_results and "eta" in mu_results:
            try:
                idx = mu_results["eta"].index(eta)
                mu_dft_val = mu_results["mu_dft"][idx]
                mu_mc_val  = mu_results["mu_mc"][idx]
            except (ValueError, IndexError):
                pass

        # g(σ+) from bulk g(r) results
        g_contact_dft = np.nan
        g_contact_mc  = np.nan
        if eta in gr_results:
            g_contact_dft = gr_results[eta].get("g_contact_dft", np.nan)
            g_contact_mc  = gr_results[eta].get("g_contact_mc", np.nan)

        def _fmt(v):
            return f"{v:8.4f}" if np.isfinite(v) else f"{'N/A':>8s}"

        row = (f"  {eta:6.3f}  "
               f"{_fmt(Z_dft)}  {_fmt(Z_cs)}  {_fmt(Z_mc)}  "
               f"{_fmt(mu_dft_val)}  {_fmt(mu_cs_val)}  {_fmt(mu_mc_val)}  "
               f"{_fmt(g_contact_dft):>10s}  {_fmt(g_contact_mc):>10s}")
        print(row)
        rows.append(row)

    print("=" * 80)

    # Save
    tbl_path = outdir / "summary_table.txt"
    with open(tbl_path, "w") as f:
        f.write("DFT vs MC Comparison Summary\n")
        f.write("=" * 80 + "\n")
        f.write(hdr + "\n")
        f.write("  " + "-" * 78 + "\n")
        for row in rows:
            f.write(row + "\n")
        f.write("=" * 80 + "\n")
    print(f"\n  Saved: {tbl_path}")

    # Bar-chart panel for Z comparison at wall eta values
    wall_etas = sorted(wall_results.keys())
    if wall_etas:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(wall_etas))
        w = 0.25

        z_dft_wall, z_cs_wall, z_mc_wall = [], [], []
        for eta in wall_etas:
            try:
                A, B = functional.bulk_parameters(eta)
                z_dft_wall.append(float(BT.Z_lutsko(eta, A, B)))
            except Exception:
                z_dft_wall.append(np.nan)
            z_cs_wall.append(float(BT.Z_CS(eta)))
            z_mc_wall.append(wall_results[eta].get("contact_mc", np.nan))

        ax.bar(x - w, z_dft_wall, w, label="DFT", color="steelblue", alpha=0.85)
        ax.bar(x,     z_cs_wall,  w, label="CS",  color="forestgreen", alpha=0.85)
        ax.bar(x + w, z_mc_wall,  w, label="MC/MD (contact)", color="tomato", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f"η={e}" for e in wall_etas])
        ax.set_ylabel(r"$Z = \rho(R^+)/\rho_b$")
        ax.set_title("Comparison 4: Contact Density / Z")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        figpath = outdir / "comp4_summary_Z.png"
        fig.savefig(figpath, dpi=200)
        plt.close(fig)
        print(f"  Saved: {figpath}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  DFT vs MC COMPARISON — Nonlocal Neural Functional")
    print("=" * 68)
    print(f"  Checkpoint:    {args.checkpoint or '(fresh, untrained)'}")
    print(f"  Output dir:    {outdir}")
    print(f"  Quick mode:    {args.quick}")
    print(f"  MC data dir:   {args.mc_data or '(none)'}")
    print(f"  Grid:          {args.grid_size}³, L = {args.box_length} σ")
    if not args.quick:
        print(f"  MC: N={args.n_mc_particles}, "
              f"equil={args.n_mc_equil}, prod={args.n_mc_prod} sweeps")

    # ── Build grid and functional ──────────────────────────────────
    t0 = time.time()
    grid = Grid((args.grid_size,) * 3, args.box_length)

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"\n  ERROR: checkpoint not found: {ckpt}")
            sys.exit(1)
        print(f"\n  Loading checkpoint: {ckpt}")
        functional = load_functional(str(ckpt), grid, args.seed)
    else:
        print("\n  Creating fresh (untrained) functional...")
        functional = create_functional(grid, seed=args.seed)

    print(f"  Functional ready ({time.time()-t0:.1f}s)")

    # ── Comparison 1: bulk g(r) ────────────────────────────────────
    t1 = time.time()
    gr_results = comparison1_bulk_gr(functional, grid, outdir, args)
    print(f"\n  Comparison 1 elapsed: {time.time()-t1:.1f}s")

    # ── Comparison 2: μ_ex ────────────────────────────────────────
    t2 = time.time()
    mu_results = comparison2_mu_ex(functional, outdir, args)
    print(f"\n  Comparison 2 elapsed: {time.time()-t2:.1f}s")

    # ── Comparison 3: wall profiles ───────────────────────────────
    t3 = time.time()
    wall_results = comparison3_wall_profiles(functional, outdir, args)
    print(f"\n  Comparison 3 elapsed: {time.time()-t3:.1f}s")

    # ── Comparison 4: summary table ───────────────────────────────
    comparison4_summary_table(
        functional, wall_results, gr_results, mu_results, outdir, args
    )

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  All outputs in: {outdir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
