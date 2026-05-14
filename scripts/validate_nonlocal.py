#!/usr/bin/env python
"""
Validate Nonlocal Neural Functional
=====================================

Comprehensive 5-level validation hierarchy for a trained (or fresh)
NonlocalLutskoFunctional with spatially varying A(r), B(r).

Levels
------
1. Bulk Thermodynamics   Z, mu, chi vs Carnahan-Starling across eta
2. Wall Profiles         1D Picard profiles at 4 eta values vs MD + CS contact
3. Direct Correlation    c2(k) in Fourier space vs PY analytical
4. Structure & g(r)      S(k) and g(r) from OZ vs PY reference
5. Scorecard             Summary table with pass/fail verdicts

Outputs (to --output-dir):
  - level1_bulk_thermodynamics.{png,txt}
  - level1_AB_parameters.png
  - level2_wall_profiles.png
  - level3_c2_fourier.png
  - level3_c2_real.png
  - level4_structure_factor.png
  - level4_pair_correlation.png
  - scorecard.txt

Usage:
    python -m scripts.validate_nonlocal --checkpoint path/to/ckpt.eqx
    python -m scripts.validate_nonlocal --quick              # fast, no ckpt
    python -m scripts.validate_nonlocal --levels 1,2,5       # run subset
    python -m scripts.validate_nonlocal --grid-size 48 --box-length 12.0
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
from correlations.direct import compute_c2_bulk, compute_c2_fourier, compute_c2_radial
from correlations.ornstein_zernike import (
    solve_oz_fourier,
    compute_structure_factor,
    compute_pair_correlation,
    radial_average,
    compute_g_radial,
)
from validation.percus_yevick import (
    py_direct_correlation,
    py_direct_correlation_fourier,
    py_structure_factor,
    py_contact_value,
    py_compressibility,
    compute_py_g_numerical,
)
from solvers.fmt_1d_wbii_tensor import WallSolver, esFMT_Tensor
from solvers.wall_profile import get_mc_data


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate a nonlocal neural functional (5-level hierarchy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained .eqx checkpoint. "
             "If omitted, a fresh (untrained) functional is used.",
    )
    p.add_argument(
        "--output-dir", type=str, default="outputs/validate_nonlocal",
        help="Directory for plots and results (default: outputs/validate_nonlocal)",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Quick mode: smaller grid, fewer eta points, relaxed solver",
    )
    p.add_argument(
        "--grid-size", type=int, default=None,
        help="3D grid points per dimension (default: 32, or 16 with --quick)",
    )
    p.add_argument(
        "--box-length", type=float, default=10.0,
        help="Cubic box side length in sigma (default: 10.0)",
    )
    p.add_argument(
        "--levels", type=str, default="1,2,3,4,5",
        help="Comma-separated list of levels to run (default: 1,2,3,4,5)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for fresh functional (default: 42)",
    )
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# FUNCTIONAL SETUP
# ═════════════════════════════════════════════════════════════════════

def create_functional(grid, seed=42):
    """Create a fresh NonlocalLutskoFunctional with random weights."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    kernels = FMTKernels(grid, R=0.5)
    calculator = WeightedDensityCalculator(kernels)
    network = NonlocalConditionalNetwork(k1)
    kernel = LearnableKernel(k2)

    return NonlocalLutskoFunctional(network, kernel, calculator, grid)


def load_functional(checkpoint_path, grid, seed=42):
    """Load a trained functional from an Equinox checkpoint."""
    template = create_functional(grid, seed)
    functional = eqx.tree_deserialise_leaves(checkpoint_path, template)
    return functional


# ═════════════════════════════════════════════════════════════════════
# PASS/FAIL THRESHOLDS (percentage errors)
# ═════════════════════════════════════════════════════════════════════

THRESHOLDS = {
    "Z_max_err_pct":       0.1,   # Z within 0.1% of CS
    "mu_max_err_pct":      0.1,   # mu_ex within 0.1% of CS
    "chi_max_err_pct":     0.5,   # chi_T within 0.5% of CS
    "contact_max_err_pct": 5.0,   # Wall contact within 5% of CS
    "all_converged":       True,  # All wall profiles converged
    "c2k_max_err_pct":     25.0,  # c2(k) within 25% of PY (peak region)
    "Sk_peak_err_pct":     25.0,  # S(k) first-peak height within 25%
    "gr_contact_err_pct":  25.0,  # g(r) contact region within 25%
}


# ═════════════════════════════════════════════════════════════════════
# HELPER: Radial binning of a 3D Fourier-space field by |k|
# ═════════════════════════════════════════════════════════════════════

def _radial_bin_fourier(field_3d, grid, n_bins=150):
    """
    Radially average a 3D field defined on the Fourier grid by |k|.

    Returns (k_centers, f_radial, counts) where counts is per-bin
    sample count (useful for masking empty bins).
    """
    k_abs = np.array(grid.k_abs)
    k_max = float(np.pi / grid.dx)  # Nyquist frequency
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
# LEVEL 1: BULK THERMODYNAMICS
# ═════════════════════════════════════════════════════════════════════

def level1_bulk_thermodynamics(functional, outdir, quick=False):
    """
    Validate Z, mu_ex, chi_T vs Carnahan-Starling across eta in [0.05, 0.50].

    Returns dict with max percentage errors for Z, mu, chi.
    """
    n_pts = 10 if quick else 20
    eta_vals = np.linspace(0.05, 0.50, n_pts)

    print("\n" + "=" * 72)
    print("  LEVEL 1: Bulk Thermodynamics vs Carnahan-Starling")
    print("=" * 72)
    header = (f"  {'eta':>6s}  {'A':>7s}  {'B':>7s}  {'C':>8s}  "
              f"{'Z_NN':>8s}  {'Z_CS':>8s}  {'Z%':>7s}  "
              f"{'mu%':>7s}  {'chi%':>7s}")
    print(header)
    print("  " + "-" * 70)

    rows = []
    Z_errs, mu_errs, chi_errs = [], [], []

    for eta in eta_vals:
        A, B = functional.bulk_parameters(eta)
        A_f, B_f = float(A), float(B)
        C = 8.0 * A_f + 2.0 * B_f - 9.0

        z_nn = float(BT.Z_lutsko(eta, A, B))
        z_cs = float(BT.Z_CS(eta))
        z_err = 100.0 * (z_nn / z_cs - 1.0)

        m_nn = float(BT.mu_ex_bulk_lutsko(eta, A, B))
        m_cs = float(BT.mu_ex_CS(eta))
        m_err = 100.0 * (m_nn / m_cs - 1.0) if abs(m_cs) > 1e-10 else 0.0

        x_nn = float(BT.chi_T_bulk_lutsko(eta, A, B))
        x_cs = float(BT.chi_T_CS(eta))
        x_err = 100.0 * (x_nn / x_cs - 1.0) if abs(x_cs) > 1e-10 else 0.0

        Z_errs.append(abs(z_err))
        mu_errs.append(abs(m_err))
        chi_errs.append(abs(x_err))

        rows.append(dict(
            eta=eta, A=A_f, B=B_f, C=C,
            Z_nn=z_nn, Z_cs=z_cs, z_err=z_err,
            mu_nn=m_nn, mu_cs=m_cs, m_err=m_err,
            chi_nn=x_nn, chi_cs=x_cs, x_err=x_err,
        ))

        print(f"  {eta:6.3f}  {A_f:7.4f}  {B_f:7.4f}  {C:+8.4f}  "
              f"{z_nn:8.4f}  {z_cs:8.4f}  {z_err:+7.3f}  "
              f"{m_err:+7.3f}  {x_err:+7.3f}")

    max_Z = max(Z_errs)
    max_mu = max(mu_errs)
    max_chi = max(chi_errs)

    print("  " + "-" * 70)
    print(f"  Max |Z err|:   {max_Z:.4f}%  "
          f"(threshold: {THRESHOLDS['Z_max_err_pct']}%)")
    print(f"  Max |mu err|:  {max_mu:.4f}%  "
          f"(threshold: {THRESHOLDS['mu_max_err_pct']}%)")
    print(f"  Max |chi err|: {max_chi:.4f}%  "
          f"(threshold: {THRESHOLDS['chi_max_err_pct']}%)")

    # ── Save table ──
    table_path = outdir / "level1_bulk_thermodynamics.txt"
    with open(table_path, "w") as f:
        f.write(f"{'eta':>8s}  {'A':>10s}  {'B':>10s}  {'C':>10s}  "
                f"{'Z_NN':>10s}  {'Z_CS':>10s}  {'Z_err%':>10s}  "
                f"{'mu_NN':>10s}  {'mu_CS':>10s}  {'mu_err%':>10s}  "
                f"{'chi_NN':>10s}  {'chi_CS':>10s}  {'chi_err%':>10s}\n")
        for r in rows:
            f.write(f"{r['eta']:8.4f}  {r['A']:10.6f}  {r['B']:10.6f}  "
                    f"{r['C']:+10.6f}  {r['Z_nn']:10.6f}  {r['Z_cs']:10.6f}  "
                    f"{r['z_err']:+10.4f}  {r['mu_nn']:10.6f}  {r['mu_cs']:10.6f}  "
                    f"{r['m_err']:+10.4f}  {r['chi_nn']:10.6f}  {r['chi_cs']:10.6f}  "
                    f"{r['x_err']:+10.4f}\n")
    print(f"\n  Saved: {table_path}")

    # ── Plot: Z, mu, chi ──
    etas = np.array([r["eta"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].plot(etas, [r["Z_cs"] for r in rows], "k-", lw=2, label="CS")
    axes[0].plot(etas, [r["Z_nn"] for r in rows], "ro-", ms=4, lw=1, label="Neural")
    axes[0].set_xlabel(r"$\eta$")
    axes[0].set_ylabel(r"$Z$")
    axes[0].set_title("Compressibility Factor")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(etas, [r["mu_cs"] for r in rows], "k-", lw=2, label="CS")
    axes[1].plot(etas, [r["mu_nn"] for r in rows], "ro-", ms=4, lw=1, label="Neural")
    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel(r"$\beta\mu_{\rm ex}$")
    axes[1].set_title("Excess Chemical Potential")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(etas, [r["chi_cs"] for r in rows], "k-", lw=2, label="CS")
    axes[2].plot(etas, [r["chi_nn"] for r in rows], "gs-", ms=4, lw=1, label="Neural")
    axes[2].set_xlabel(r"$\eta$")
    axes[2].set_ylabel(r"$\chi_T$")
    axes[2].set_title("Isothermal Compressibility")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    plt.suptitle("Level 1: Bulk Thermodynamics vs Carnahan-Starling", fontsize=13)
    plt.tight_layout()
    figpath = outdir / "level1_bulk_thermodynamics.png"
    plt.savefig(figpath, dpi=200)
    plt.close()
    print(f"  Saved: {figpath}")

    # ── Plot: A(eta), B(eta), C(eta) parameter curves ──
    etas_dense = np.linspace(0.01, 0.52, 100)
    As, Bs, Cs = [], [], []
    for eta in etas_dense:
        A, B = functional.bulk_parameters(eta)
        As.append(float(A))
        Bs.append(float(B))
        Cs.append(8.0 * float(A) + 2.0 * float(B) - 9.0)

    fig2, ax2 = plt.subplots(1, 3, figsize=(13, 4))

    ax2[0].plot(etas_dense, As, "b-", lw=2)
    ax2[0].set_xlabel(r"$\eta$")
    ax2[0].set_ylabel(r"$A(\eta)$")
    ax2[0].set_title("Learned A parameter")
    ax2[0].grid(True, alpha=0.2)

    ax2[1].plot(etas_dense, Bs, "r-", lw=2)
    ax2[1].set_xlabel(r"$\eta$")
    ax2[1].set_ylabel(r"$B(\eta)$")
    ax2[1].set_title("Learned B parameter")
    ax2[1].grid(True, alpha=0.2)

    ax2[2].plot(etas_dense, Cs, "k-", lw=2)
    ax2[2].axhline(-3, color="g", ls=":", label="CS (C=-3)")
    ax2[2].axhline(0, color="b", ls=":", label="PY (C=0)")
    ax2[2].axhline(-0.6, color="orange", ls=":", label=r"G\"ul (C=-0.6)")
    ax2[2].set_xlabel(r"$\eta$")
    ax2[2].set_ylabel(r"$C(\eta) = 8A + 2B - 9$")
    ax2[2].set_title("Constraint Parameter")
    ax2[2].legend(fontsize=8)
    ax2[2].grid(True, alpha=0.2)

    plt.suptitle("Level 1: Learned Bulk Parameters", fontsize=13)
    plt.tight_layout()
    figpath2 = outdir / "level1_AB_parameters.png"
    plt.savefig(figpath2, dpi=200)
    plt.close()
    print(f"  Saved: {figpath2}")

    return {
        "Z_max_err_pct": max_Z,
        "mu_max_err_pct": max_mu,
        "chi_max_err_pct": max_chi,
    }


# ═════════════════════════════════════════════════════════════════════
# LEVEL 2: WALL PROFILES
# ═════════════════════════════════════════════════════════════════════

def level2_wall_profiles(functional, outdir, quick=False):
    """
    Solve 1D wall profiles at eta = [0.367, 0.393, 0.449, 0.492] and
    compare contact density to CS prediction and MD data.

    Returns dict with max contact error and convergence status.
    """
    eta_targets = [0.367, 0.393, 0.449, 0.492]
    solver_nz = 1024 if quick else 2048
    solver_max_iter = 3000 if quick else 6000
    solver = WallSolver(nz=solver_nz, Lz=8.0, R=0.5)

    print("\n" + "=" * 72)
    print("  LEVEL 2: Wall Density Profiles vs MD Data")
    print("=" * 72)
    print(f"  {'eta':>6s}  {'contact_NN':>12s}  {'contact_CS':>12s}  "
          f"{'err%':>8s}  {'conv':>5s}")
    print("  " + "-" * 55)

    profile_results = []
    contact_errs = []

    for eta in eta_targets:
        A, B = functional.bulk_parameters(eta)
        fmt = esFMT_Tensor(A=float(A), B=float(B))
        res = solver.solve(eta, fmt, max_iter=solver_max_iter,
                           tol=1e-8, verbose=False)
        profile_results.append(res)

        err_pct = 100.0 * (res["contact"] / res["contact_CS"] - 1.0)
        contact_errs.append(abs(err_pct))
        conv = "Y" if res["converged"] else "N"
        print(f"  {eta:6.3f}  {res['contact']:12.4f}  "
              f"{res['contact_CS']:12.4f}  {err_pct:+8.2f}  {conv:>5s}")

    max_err = max(contact_errs)
    all_conv = all(r["converged"] for r in profile_results)
    print("  " + "-" * 55)
    print(f"  Max |contact err|: {max_err:.2f}%  "
          f"(threshold: {THRESHOLDS['contact_max_err_pct']}%)")
    print(f"  All converged: {all_conv}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (eta, res) in enumerate(zip(eta_targets, profile_results)):
        ax = axes[i]
        z = np.array(res["z"])
        rho_norm = np.array(res["rho_norm"])
        ax.plot(z, rho_norm, "b-", lw=1.5, label="Nonlocal NN")

        mc = get_mc_data(eta)
        if mc is not None:
            z_mc = np.array(mc["z"])
            rho_mc = np.array(mc["rho"]) / mc["rho_bulk"]
            ax.plot(z_mc, rho_mc, "ko", ms=2, alpha=0.5, label="MD data")

        ax.axhline(res["contact_CS"], color="g", ls=":", lw=1,
                    alpha=0.6, label=f"CS contact={res['contact_CS']:.2f}")

        ax.set_xlim(0, 5)
        ax.set_xlabel(r"$z / \sigma$")
        ax.set_ylabel(r"$\rho(z) / \rho_b$")
        ax.set_title(rf"$\eta = {eta}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Level 2: Wall Density Profiles (Davidchack 2016 MD)", fontsize=13)
    plt.tight_layout()
    figpath = outdir / "level2_wall_profiles.png"
    plt.savefig(figpath, dpi=200)
    plt.close()
    print(f"\n  Saved: {figpath}")

    return {
        "contact_max_err_pct": max_err,
        "contact_mean_err_pct": float(np.mean(contact_errs)),
        "all_converged": all_conv,
        "per_eta": {
            eta: {
                "contact": res["contact"],
                "contact_CS": res["contact_CS"],
                "err_pct": 100.0 * (res["contact"] / res["contact_CS"] - 1.0),
                "converged": res["converged"],
            }
            for eta, res in zip(eta_targets, profile_results)
        },
    }


# ═════════════════════════════════════════════════════════════════════
# LEVEL 3: DIRECT CORRELATION c2
# ═════════════════════════════════════════════════════════════════════

def level3_direct_correlation(functional, grid, outdir, quick=False):
    """
    Compute c2(k) in Fourier space at several eta values and compare
    to PY analytical.  Also plot c2(r) in real space.

    Returns dict with max relative error in c2(k) peak region.
    """
    eta_vals = [0.1, 0.3] if quick else [0.1, 0.2, 0.3, 0.4]
    n_bins_k = 80 if quick else 150

    print("\n" + "=" * 72)
    print("  LEVEL 3: Direct Correlation Function c2(k) vs PY")
    print("=" * 72)

    n_cols = len(eta_vals)
    fig_k, axes_k = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    fig_r, axes_r = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes_k = [axes_k]
        axes_r = [axes_r]

    c2k_errs = []

    for idx, eta in enumerate(eta_vals):
        rho_bulk = 6.0 * eta / np.pi

        print(f"\n  eta = {eta:.2f}, rho = {rho_bulk:.4f}")
        t0 = time.time()

        # Neural c2(k) via forward-over-reverse autodiff
        c2_k_3d = compute_c2_fourier(functional, rho_bulk, grid)
        elapsed = time.time() - t0
        print(f"    c2(k) computed in {elapsed:.1f}s")

        # Radial average in Fourier space
        k_centers, c2k_radial, counts_k = _radial_bin_fourier(
            c2_k_3d, grid, n_bins=n_bins_k
        )

        # PY analytical
        k_arr = jnp.array(k_centers)
        c2k_py = np.array(py_direct_correlation_fourier(k_arr, eta, sigma=1.0))

        # Relative error in physically relevant range k*sigma in [1, 20]
        mask = (k_centers > 1.0) & (k_centers < 20.0) & (counts_k > 0)
        if mask.any():
            denom = np.maximum(np.abs(c2k_py[mask]), 1e-10)
            rel_err = np.abs(c2k_radial[mask] - c2k_py[mask]) / denom
            max_err = 100.0 * float(np.max(rel_err))
            mean_err = 100.0 * float(np.mean(rel_err))
        else:
            max_err, mean_err = np.nan, np.nan

        c2k_errs.append(max_err)
        print(f"    c2(k) vs PY: max_rel_err = {max_err:.1f}%, "
              f"mean_rel_err = {mean_err:.1f}%")

        # Fourier-space plot
        ax = axes_k[idx]
        ax.plot(k_centers, c2k_py, "k-", lw=1.5, label="PY exact")
        ax.plot(k_centers, c2k_radial, "b--", lw=1.2, label="Neural")
        ax.set_xlabel(r"$k\sigma$")
        ax.set_ylabel(r"$\hat{c}_2(k)$")
        ax.set_title(rf"$\eta = {eta}$  (max err {max_err:.1f}%)")
        ax.set_xlim(0, 30)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # ── Real-space c2(r) ──
        c2_real_3d = compute_c2_bulk(functional, rho_bulk, grid)
        r_vals, c2_r = compute_c2_radial(c2_real_3d, grid, n_bins=150)
        r_vals = np.array(r_vals)
        c2_r = np.array(c2_r)

        r_fine = jnp.linspace(0.01, min(float(grid.Lx) / 2.0, 3.0), 300)
        c2_py_r = np.array(py_direct_correlation(r_fine, eta, sigma=1.0))

        ax_r = axes_r[idx]
        ax_r.plot(np.array(r_fine), c2_py_r, "k-", lw=1.5, label="PY exact")
        ax_r.plot(r_vals, c2_r, "b--", lw=1.2, label="Neural")
        ax_r.axvline(1.0, color="gray", ls=":", alpha=0.5, label=r"$r=\sigma$")
        ax_r.set_xlabel(r"$r / \sigma$")
        ax_r.set_ylabel(r"$c_2(r)$")
        ax_r.set_title(rf"$\eta = {eta}$")
        ax_r.set_xlim(0, 3)
        ax_r.legend(fontsize=8)
        ax_r.grid(True, alpha=0.2)

    fig_k.suptitle(r"Level 3: $\hat{c}_2(k)$ in Fourier Space", fontsize=13)
    fig_k.tight_layout()
    figpath_k = outdir / "level3_c2_fourier.png"
    fig_k.savefig(figpath_k, dpi=200)
    plt.close(fig_k)
    print(f"\n  Saved: {figpath_k}")

    fig_r.suptitle(r"Level 3: $c_2(r)$ in Real Space", fontsize=13)
    fig_r.tight_layout()
    figpath_r = outdir / "level3_c2_real.png"
    fig_r.savefig(figpath_r, dpi=200)
    plt.close(fig_r)
    print(f"  Saved: {figpath_r}")

    overall_max = max(c2k_errs) if c2k_errs else np.nan
    print(f"\n  Overall max c2(k) error: {overall_max:.1f}%  "
          f"(threshold: {THRESHOLDS['c2k_max_err_pct']}%)")

    return {
        "c2k_max_err_pct": overall_max,
        "c2k_mean_err_pct": float(np.nanmean(c2k_errs)),
        "per_eta": {eta: err for eta, err in zip(eta_vals, c2k_errs)},
    }


# ═════════════════════════════════════════════════════════════════════
# LEVEL 4: STRUCTURE FACTOR & g(r)
# ═════════════════════════════════════════════════════════════════════

def level4_structure_and_gr(functional, grid, outdir, quick=False):
    """
    Compute S(k) and g(r) from OZ equation using neural c2, compare to PY.

    Returns dict with max errors in S(k) peak height and g(r) contact.
    """
    eta_vals = [0.1, 0.3] if quick else [0.1, 0.2, 0.3, 0.4]
    n_bins_k = 80 if quick else 150

    print("\n" + "=" * 72)
    print("  LEVEL 4: Structure Factor S(k) & Pair Correlation g(r)")
    print("=" * 72)

    n_cols = len(eta_vals)
    fig_s, axes_s = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    fig_g, axes_g = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4.5))
    if n_cols == 1:
        axes_s = [axes_s]
        axes_g = [axes_g]

    Sk_errs = []
    gr_errs = []

    for idx, eta in enumerate(eta_vals):
        rho_bulk = 6.0 * eta / np.pi

        print(f"\n  eta = {eta:.2f}, rho = {rho_bulk:.4f}")
        t0 = time.time()

        # Neural c2(k), then S(k)
        c2_k = compute_c2_fourier(functional, rho_bulk, grid)
        S_k_3d = compute_structure_factor(c2_k, rho_bulk)

        # Radial average S(k)
        k_centers, Sk_radial, counts_k = _radial_bin_fourier(
            S_k_3d, grid, n_bins=n_bins_k
        )

        # PY S(k) reference
        k_arr = jnp.array(k_centers)
        Sk_py = np.array(py_structure_factor(k_arr, eta, sigma=1.0))

        # First-peak error: k*sigma in [4, 10]
        mask_peak = (k_centers > 4.0) & (k_centers < 10.0) & (counts_k > 0)
        if mask_peak.any():
            denom = np.maximum(np.abs(Sk_py[mask_peak]), 1e-6)
            rel_err_Sk = np.abs(Sk_radial[mask_peak] - Sk_py[mask_peak]) / denom
            Sk_max_err = 100.0 * float(np.max(rel_err_Sk))
        else:
            Sk_max_err = np.nan

        Sk_errs.append(Sk_max_err)

        # Compressibility sum rule: S(k=0)
        S0_nn = float(jnp.real(S_k_3d[0, 0, 0]))
        S0_py = float(py_compressibility(eta))
        S0_err = 100.0 * abs(S0_nn / S0_py - 1.0) if abs(S0_py) > 1e-12 else 0.0
        print(f"    S(k) peak err: {Sk_max_err:.1f}%  |  "
              f"S(0): NN={S0_nn:.6f}, PY={S0_py:.6f} ({S0_err:.1f}%)")

        # Plot S(k)
        ax = axes_s[idx]
        ax.plot(k_centers, Sk_py, "k-", lw=1.5, label="PY exact")
        ax.plot(k_centers, Sk_radial, "b--", lw=1.2, label="Neural")
        ax.set_xlabel(r"$k\sigma$")
        ax.set_ylabel(r"$S(k)$")
        ax.set_title(rf"$\eta = {eta}$  (peak err {Sk_max_err:.1f}%)")
        ax.set_xlim(0, 30)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # ── g(r) via OZ ──
        n_bins_r = 100 if quick else 200
        r_nn, g_nn = compute_g_radial(c2_k, rho_bulk, grid, n_bins=n_bins_r)
        r_nn = np.array(r_nn)
        g_nn = np.array(g_nn)

        elapsed = time.time() - t0

        # PY g(r) reference (numerical inverse FT)
        r_for_py = jnp.linspace(0.5, min(float(grid.Lx) / 2.0, 4.0), 300)
        g_py = np.array(compute_py_g_numerical(
            r_for_py, eta, sigma=1.0,
            k_max=100.0 if quick else 200.0,
            n_k=5000 if quick else 15000,
        ))
        r_for_py_np = np.array(r_for_py)

        # Contact-region error: r/sigma in [1.0, 2.0]
        mask_contact = (r_for_py_np > 1.0) & (r_for_py_np < 2.0)
        if mask_contact.any():
            g_nn_interp = np.interp(r_for_py_np[mask_contact], r_nn, g_nn)
            g_py_contact = g_py[mask_contact]
            denom = np.maximum(np.abs(g_py_contact), 1e-6)
            rel_err_g = np.abs(g_nn_interp - g_py_contact) / denom
            gr_max_err = 100.0 * float(np.max(rel_err_g))
        else:
            gr_max_err = np.nan

        gr_errs.append(gr_max_err)
        print(f"    g(r) contact err: {gr_max_err:.1f}%  "
              f"(computed in {elapsed:.1f}s)")

        # Plot g(r)
        g_contact_py = float(py_contact_value(eta))
        ax = axes_g[idx]
        ax.plot(r_for_py_np, g_py, "k-", lw=1.5, label="PY (numerical)")
        ax.plot(r_nn, g_nn, "b--", lw=1.2, label="Neural (OZ)")
        ax.axvline(1.0, color="gray", ls=":", alpha=0.5, label=r"$r=\sigma$")
        ax.plot(1.0, g_contact_py, "r^", ms=8,
                label=f"PY contact = {g_contact_py:.2f}")
        ax.set_xlabel(r"$r / \sigma$")
        ax.set_ylabel(r"$g(r)$")
        ax.set_title(rf"$\eta = {eta}$  (contact err {gr_max_err:.1f}%)")
        ax.set_xlim(0.4, 4)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig_s.suptitle("Level 4: Structure Factor S(k)", fontsize=13)
    fig_s.tight_layout()
    figpath_s = outdir / "level4_structure_factor.png"
    fig_s.savefig(figpath_s, dpi=200)
    plt.close(fig_s)
    print(f"\n  Saved: {figpath_s}")

    fig_g.suptitle("Level 4: Pair Distribution g(r)", fontsize=13)
    fig_g.tight_layout()
    figpath_g = outdir / "level4_pair_correlation.png"
    fig_g.savefig(figpath_g, dpi=200)
    plt.close(fig_g)
    print(f"  Saved: {figpath_g}")

    Sk_overall = max(Sk_errs) if Sk_errs else np.nan
    gr_overall = max(gr_errs) if gr_errs else np.nan
    print(f"\n  Overall max S(k) peak err: {Sk_overall:.1f}%  "
          f"(threshold: {THRESHOLDS['Sk_peak_err_pct']}%)")
    print(f"  Overall max g(r) contact err: {gr_overall:.1f}%  "
          f"(threshold: {THRESHOLDS['gr_contact_err_pct']}%)")

    return {
        "Sk_peak_err_pct": Sk_overall,
        "gr_contact_err_pct": gr_overall,
        "Sk_mean_err_pct": float(np.nanmean(Sk_errs)),
        "gr_mean_err_pct": float(np.nanmean(gr_errs)),
    }


# ═════════════════════════════════════════════════════════════════════
# LEVEL 5: SCORECARD
# ═════════════════════════════════════════════════════════════════════

def level5_scorecard(all_results, outdir):
    """
    Print and save a summary scorecard with pass/fail for each test.

    Parameters
    ----------
    all_results : dict
        Merged results from levels 1--4.
    outdir : Path
        Output directory.

    Returns
    -------
    dict with n_pass, n_total, overall ("PASS" or "FAIL").
    """
    print("\n" + "=" * 72)
    print("  LEVEL 5: VALIDATION SCORECARD")
    print("=" * 72)

    lines = []
    n_pass = 0
    n_total = 0

    def _check(label, key, unit="%"):
        """Evaluate a single metric against its threshold."""
        nonlocal n_pass, n_total
        val = all_results.get(key)
        thresh = THRESHOLDS.get(key)

        if val is None:
            status = "SKIP"
            detail = "level not run"
        elif isinstance(thresh, bool):
            passed = bool(val) == thresh
            status = "PASS" if passed else "FAIL"
            detail = str(val)
            n_total += 1
            if passed:
                n_pass += 1
        else:
            if val is not None and np.isnan(val):
                status = "WARN"
                detail = "NaN (could not compute)"
            else:
                passed = val <= thresh
                status = "PASS" if passed else "FAIL"
                detail = f"{val:.2f}{unit}  (threshold: {thresh}{unit})"
                n_total += 1
                if passed:
                    n_pass += 1

        line = f"  [{status:4s}]  {label:<45s}  {detail}"
        lines.append(line)
        print(line)

    print()
    print("  --- Level 1: Bulk Thermodynamics ---")
    _check("Z vs Carnahan-Starling (max err)",     "Z_max_err_pct")
    _check("mu_ex vs Carnahan-Starling (max err)",  "mu_max_err_pct")
    _check("chi_T vs Carnahan-Starling (max err)",  "chi_max_err_pct")

    print()
    print("  --- Level 2: Wall Profiles ---")
    _check("Contact density vs CS (max err)",       "contact_max_err_pct")
    _check("All wall profiles converged",           "all_converged")

    print()
    print("  --- Level 3: Direct Correlation ---")
    _check("c2(k) vs PY in peak region (max err)",  "c2k_max_err_pct")

    print()
    print("  --- Level 4: Structure Factor & g(r) ---")
    _check("S(k) first peak vs PY (max err)",       "Sk_peak_err_pct")
    _check("g(r) contact region vs PY (max err)",   "gr_contact_err_pct")

    # Summary
    pct = 100.0 * n_pass / n_total if n_total > 0 else 0.0
    overall = "PASS" if n_pass == n_total else "FAIL"

    print()
    summary_sep = "=" * 60
    print(f"  {summary_sep}")
    print(f"  OVERALL: {n_pass}/{n_total} tests passed ({pct:.0f}%) -- {overall}")
    print(f"  {summary_sep}")

    # Save scorecard
    card_path = outdir / "scorecard.txt"
    with open(card_path, "w") as f:
        f.write("NONLOCAL NEURAL FUNCTIONAL -- VALIDATION SCORECARD\n")
        f.write("=" * 62 + "\n\n")
        for line in lines:
            f.write(line + "\n")
        f.write(f"\nOVERALL: {n_pass}/{n_total} passed ({pct:.0f}%) -- {overall}\n")
    print(f"\n  Saved: {card_path}")

    return {"n_pass": n_pass, "n_total": n_total, "overall": overall}


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    levels = set(int(x.strip()) for x in args.levels.split(","))

    grid_n = args.grid_size or (16 if args.quick else 32)
    box_l = args.box_length

    print("=" * 72)
    print("  NONLOCAL NEURAL FUNCTIONAL -- COMPREHENSIVE VALIDATION")
    print("=" * 72)
    print(f"  Checkpoint:  {args.checkpoint or '(fresh, untrained)'}")
    print(f"  Output:      {outdir}")
    print(f"  Grid:        {grid_n}^3, L = {box_l} sigma")
    print(f"  Quick:       {args.quick}")
    print(f"  Levels:      {sorted(levels)}")

    # ── Build grid and functional ──
    t0 = time.time()
    grid = Grid((grid_n, grid_n, grid_n), box_l)

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

    print(f"  Setup time: {time.time() - t0:.1f}s")

    # ── Run requested levels ──
    all_results = {}

    if 1 in levels:
        t1 = time.time()
        r1 = level1_bulk_thermodynamics(functional, outdir, quick=args.quick)
        all_results.update(r1)
        print(f"\n  Level 1 elapsed: {time.time() - t1:.1f}s")

    if 2 in levels:
        t2 = time.time()
        r2 = level2_wall_profiles(functional, outdir, quick=args.quick)
        all_results.update(r2)
        print(f"\n  Level 2 elapsed: {time.time() - t2:.1f}s")

    if 3 in levels:
        t3 = time.time()
        r3 = level3_direct_correlation(functional, grid, outdir, quick=args.quick)
        all_results.update(r3)
        print(f"\n  Level 3 elapsed: {time.time() - t3:.1f}s")

    if 4 in levels:
        t4 = time.time()
        r4 = level4_structure_and_gr(functional, grid, outdir, quick=args.quick)
        all_results.update(r4)
        print(f"\n  Level 4 elapsed: {time.time() - t4:.1f}s")

    if 5 in levels:
        level5_scorecard(all_results, outdir)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  All outputs in: {outdir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
