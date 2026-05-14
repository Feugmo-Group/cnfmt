#!/usr/bin/env python
"""
Compare Nonlocal Neural Functional Against Other Approaches
=============================================================

Produces a 2x2 comparison figure plus individual detailed plots:

  Panel 1: Bulk Thermodynamics — Z(eta) for all approaches vs CS reference
  Panel 2: Wall Profiles — rho(z)/rho_b at eta=0.449 vs MD data
  Panel 3: Contact Densities — rho_contact vs eta for all approaches vs CS
  Panel 4: Constraint Parameter — C(eta) = 8A + 2B - 9 for all approaches

Approaches compared:
  1. Nonlocal Neural  — NonlocalLutskoFunctional (loaded from checkpoint or fresh)
  2. White Bear II    — esFMT with A=1.0, B=-1.0 (C=-3, CS equation of state)
  3. Rosenfeld        — esFMT with A=1.0, B=0.5  (C=0, PY equation of state)
  4. Gul optimal      — esFMT with A=1.3, B=-1.0 (C=-0.6)
  5. Local Neural     — ConditionalNetwork (loaded from checkpoint, or skipped)

Usage:
    python -m scripts.compare_approaches
    python -m scripts.compare_approaches --checkpoint outputs/nonlocal/checkpoints/final.eqx
    python -m scripts.compare_approaches --output-dir results/compare
    python -m scripts.compare_approaches --eta-values 0.367,0.393,0.449,0.492
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

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_figure_style import apply_paper_style
apply_paper_style()

from core.grid import Grid
from core.weights import FMTKernels
from core.densities import WeightedDensityCalculator
from core.thermodynamics import BulkThermodynamics as BT
from nonlocal_ext.functional import NonlocalLutskoFunctional
from nonlocal_ext.kernels import LearnableKernel
from neural.network import NonlocalConditionalNetwork, ConditionalNetwork
from solvers.fmt_1d_wbii_tensor import (
    WallSolver, esFMT_Tensor, WhiteBearIIFMT, RosenfeldFMT,
)
from solvers.wall_profile import get_mc_data


PI = np.pi

# Reference MD contact densities (Davidchack, Laird, Roth 2016)
MD_CONTACTS = {0.367: 5.36, 0.393: 6.15, 0.449: 8.34, 0.492: 10.65}

# Fixed-parameter approaches: (A, B) values
#   WBII:      C = 8(1.0) + 2(-1.0) - 9 = -3   (Carnahan-Starling EOS)
#   Rosenfeld: C = 8(1.0) + 2(0.5)  - 9 =  0   (Percus-Yevick EOS)
#   Gul:       C = 8(1.3) + 2(-1.0) - 9 = -0.6 (empirically optimal)
FIXED_PARAMS = {
    "White Bear II": (1.0, -1.0),
    "Rosenfeld":     (1.0, 0.5),
    "Gul optimal":   (1.3, -1.0),
}

# Plot styles per approach
STYLES = {
    "Nonlocal NN":   dict(color="#1f77b4", ls="-",  lw=2.2, marker="o", ms=4),
    "Local NN":      dict(color="#ff7f0e", ls="--", lw=2.0, marker="s", ms=4),
    "White Bear II": dict(color="#2ca02c", ls="-.", lw=1.8, marker="^", ms=4),
    "Rosenfeld":     dict(color="#d62728", ls=":",  lw=1.8, marker="v", ms=4),
    "Gul optimal":   dict(color="#9467bd", ls="--", lw=1.8, marker="D", ms=4),
}


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare nonlocal neural functional against other approaches."
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained nonlocal model checkpoint (.eqx)")
    p.add_argument(
        "--local-checkpoint", type=str, default=None,
        help="Path to local neural network checkpoint (.eqx). "
             "Skipped if not provided.")
    p.add_argument(
        "--output-dir", type=str, default="outputs/comparison",
        help="Output directory (default: outputs/comparison)")
    p.add_argument(
        "--grid-size", type=int, default=32,
        help="3D grid size for nonlocal functional (default: 32)")
    p.add_argument(
        "--box-length", type=float, default=10.0,
        help="Box length in sigma (default: 10.0)")
    p.add_argument(
        "--eta-values", type=str, default="0.367,0.393,0.449,0.492",
        help="Comma-separated eta values for wall profiles "
             "(default: 0.367,0.393,0.449,0.492)")
    p.add_argument(
        "--nz", type=int, default=2048,
        help="1D grid points for wall solver (default: 2048)")
    p.add_argument(
        "--Lz", type=float, default=8.0,
        help="Wall solver domain size in sigma (default: 8.0)")
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# APPROACH SETUP
# ═══════════════════════════════════════════════════════════════════════

def create_nonlocal_functional(grid, seed=42, ckpt_path=None):
    """Create (or load) the NonlocalLutskoFunctional."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    kernels = FMTKernels(grid, R=0.5)
    calculator = WeightedDensityCalculator(kernels)
    network = NonlocalConditionalNetwork(k1)
    kernel = LearnableKernel(k2)
    func = NonlocalLutskoFunctional(network, kernel, calculator, grid)

    if ckpt_path is not None:
        print(f"  Loading nonlocal checkpoint: {ckpt_path}")
        func = eqx.tree_deserialise_leaves(ckpt_path, func)

    return func


def create_local_network(seed=42, ckpt_path=None):
    """Create (or load) the local ConditionalNetwork. Returns None if no ckpt."""
    if ckpt_path is None:
        return None

    key = jax.random.PRNGKey(seed)
    net = ConditionalNetwork(key, n_features=5, hidden_dim=32, n_hidden=2,
                             A_bounds=(0.8, 1.5), B_bounds=(-1.5, 0.0))
    print(f"  Loading local checkpoint: {ckpt_path}")
    net = eqx.tree_deserialise_leaves(ckpt_path, net)
    return net


def get_AB(name, eta, nonlocal_func=None, local_net=None):
    """Return (A, B) for a given approach at packing fraction eta.

    Returns None if the approach is unavailable (e.g., no checkpoint).
    """
    if name == "Nonlocal NN":
        if nonlocal_func is None:
            return None
        A, B = nonlocal_func.bulk_parameters(eta)
        return float(A), float(B)
    elif name == "Local NN":
        if local_net is None:
            return None
        A, B = local_net.from_eta(eta)
        return float(A), float(B)
    elif name in FIXED_PARAMS:
        return FIXED_PARAMS[name]
    else:
        raise ValueError(f"Unknown approach: {name}")


def get_functional_for_solver(name, A, B):
    """Return a 1D functional object for the WallSolver.

    White Bear II and Rosenfeld use their own dedicated classes
    (which include phi2/phi3 corrections or the original Rosenfeld form).
    All others use esFMT_Tensor with the given (A, B).
    """
    if name == "White Bear II":
        return WhiteBearIIFMT()
    elif name == "Rosenfeld":
        return RosenfeldFMT()
    else:
        return esFMT_Tensor(A=A, B=B)


# ═══════════════════════════════════════════════════════════════════════
# PANEL 1: BULK THERMODYNAMICS  Z(eta)
# ═══════════════════════════════════════════════════════════════════════

def compute_bulk_data(approaches, nonlocal_func, local_net):
    """Compute Z(eta) for all approaches. Returns dict name -> Z_array."""
    etas = np.linspace(0.01, 0.52, 80)
    Z_cs = np.array([float(BT.Z_CS(e)) for e in etas])

    data = {"etas": etas, "Z_cs": Z_cs}

    for name in approaches:
        Z_vals = []
        skip = False
        for e in etas:
            ab = get_AB(name, e, nonlocal_func, local_net)
            if ab is None:
                skip = True
                break
            A, B = ab
            Z_vals.append(float(BT.Z_lutsko(e, A, B)))
        if skip:
            data[name] = None
        else:
            data[name] = np.array(Z_vals)
    return data


def plot_bulk_Z(ax, bulk_data, approaches):
    """Plot Z(eta) on the given axes."""
    etas = bulk_data["etas"]
    ax.plot(etas, bulk_data["Z_cs"], "k-", lw=2.5, label="CS (exact)")

    for name in approaches:
        Z = bulk_data.get(name)
        if Z is None:
            continue
        sty = STYLES[name]
        ax.plot(etas, Z, color=sty["color"], ls=sty["ls"],
                lw=sty["lw"], label=name)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$Z = \beta P / \rho$")
    ax.set_title("(a) Compressibility Factor")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 0.52)
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════
# PANEL 2: WALL PROFILES
# ═══════════════════════════════════════════════════════════════════════

def solve_wall_profile(name, eta, solver, nonlocal_func, local_net):
    """Solve a single wall profile. Returns result dict or None."""
    ab = get_AB(name, eta, nonlocal_func, local_net)
    if ab is None:
        return None
    A, B = ab
    functional = get_functional_for_solver(name, A, B)
    try:
        return solver.solve(eta, functional,
                            max_iter=6000, tol=1e-8, verbose=False)
    except Exception as exc:
        print(f"    WARNING: {name} eta={eta:.3f} solver failed: {exc}")
        return None


def plot_wall_profiles_panel(ax, eta, approaches, solver,
                             nonlocal_func, local_net):
    """Plot wall profiles for a single eta on the given axes."""
    # MD reference
    mc = get_mc_data(eta)
    if mc is not None:
        z_mc = np.array(mc["z"])
        rho_mc = np.array(mc["rho"]) / mc["rho_bulk"]
        ax.plot(z_mc, rho_mc, "ko", ms=1.5, alpha=0.4, label="MD data",
                zorder=10)

    # CS contact line
    Z_cs = float(BT.Z_CS(eta))
    ax.axhline(Z_cs, color="gray", ls=":", lw=0.8, alpha=0.5)

    for name in approaches:
        res = solve_wall_profile(name, eta, solver, nonlocal_func, local_net)
        if res is None:
            continue
        sty = STYLES[name]
        z = np.array(res["z"])
        rho_norm = np.array(res["rho_norm"])
        ax.plot(z, rho_norm, color=sty["color"], ls=sty["ls"],
                lw=sty["lw"], label=name)

    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$z / \sigma$")
    ax.set_ylabel(r"$\rho(z) / \rho_b$")
    ax.set_title(rf"$\eta = {eta}$")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════
# PANEL 3: CONTACT DENSITIES
# ═══════════════════════════════════════════════════════════════════════

def compute_contact_data(approaches, solver, nonlocal_func, local_net,
                         n_points=20):
    """Compute contact density vs eta for all approaches."""
    eta_range = np.linspace(0.10, 0.50, n_points)
    Z_cs = np.array([float(BT.Z_CS(e)) for e in eta_range])

    data = {"etas": eta_range, "Z_cs": Z_cs}

    for name in approaches:
        contacts = []
        skip = False
        for eta in eta_range:
            res = solve_wall_profile(name, eta, solver,
                                     nonlocal_func, local_net)
            if res is None:
                skip = True
                break
            contacts.append(float(res["contact"]))

        if skip:
            data[name] = None
        else:
            data[name] = np.array(contacts)

    return data


def plot_contact_densities(ax, contact_data, approaches):
    """Plot contact density vs eta on the given axes."""
    etas = contact_data["etas"]
    ax.plot(etas, contact_data["Z_cs"], "k-", lw=2.5, label="CS (exact)",
            zorder=5)

    # MD reference points
    md_etas = sorted(MD_CONTACTS.keys())
    md_vals = [MD_CONTACTS[e] for e in md_etas]
    ax.plot(md_etas, md_vals, "ks", ms=7, mfc="none", mew=1.8,
            label="MD data", zorder=10)

    for name in approaches:
        vals = contact_data.get(name)
        if vals is None:
            continue
        sty = STYLES[name]
        ax.plot(etas, vals, color=sty["color"], ls=sty["ls"],
                lw=sty["lw"], label=name)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\rho(R^+) / \rho_b$")
    ax.set_title("(c) Contact Density")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0.10, 0.50)
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════
# PANEL 4: CONSTRAINT PARAMETER C(eta) = 8A + 2B - 9
# ═══════════════════════════════════════════════════════════════════════

def compute_constraint_data(approaches, nonlocal_func, local_net):
    """Compute C(eta) for all approaches."""
    etas = np.linspace(0.01, 0.52, 80)
    data = {"etas": etas}

    for name in approaches:
        C_vals = []
        skip = False
        for e in etas:
            ab = get_AB(name, e, nonlocal_func, local_net)
            if ab is None:
                skip = True
                break
            A, B = ab
            C_vals.append(8 * A + 2 * B - 9)

        if skip:
            data[name] = None
        else:
            data[name] = np.array(C_vals)

    return data


def plot_constraint_parameter(ax, constraint_data, approaches):
    """Plot C(eta) on the given axes."""
    etas = constraint_data["etas"]

    # Reference lines
    ax.axhline(-3.0, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(0.0,  color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(-0.6, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.text(0.52, -3.0, " CS", fontsize=7, va="center", color="gray")
    ax.text(0.52, 0.0,  " PY", fontsize=7, va="center", color="gray")
    ax.text(0.52, -0.6, r" G\"ul", fontsize=7, va="center", color="gray")

    for name in approaches:
        C = constraint_data.get(name)
        if C is None:
            continue
        sty = STYLES[name]
        ax.plot(etas, C, color=sty["color"], ls=sty["ls"],
                lw=sty["lw"], label=name)

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$C(\eta) = 8A + 2B - 9$")
    ax.set_title(r"(d) Constraint Parameter")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlim(0, 0.52)
    ax.set_ylim(-4.5, 5.0)
    ax.grid(True, alpha=0.2)


# ═══════════════════════════════════════════════════════════════════════
# COMBINED 2x2 FIGURE
# ═══════════════════════════════════════════════════════════════════════

def figure_combined(approaches, bulk_data, contact_data, constraint_data,
                    solver, nonlocal_func, local_net, eta_wall, outdir):
    """Generate the main 2x2 comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Bulk Z(eta)
    plot_bulk_Z(axes[0, 0], bulk_data, approaches)

    # (b) Wall profiles at the specified eta (default 0.449)
    plot_wall_profiles_panel(axes[0, 1], eta_wall, approaches, solver,
                             nonlocal_func, local_net)
    axes[0, 1].set_title(rf"(b) Wall Profile, $\eta = {eta_wall}$")

    # (c) Contact densities
    plot_contact_densities(axes[1, 0], contact_data, approaches)

    # (d) Constraint parameter
    plot_constraint_parameter(axes[1, 1], constraint_data, approaches)

    fig.suptitle("Nonlocal Neural Functional vs Classical Approaches",
                 fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    figpath = outdir / "comparison_2x2.png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"  Saved: {figpath}")


# ═══════════════════════════════════════════════════════════════════════
# INDIVIDUAL DETAILED PLOTS
# ═══════════════════════════════════════════════════════════════════════

def figure_bulk_detailed(approaches, bulk_data, outdir,
                         nonlocal_func, local_net):
    """3-panel figure: Z, mu_ex, chi_T vs CS."""
    print("\n  Detailed Figure: Bulk Thermodynamics (3 panels)")
    etas = bulk_data["etas"]

    Z_cs = bulk_data["Z_cs"]
    mu_cs = np.array([float(BT.mu_ex_CS(e)) for e in etas])
    chi_cs = np.array([float(BT.chi_T_CS(e)) for e in etas])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax in axes:
        ax.grid(True, alpha=0.2)

    axes[0].plot(etas, Z_cs, "k-", lw=2.5, label="CS (exact)")
    axes[1].plot(etas, mu_cs, "k-", lw=2.5, label="CS (exact)")
    axes[2].plot(etas, chi_cs, "k-", lw=2.5, label="CS (exact)")

    for name in approaches:
        sty = STYLES[name]
        Z_vals, mu_vals, chi_vals = [], [], []
        skip = False
        for e in etas:
            ab = get_AB(name, e, nonlocal_func, local_net)
            if ab is None:
                skip = True
                break
            A, B = ab
            Z_vals.append(float(BT.Z_lutsko(e, A, B)))
            mu_vals.append(float(BT.mu_ex_bulk_lutsko(e, A, B)))
            chi_vals.append(float(BT.chi_T_bulk_lutsko(e, A, B)))
        if skip:
            print(f"    {name}: skipped (no checkpoint)")
            continue

        Z_vals = np.array(Z_vals)
        mu_vals = np.array(mu_vals)
        chi_vals = np.array(chi_vals)

        axes[0].plot(etas, Z_vals, color=sty["color"], ls=sty["ls"],
                     lw=sty["lw"], label=name)
        axes[1].plot(etas, mu_vals, color=sty["color"], ls=sty["ls"],
                     lw=sty["lw"], label=name)
        axes[2].plot(etas, chi_vals, color=sty["color"], ls=sty["ls"],
                     lw=sty["lw"], label=name)

        Z_err = np.max(np.abs(Z_vals / Z_cs - 1)) * 100
        mu_err = np.max(np.abs(mu_vals / (mu_cs + 1e-10) - 1)) * 100
        print(f"    {name:20s}  Z_err={Z_err:5.2f}%  mu_err={mu_err:5.2f}%")

    axes[0].set(xlabel=r"$\eta$", ylabel=r"$Z = \beta P / \rho$",
                title="(a) Compressibility Factor", xlim=(0, 0.52))
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel=r"$\eta$", ylabel=r"$\beta\mu_{\mathrm{ex}}$",
                title="(b) Excess Chemical Potential", xlim=(0, 0.52))
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel=r"$\eta$",
                ylabel=r"$\chi_T / \chi_T^{\mathrm{id}}$",
                title="(c) Isothermal Compressibility",
                xlim=(0, 0.52), ylim=(0, 1.0))
    axes[2].legend(fontsize=8)

    fig.suptitle("Bulk Thermodynamics: All Approaches vs Carnahan-Starling",
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figpath = outdir / "detail_bulk_thermodynamics.png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"  Saved: {figpath}")


def figure_wall_profiles_multi(approaches, eta_targets, solver, outdir,
                               nonlocal_func, local_net):
    """Multi-panel wall profiles at all requested eta values.

    Returns dict of {name: {eta: contact}} for summary table.
    """
    print("\n  Detailed Figure: Wall Density Profiles")
    n_eta = len(eta_targets)
    ncols = min(n_eta, 2)
    nrows = (n_eta + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    all_contacts = {name: {} for name in approaches}

    for i, eta in enumerate(eta_targets):
        ax = axes_flat[i]

        # MD reference
        mc = get_mc_data(eta)
        if mc is not None:
            z_mc = np.array(mc["z"])
            rho_mc = np.array(mc["rho"]) / mc["rho_bulk"]
            ax.plot(z_mc, rho_mc, "ko", ms=2, alpha=0.4, label="MD data",
                    zorder=10)

        Z_cs = float(BT.Z_CS(eta))
        ax.axhline(Z_cs, color="gray", ls=":", lw=0.8, alpha=0.5)

        for name in approaches:
            res = solve_wall_profile(name, eta, solver,
                                     nonlocal_func, local_net)
            if res is None:
                continue
            sty = STYLES[name]
            ax.plot(np.array(res["z"]), np.array(res["rho_norm"]),
                    color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                    label=name)
            contact = float(res["contact"])
            all_contacts[name][eta] = contact
            conv = "Y" if res["converged"] else "N"
            print(f"    eta={eta:.3f}  {name:20s}  "
                  f"contact={contact:7.3f}  conv={conv}")

        ax.set_xlim(0, 5)
        ax.set_xlabel(r"$z / \sigma$")
        ax.set_ylabel(r"$\rho(z) / \rho_b$")
        ax.set_title(rf"$\eta = {eta}$")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Wall Density Profiles: All Approaches", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    figpath = outdir / "detail_wall_profiles.png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"  Saved: {figpath}")

    return all_contacts


def figure_contact_detailed(approaches, contact_data, outdir):
    """Standalone contact density figure."""
    print("\n  Detailed Figure: Contact Densities")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_contact_densities(ax, contact_data, approaches)
    ax.set_title("Contact Density vs Packing Fraction")
    plt.tight_layout()
    figpath = outdir / "detail_contact_densities.png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"  Saved: {figpath}")


def figure_constraint_detailed(approaches, constraint_data, outdir):
    """Standalone constraint parameter figure."""
    print("\n  Detailed Figure: Constraint Parameter")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    plot_constraint_parameter(ax, constraint_data, approaches)
    ax.set_title(r"Constraint Parameter $C(\eta) = 8A + 2B - 9$")
    plt.tight_layout()
    figpath = outdir / "detail_constraint_parameter.png"
    plt.savefig(figpath, dpi=300)
    plt.close()
    print(f"  Saved: {figpath}")


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════

def print_summary_table(approaches, all_contacts, outdir,
                        nonlocal_func, local_net):
    """Print and save comparison table with contact density errors."""
    eta_targets = sorted(MD_CONTACTS.keys())

    print("\n" + "=" * 80)
    print("  SUMMARY TABLE: Contact Densities vs MD")
    print("=" * 80)

    # Header
    hdr = f"  {'Approach':20s}"
    for eta in eta_targets:
        hdr += f"  eta={eta:.3f}"
    hdr += "   MAE%"
    print(hdr)
    print("  " + "-" * 76)

    # MD reference
    row = f"  {'MD (reference)':20s}"
    for eta in eta_targets:
        row += f"  {MD_CONTACTS[eta]:>9.2f}"
    print(row)

    # CS exact
    row = f"  {'CS exact':20s}"
    cs_errs = []
    for eta in eta_targets:
        Z_cs = float(BT.Z_CS(eta))
        err = abs(100 * (Z_cs - MD_CONTACTS[eta]) / MD_CONTACTS[eta])
        cs_errs.append(err)
        row += f"  {Z_cs:8.2f} "
    row += f"  {np.mean(cs_errs):5.2f}%"
    print(row)
    print("  " + "-" * 76)

    lines = []
    for name in approaches:
        row = f"  {name:20s}"
        errs = []
        for eta in eta_targets:
            contact = all_contacts.get(name, {}).get(eta)
            if contact is not None:
                err = abs(100 * (contact - MD_CONTACTS[eta]) / MD_CONTACTS[eta])
                errs.append(err)
                row += f"  {contact:8.2f} "
            else:
                row += f"  {'N/A':>9s}"

        if errs:
            row += f"  {np.mean(errs):5.2f}%"
        else:
            row += f"  {'N/A':>6s}"
        print(row)

        # Print C values below
        detail = f"  {'':20s}"
        for eta in eta_targets:
            ab = get_AB(name, eta, nonlocal_func, local_net)
            if ab is not None:
                A, B = ab
                C = 8 * A + 2 * B - 9
                detail += f"  C={C:+5.2f}  "
            else:
                detail += f"  {'':>9s}"
        print(detail)
        lines.append(row)

    print("=" * 80)

    # Save to file
    tbl_path = outdir / "summary_table.txt"
    with open(tbl_path, "w") as f:
        f.write(hdr + "\n")
        f.write("  " + "-" * 76 + "\n")
        for line in lines:
            f.write(line + "\n")
    print(f"  Saved: {tbl_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    eta_targets = [float(e.strip()) for e in args.eta_values.split(",")]
    # Use the highest eta for the combined figure wall panel
    eta_wall = max(eta_targets)

    print("=" * 68)
    print("  APPROACH COMPARISON: Nonlocal Neural vs Classical FMT")
    print("=" * 68)
    print(f"  Output dir:       {outdir}")
    print(f"  Nonlocal ckpt:    {args.checkpoint or '(fresh/untrained)'}")
    print(f"  Local ckpt:       {args.local_checkpoint or '(none — skipped)'}")
    print(f"  Nonlocal grid:    {args.grid_size}^3, L={args.box_length}")
    print(f"  Wall solver:      nz={args.nz}, Lz={args.Lz}")
    print(f"  eta values:       {eta_targets}")

    # ── Build approaches ─────────────────────────────────────────────
    print("\n  Setting up approaches...")
    t0 = time.time()

    grid = Grid((args.grid_size,) * 3, args.box_length)
    nonlocal_func = create_nonlocal_functional(
        grid, args.seed, args.checkpoint)
    local_net = create_local_network(args.seed, args.local_checkpoint)

    solver = WallSolver(nz=args.nz, Lz=args.Lz, R=0.5)

    approaches = ["Nonlocal NN", "White Bear II", "Rosenfeld",
                   "Gul optimal"]
    if local_net is not None:
        approaches.insert(1, "Local NN")
    else:
        print("  (Local NN skipped — no --local-checkpoint provided)")

    setup_time = time.time() - t0
    print(f"  Setup complete ({setup_time:.1f}s)")

    # ── Compute data ─────────────────────────────────────────────────
    print("\n  Computing bulk thermodynamics...")
    t1 = time.time()
    bulk_data = compute_bulk_data(approaches, nonlocal_func, local_net)
    print(f"  Bulk data computed ({time.time() - t1:.1f}s)")

    print("\n  Computing constraint parameters...")
    constraint_data = compute_constraint_data(
        approaches, nonlocal_func, local_net)

    print("\n  Computing contact densities (this may take a while)...")
    t2 = time.time()
    contact_data = compute_contact_data(
        approaches, solver, nonlocal_func, local_net, n_points=20)
    print(f"  Contact data computed ({time.time() - t2:.1f}s)")

    # ── Combined 2x2 figure ──────────────────────────────────────────
    print("\n  Generating combined 2x2 figure...")
    t3 = time.time()
    figure_combined(approaches, bulk_data, contact_data, constraint_data,
                    solver, nonlocal_func, local_net, eta_wall, outdir)
    print(f"  Combined figure done ({time.time() - t3:.1f}s)")

    # ── Individual detailed figures ──────────────────────────────────
    figure_bulk_detailed(approaches, bulk_data, outdir,
                         nonlocal_func, local_net)

    t4 = time.time()
    all_contacts = figure_wall_profiles_multi(
        approaches, eta_targets, solver, outdir, nonlocal_func, local_net)
    print(f"  Wall profiles done ({time.time() - t4:.1f}s)")

    figure_contact_detailed(approaches, contact_data, outdir)
    figure_constraint_detailed(approaches, constraint_data, outdir)

    # ── Summary table ────────────────────────────────────────────────
    print_summary_table(approaches, all_contacts, outdir,
                        nonlocal_func, local_net)

    # ── Done ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  All outputs saved to: {outdir}/")
    print("=" * 68)


if __name__ == "__main__":
    main()
