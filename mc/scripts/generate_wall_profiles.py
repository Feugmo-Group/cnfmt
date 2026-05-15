"""
Generate GCMC Hard-Sphere Wall Density Profiles
================================================

Runs Grand Canonical Monte Carlo simulations of hard spheres at a planar
hard wall for one or more bulk packing fractions η, saves the resulting
ρ(z)/ρ_bulk profiles, and validates against the contact-density sum rule.

Usage
-----
Single η value:
    python -m mc.scripts.generate_wall_profiles --eta 0.367 \\
           --output-dir outputs/mc_walls

Multiple η values (comma-separated):
    python -m mc.scripts.generate_wall_profiles \\
           --eta-values 0.367,0.393,0.449,0.492 \\
           --output-dir outputs/mc_walls

Full options:
    python -m mc.scripts.generate_wall_profiles --help

Physics
-------
For hard spheres at a planar hard wall:
  - Contact density sum rule (exact):  ρ(σ/2) = βP = ρ_bulk · Z_CS(η)
  - Chemical potential (CS EOS):       βμ_ex = η(8 - 9η + 3η²)/(1-η)³
  - Activity:                          z_act = ρ_bulk · exp(βμ_ex)

Output
------
For each η, saves:
  outputs/mc_walls/wall_eta{eta:.3f}.npz  — z, rho, rho_bulk arrays
  outputs/mc_walls/wall_eta{eta:.3f}.png  — plot of ρ(z)/ρ_bulk vs z/σ

The .npz format is compatible with ``solvers/wall_profile.py::get_mc_data()``.

References
----------
Frenkel & Smit (2002) Ch. 5
Davidchack, Laird, Roth, Cond. Matt. Phys. (2016) — reference MC data
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Add repo root to path so mc.* imports work ────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mc.ensembles.gcmc import GCMCSampler
from mc.observables.density_profile import DensityProfileAccumulator


# ── Physics helpers ────────────────────────────────────────────────────


def mu_ex_CS(eta: float) -> float:
    """Carnahan-Starling excess chemical potential βμ_ex(η).

    βμ_ex = η(8 - 9η + 3η²) / (1 - η)³
    """
    return eta * (8.0 - 9.0 * eta + 3.0 * eta ** 2) / (1.0 - eta) ** 3


def Z_CS(eta: float) -> float:
    """Carnahan-Starling compressibility factor Z(η).

    Z = (1 + η + η² - η³) / (1 - η)³
    """
    return (1.0 + eta + eta ** 2 - eta ** 3) / (1.0 - eta) ** 3


def rho_from_eta(eta: float, sigma: float = 1.0) -> float:
    """Number density from packing fraction: ρ = 6η/(π σ³)."""
    return 6.0 * eta / (math.pi * sigma ** 3)


# ── Plotting (matplotlib optional) ────────────────────────────────────


def _plot_profile(
    z: np.ndarray,
    rho: np.ndarray,
    rho_bulk: float,
    eta: float,
    output_path: Path,
    rho_contact_mc: float,
    rho_contact_exact: float,
) -> None:
    """Save a plot of ρ(z)/ρ_bulk vs z/σ."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping plot.")
        return

    g_z = rho / max(rho_bulk, 1e-12)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z, g_z, "b-", linewidth=1.5, label=f"GCMC  η={eta:.3f}")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="bulk")
    ax.axvline(0.5, color="k", linestyle=":", linewidth=0.8, label="z = σ/2")

    # Mark contact density
    ax.scatter(
        [0.5], [rho_contact_mc / max(rho_bulk, 1e-12)],
        color="red", zorder=5, s=60,
        label=f"MC contact: {rho_contact_mc/max(rho_bulk,1e-12):.3f}",
    )
    ax.axhline(
        rho_contact_exact / max(rho_bulk, 1e-12),
        color="orange", linestyle="-.", linewidth=1.0,
        label=f"Z_CS (exact): {rho_contact_exact/max(rho_bulk,1e-12):.3f}",
    )

    ax.set_xlabel("z / σ")
    ax.set_ylabel("ρ(z) / ρ_bulk")
    ax.set_title(f"Hard-sphere wall profile  η = {eta:.3f}")
    ax.legend(fontsize=8)
    ax.set_xlim(0.0, min(8.0, z[-1]))
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved to {output_path}")


# ── Single η run ───────────────────────────────────────────────────────


def run_single_eta(
    eta: float,
    output_dir: Path,
    Lx: float = 6.0,
    Ly: Optional[float] = None,
    Lz: float = 10.0,
    n_equil: int = 100_000,
    n_prod: int = 500_000,
    n_sample: int = 100,
    max_disp: float = 0.15,
    n_bins: int = 200,
    sigma: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run GCMC for a single packing fraction and save results.

    Parameters
    ----------
    eta : float
        Target bulk packing fraction η ∈ (0, 0.52).
    output_dir : Path
        Directory where .npz and .png are saved.
    Lx, Ly : float
        Box dimensions in x, y (periodic).  Ly defaults to Lx.
    Lz : float
        Box depth in z (wall-normal).
    n_equil : int
        Equilibration steps.
    n_prod : int
        Production steps.
    n_sample : int
        Sample density profile every this many production steps.
    max_disp : float
        Initial max displacement (adjusted during equilibration).
    n_bins : int
        Number of histogram bins for ρ(z).
    sigma : float
        Hard-sphere diameter.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys: 'z', 'rho', 'rho_bulk', 'eta', 'contact_mc',
                    'contact_exact', 'contact_error_pct'
    """
    if Ly is None:
        Ly = Lx

    rho_bulk = rho_from_eta(eta, sigma)
    mu_ex = mu_ex_CS(eta)
    Z = Z_CS(eta)

    print(f"\n{'='*60}")
    print(f"GCMC  η = {eta:.4f}  ρ_bulk = {rho_bulk:.4f} σ⁻³")
    print(f"  βμ_ex = {mu_ex:.4f}   Z_CS = {Z:.4f}")
    print(f"  Box: {Lx:.1f} × {Ly:.1f} × {Lz:.1f} σ")
    print(f"  Equilibration: {n_equil:,}  Production: {n_prod:,}  Sample: every {n_sample}")
    print(f"{'='*60}")

    t0 = time.time()

    # ── Initialise GCMC ───────────────────────────────────────────────
    sampler = GCMCSampler(
        Lx=Lx * sigma,
        Ly=Ly * sigma,
        Lz=Lz * sigma,
        mu_ex=mu_ex,
        rho_bulk=rho_bulk,
        sigma=sigma,
        seed=seed,
    )

    # ── Profile accumulator ───────────────────────────────────────────
    acc = DensityProfileAccumulator(
        n_bins=n_bins,
        Lz=Lz * sigma,
        Lx=Lx * sigma,
        Ly=Ly * sigma,
        sigma=sigma,
    )

    # ── Equilibration ─────────────────────────────────────────────────
    print("Equilibrating...")
    sampler._reset_counters()
    disp = float(max_disp)

    for step in range(n_equil):
        move = sampler.rng.integers(0, 3)
        if move == 0:
            sampler._try_displace(disp)
        elif move == 1:
            sampler._try_insert()
        else:
            sampler._try_delete()

        # Adjust displacement every 500 steps
        if sampler.n_disp_att > 0 and (step + 1) % 500 == 0:
            rate = sampler.n_disp_acc / sampler.n_disp_att
            if rate > 0.40:
                disp *= 1.05
            elif rate < 0.30:
                disp *= 0.95
            disp = float(np.clip(disp, 0.01, min(Lx, Ly, Lz) / 2.0))
            sampler.n_disp_acc = 0
            sampler.n_disp_att = 0

        if verbose and (step + 1) % 20000 == 0:
            rho_now = sampler.N / sampler.V
            acc_d = sampler.n_disp_acc / max(sampler.n_disp_att, 1)
            acc_i = sampler.n_ins_acc / max(sampler.n_ins_att, 1)
            acc_del = sampler.n_del_acc / max(sampler.n_del_att, 1)
            print(
                f"  equil {step+1:>8,}/{n_equil:<8,}  "
                f"N={sampler.N:4d}  ρ={rho_now:.4f}  "
                f"acc[d/i/r]={acc_d:.2f}/{acc_i:.2f}/{acc_del:.2f}  "
                f"disp={disp:.4f}"
            )

    sampler._reset_counters()

    # ── Production ────────────────────────────────────────────────────
    print("Production run...")
    n_sampled = 0

    for step in range(n_prod):
        move = sampler.rng.integers(0, 3)
        if move == 0:
            sampler._try_displace(disp)
        elif move == 1:
            sampler._try_insert()
        else:
            sampler._try_delete()

        # Sample density profile
        if (step + 1) % n_sample == 0:
            if sampler.N > 0:
                acc.update(sampler.positions)
            else:
                acc.update(np.empty((0, 3)))
            n_sampled += 1

        if verbose and (step + 1) % 50000 == 0:
            rho_now = sampler.N / sampler.V
            acc_d = sampler.n_disp_acc / max(sampler.n_disp_att, 1)
            acc_i = sampler.n_ins_acc / max(sampler.n_ins_att, 1)
            acc_del = sampler.n_del_acc / max(sampler.n_del_att, 1)
            print(
                f"  prod  {step+1:>8,}/{n_prod:<8,}  "
                f"N={sampler.N:4d}  ρ={rho_now:.4f}  "
                f"acc[d/i/r]={acc_d:.2f}/{acc_i:.2f}/{acc_del:.2f}  "
                f"frames={n_sampled}"
            )

    t1 = time.time()
    elapsed = t1 - t0

    # ── Extract profile ───────────────────────────────────────────────
    z, rho, rho_bulk_measured = acc.get()

    # Contact density: first bin at z ≈ σ/2
    rho_contact_mc, z_contact = acc.contact_density(sigma=sigma)
    rho_contact_exact = rho_bulk * Z   # sum rule: ρ_contact = βP = ρ·Z

    error_pct = 100.0 * abs(rho_contact_mc - rho_contact_exact) / max(rho_contact_exact, 1e-12)

    # Mean N from production
    rho_mean = rho_bulk_measured if rho_bulk_measured > 0 else rho_bulk

    print(f"\n--- Results for η = {eta:.4f} ---")
    print(f"  Elapsed time         : {elapsed:.1f} s")
    print(f"  Frames accumulated   : {n_sampled}")
    print(f"  ρ_bulk (target)      : {rho_bulk:.4f} σ⁻³")
    print(f"  ρ_bulk (measured)    : {rho_bulk_measured:.4f} σ⁻³")
    print(f"  Contact density MC   : {rho_contact_mc:.4f} σ⁻³  at z={z_contact:.3f} σ")
    print(f"  Contact density exact: {rho_contact_exact:.4f} σ⁻³  (= ρ·Z_CS)")
    print(f"  Error vs sum rule    : {error_pct:.2f} %")

    # ── Save .npz ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"wall_eta{eta:.3f}.npz"
    np.savez(
        str(npz_path),
        z=z,
        rho=rho,
        rho_bulk=np.array(rho_bulk),
        rho_bulk_measured=np.array(rho_bulk_measured),
        eta=np.array(eta),
        contact_mc=np.array(rho_contact_mc),
        contact_exact=np.array(rho_contact_exact),
        Lx=np.array(Lx),
        Ly=np.array(Ly),
        Lz=np.array(Lz),
        n_bins=np.array(n_bins),
        n_equil=np.array(n_equil),
        n_prod=np.array(n_prod),
        n_sample=np.array(n_sample),
        n_frames=np.array(n_sampled),
    )
    print(f"  Saved → {npz_path}")

    # ── Save .png ─────────────────────────────────────────────────────
    png_path = output_dir / f"wall_eta{eta:.3f}.png"
    _plot_profile(
        z=z,
        rho=rho,
        rho_bulk=rho_bulk,
        eta=eta,
        output_path=png_path,
        rho_contact_mc=rho_contact_mc,
        rho_contact_exact=rho_contact_exact,
    )

    return {
        "z": z,
        "rho": rho,
        "rho_bulk": float(rho_bulk),
        "eta": float(eta),
        "contact_mc": float(rho_contact_mc),
        "contact_exact": float(rho_contact_exact),
        "contact_error_pct": float(error_pct),
        "n_frames": n_sampled,
        "elapsed_s": elapsed,
    }


# ── Summary report ─────────────────────────────────────────────────────


def _print_summary(results: List[dict]) -> None:
    """Print a summary table of contact-density validation."""
    print("\n" + "=" * 65)
    print("Contact Density Sum Rule Validation")
    print("  ρ_contact = βP = ρ_bulk · Z_CS(η)  (exact for hard wall)")
    print("=" * 65)
    print(f"{'η':>6}  {'ρ_bulk':>8}  {'ρ_contact(MC)':>14}  {'ρ_contact(exact)':>16}  {'error%':>7}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['eta']:6.3f}  "
            f"{r['rho_bulk']:8.4f}  "
            f"{r['contact_mc']:14.4f}  "
            f"{r['contact_exact']:16.4f}  "
            f"{r['contact_error_pct']:7.2f}"
        )
    print("=" * 65)

    # Overall
    errors = [r["contact_error_pct"] for r in results]
    print(f"Mean |error| = {np.mean(errors):.2f} %   Max = {np.max(errors):.2f} %")


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate GCMC hard-sphere wall density profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # η specification (mutually exclusive)
    eta_group = p.add_mutually_exclusive_group(required=True)
    eta_group.add_argument(
        "--eta",
        type=float,
        default=None,
        metavar="ETA",
        help="Single bulk packing fraction (e.g. 0.367).",
    )
    eta_group.add_argument(
        "--eta-values",
        type=str,
        default=None,
        metavar="0.367,0.393,...",
        help="Comma-separated list of packing fractions.",
    )

    # Box geometry
    p.add_argument("--Lx", type=float, default=6.0, help="Box length in x (σ).")
    p.add_argument("--Ly", type=float, default=None, help="Box length in y (σ). Defaults to Lx.")
    p.add_argument("--Lz", type=float, default=10.0, help="Box depth in z/wall-normal (σ).")

    # Simulation parameters
    p.add_argument("--n-equil", type=int, default=100_000, help="Equilibration steps.")
    p.add_argument("--n-prod", type=int, default=500_000, help="Production steps.")
    p.add_argument("--n-sample", type=int, default=100, help="Sample profile every N steps.")
    p.add_argument("--max-disp", type=float, default=0.15, help="Initial max displacement (σ).")
    p.add_argument("--n-bins", type=int, default=200, help="Number of histogram bins.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mc_walls",
        help="Directory for output .npz and .png files.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress verbose output.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Parse η values
    if args.eta is not None:
        eta_list: List[float] = [args.eta]
    else:
        eta_list = [float(x.strip()) for x in args.eta_values.split(",")]

    # Validate
    for eta in eta_list:
        if not (0.0 < eta < 0.52):
            print(f"ERROR: η = {eta} is outside the physical range (0, 0.52).")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    Ly = args.Ly if args.Ly is not None else args.Lx
    verbose = not args.quiet

    print(f"GCMC Wall Profile Generator")
    print(f"  η values  : {eta_list}")
    print(f"  Box       : {args.Lx:.1f} × {Ly:.1f} × {args.Lz:.1f} σ")
    print(f"  n_equil   : {args.n_equil:,}")
    print(f"  n_prod    : {args.n_prod:,}")
    print(f"  n_sample  : {args.n_sample}")
    print(f"  n_bins    : {args.n_bins}")
    print(f"  seed      : {args.seed}")
    print(f"  output    : {output_dir}")

    results = []
    for i, eta in enumerate(eta_list):
        print(f"\n[{i+1}/{len(eta_list)}] Running η = {eta:.4f} ...")
        result = run_single_eta(
            eta=eta,
            output_dir=output_dir,
            Lx=args.Lx,
            Ly=Ly,
            Lz=args.Lz,
            n_equil=args.n_equil,
            n_prod=args.n_prod,
            n_sample=args.n_sample,
            max_disp=args.max_disp,
            n_bins=args.n_bins,
            seed=args.seed,
            verbose=verbose,
        )
        results.append(result)

    _print_summary(results)
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
