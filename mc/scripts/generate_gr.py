"""
Generate g(r) at Multiple Packing Fractions
=============================================

Run NVT Monte Carlo for hard spheres at one or more packing fractions,
accumulating g(r) and Widom μ_ex.  Results are saved as NumPy .npz files
and compared to Carnahan-Starling.

Usage
-----
Single η:
    python -m mc.scripts.generate_gr --eta 0.3 --n-particles 500 \\
        --output-dir outputs/mc

Multiple η values (comma-separated):
    python -m mc.scripts.generate_gr \\
        --eta-values 0.1,0.2,0.3,0.367,0.393,0.449 \\
        --output-dir outputs/mc

The virial contact pressure is estimated from the RDF contact value g(σ+):
    Z_virial = 1 + (2π/3) * ρ * σ³ * g(σ+)

Output files
------------
<output_dir>/gr_eta<ETA>.npz  with keys  r, g, r_bins, eta, Z_mc, mu_ex_mc
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Enable double precision
jax.config.update("jax_enable_x64", True)

# Allow running as `python -m mc.scripts.generate_gr` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mc.core.state import MCState
from mc.ensembles.nvt import NVTSampler, NVTState, mc_sweep
from mc.observables.rdf import RDFAccumulator
from mc.observables.widom import WidomAccumulator
from mc.analysis.block_average import block_average, equilibration_check

# Import CS reference from existing codebase
try:
    from core.thermodynamics import BulkThermodynamics
    _HAS_THERMO = True
except ImportError:
    _HAS_THERMO = False


# ── Physics helpers ───────────────────────────────────────────────────


def _box_length_from_eta(N: int, eta: float, sigma: float = 1.0) -> float:
    """L = (N * pi * sigma^3 / (6 * eta))^(1/3)."""
    V = N * np.pi * sigma**3 / (6.0 * eta)
    return float(V ** (1.0 / 3.0))


def _contact_value_from_rdf(
    r: np.ndarray, g: np.ndarray, sigma: float = 1.0
) -> float:
    """Estimate g(σ+) as the first bin above σ."""
    # The first bin centre just above sigma
    mask = r > sigma
    if not np.any(mask):
        return float("nan")
    idx = np.argmax(mask)   # first True
    return float(g[idx])


def _z_virial_from_contact(
    g_contact: float, rho: float, sigma: float = 1.0
) -> float:
    """Z = 1 + (2π/3) * ρ * σ³ * g(σ+)  [virial route for hard spheres]."""
    return 1.0 + (2.0 * np.pi / 3.0) * rho * sigma**3 * g_contact


def _cs_references(eta: float):
    """Return Z_CS and mu_ex_CS for a packing fraction."""
    if _HAS_THERMO:
        Z = float(BulkThermodynamics.Z_CS(eta))
        mu_ex = float(BulkThermodynamics.mu_ex_CS(eta))
    else:
        # Inline CS formulas as fallback
        Z = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
        mu_ex = eta * (8 - 9*eta + 3*eta**2) / (1 - eta)**3
    return Z, mu_ex


# ── Per-η simulation ──────────────────────────────────────────────────


def run_one_eta(
    eta: float,
    n_particles: int,
    n_equil: int,
    n_prod: int,
    n_sample: int,
    n_widom: int,
    seed: int,
    sigma: float,
    rdf_bins: int,
    rdf_rmax: float,
    output_dir: Path,
    verbose: bool = True,
) -> dict:
    """Run NVT MC for one packing fraction and save results.

    Parameters
    ----------
    eta : float
        Target packing fraction.
    n_particles : int
    n_equil : int
        Number of equilibration sweeps.
    n_prod : int
        Number of production sweeps.
    n_sample : int
        Sample g(r) and Widom every n_sample sweeps.
    n_widom : int
        Number of Widom test insertions per sampled frame.
    seed : int
    sigma : float
    rdf_bins : int
    rdf_rmax : float
    output_dir : Path
    verbose : bool

    Returns
    -------
    dict with keys: r, g, Z_mc, Z_cs, mu_ex_mc, mu_ex_cs, eta, acceptance_rate
    """
    N = n_particles
    L = _box_length_from_eta(N, eta, sigma)
    rho = N / L**3

    if verbose:
        print(f"\n{'='*60}")
        print(f"η = {eta:.4f}  N = {N}  L = {L:.4f}  ρ = {rho:.4f}")
        print(f"Equilibration: {n_equil} sweeps  Production: {n_prod} sweeps")
        print(f"{'='*60}")

    # ── Initialise FCC lattice ──
    t0 = time.time()
    state = MCState.from_fcc(N, L)
    if verbose:
        print(f"FCC init: {time.time()-t0:.2f}s")

    # ── Build sampler and equilibrate ──
    sampler = NVTSampler(state, seed=seed, sigma=sigma)

    # Estimate initial max_disp (about sigma/4 is reasonable)
    init_disp = min(0.2 * sigma, L / 4.0)

    t0 = time.time()
    if verbose:
        print(f"Equilibrating ({n_equil} sweeps)...")
    equil_state, _ = _run_sweeps(
        sampler.nvt_state, n_equil, N, init_disp, sigma, verbose=verbose, label="equil"
    )
    sampler.nvt_state = equil_state
    if verbose:
        print(f"Equilibration done in {time.time()-t0:.1f}s")

    # Reset counters after equilibration
    sampler.nvt_state = NVTState(
        mc_state=sampler.nvt_state.mc_state,
        key=sampler.nvt_state.key,
        n_accepted=0,
        n_total=0,
    )

    # ── Production run with observable accumulation ──
    rdf_acc = RDFAccumulator(n_bins=rdf_bins, r_max=rdf_rmax, sigma=sigma)
    widom_acc = WidomAccumulator(n_test=n_widom, sigma=sigma, seed=seed + 1)
    mu_ex_series: list[float] = []

    t0 = time.time()
    if verbose:
        print(f"Production ({n_prod} sweeps, sample every {n_sample})...")

    nvt_state = sampler.nvt_state
    n_sampled = 0

    for sweep_idx in range(n_prod):
        nvt_state, _ = mc_sweep(nvt_state, N, init_disp, sigma)

        if (sweep_idx + 1) % n_sample == 0:
            pos = nvt_state.mc_state.positions
            rdf_acc.update(pos, L)
            mu_val = widom_acc.update(pos, L)
            mu_ex_series.append(mu_val)
            n_sampled += 1

            if verbose and (sweep_idx + 1) % (10 * n_sample) == 0:
                print(
                    f"  sweep {sweep_idx+1}/{n_prod}  "
                    f"sampled={n_sampled}  βμ_ex≈{mu_val:.3f}"
                )

    prod_time = time.time() - t0
    if verbose:
        print(f"Production done in {prod_time:.1f}s  ({prod_time/n_prod*1000:.1f}ms/sweep)")

    # ── Compute observables ──
    r, g = rdf_acc.get()
    g_contact = _contact_value_from_rdf(r, g, sigma)
    Z_mc = _z_virial_from_contact(g_contact, rho, sigma)

    mu_ex_mc, mu_ex_std = widom_acc.get()

    # CS reference
    Z_cs, mu_ex_cs = _cs_references(eta)

    # Equilibration check on μ_ex series
    mu_arr = np.array([x for x in mu_ex_series if np.isfinite(x)])
    converged = equilibration_check(mu_arr) if len(mu_arr) > 4 else True
    mu_mean, mu_se = block_average(mu_arr, max(1, len(mu_arr) // 10)) if len(mu_arr) > 0 else (float("nan"), float("nan"))

    acc_rate = nvt_state.n_accepted / max(nvt_state.n_total, 1)

    if verbose:
        print(f"\n  Results for η = {eta:.4f}:")
        print(f"    Z_MC  = {Z_mc:.4f}   Z_CS  = {Z_cs:.4f}   err = {abs(Z_mc-Z_cs)/Z_cs*100:.2f}%")
        print(f"    μ_MC  = {mu_mean:.4f} ± {mu_se:.4f}   μ_CS = {mu_ex_cs:.4f}")
        print(f"    g(σ+) = {g_contact:.4f}")
        print(f"    Acceptance rate = {acc_rate:.3f}")
        print(f"    μ_ex converged? {converged}")

    # ── Save ──
    output_dir.mkdir(parents=True, exist_ok=True)
    eta_str = f"{eta:.4f}".replace(".", "p")
    out_path = output_dir / f"gr_eta{eta_str}.npz"

    np.savez(
        out_path,
        r=r,
        g=g,
        r_bins=rdf_acc.bins,
        eta=np.float64(eta),
        rho=np.float64(rho),
        Z_mc=np.float64(Z_mc),
        Z_cs=np.float64(Z_cs),
        mu_ex_mc=np.float64(mu_mean),
        mu_ex_mc_std=np.float64(mu_se),
        mu_ex_cs=np.float64(mu_ex_cs),
        g_contact=np.float64(g_contact),
        acceptance_rate=np.float64(acc_rate),
        n_particles=np.int64(N),
        box_length=np.float64(L),
        n_equil=np.int64(n_equil),
        n_prod=np.int64(n_prod),
        n_sampled=np.int64(n_sampled),
    )
    if verbose:
        print(f"  Saved → {out_path}")

    return dict(
        r=r, g=g, eta=eta, rho=rho,
        Z_mc=Z_mc, Z_cs=Z_cs,
        mu_ex_mc=mu_mean, mu_ex_mc_std=mu_se, mu_ex_cs=mu_ex_cs,
        g_contact=g_contact,
        acceptance_rate=acc_rate,
        converged=converged,
    )


def _run_sweeps(
    nvt_state: NVTState,
    n_sweeps: int,
    N: int,
    max_disp: float,
    sigma: float,
    verbose: bool = False,
    label: str = "sweep",
) -> tuple[NVTState, float]:
    """Run n_sweeps with optional displacement tuning, return (state, acc_rate)."""
    ADJUST_EVERY = 100
    n_acc_block = 0
    n_att_block = 0
    disp = float(max_disp)

    for s in range(n_sweeps):
        nvt_state, n_acc = mc_sweep(nvt_state, N, disp, sigma)
        n_acc_block += n_acc
        n_att_block += N

        if (s + 1) % ADJUST_EVERY == 0:
            rate = n_acc_block / max(n_att_block, 1)
            if rate > 0.40:
                disp *= 1.05
            elif rate < 0.30:
                disp *= 0.95
            # Clamp: never displace more than half the box
            L_half = nvt_state.mc_state.box_length / 2.0
            disp = float(np.clip(disp, 0.01, float(L_half)))
            n_acc_block = 0
            n_att_block = 0

    total_acc = nvt_state.n_accepted
    total_att = nvt_state.n_total
    rate = total_acc / max(total_att, 1)
    return nvt_state, float(rate)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate g(r) and Widom μ_ex for hard-sphere NVT MC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Packing fraction selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--eta", type=float, default=None,
        help="Single packing fraction.",
    )
    group.add_argument(
        "--eta-values", type=str, default=None,
        help="Comma-separated packing fractions, e.g. 0.1,0.2,0.3",
    )

    parser.add_argument("--n-particles", type=int, default=500,
                        help="Number of particles.")
    parser.add_argument("--n-equil", type=int, default=5000,
                        help="Equilibration sweeps.")
    parser.add_argument("--n-prod", type=int, default=20000,
                        help="Production sweeps.")
    parser.add_argument("--n-sample", type=int, default=10,
                        help="Sample g(r) / Widom every N sweeps.")
    parser.add_argument("--n-widom", type=int, default=2000,
                        help="Test insertions per sampled frame.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs/mc",
                        help="Directory for output .npz files.")
    parser.add_argument("--rdf-bins", type=int, default=200,
                        help="Number of RDF histogram bins.")
    parser.add_argument("--rdf-rmax", type=float, default=5.0,
                        help="Maximum r for RDF (in units of σ).")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Hard-sphere diameter.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-sweep output.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Determine eta values
    if args.eta is not None:
        eta_values = [args.eta]
    elif args.eta_values is not None:
        eta_values = [float(x.strip()) for x in args.eta_values.split(",")]
    else:
        # Default: representative set
        eta_values = [0.1, 0.2, 0.3, 0.367, 0.393, 0.449]

    output_dir = Path(args.output_dir)
    verbose = not args.quiet

    print(f"CNFMT Monte Carlo — generate_gr")
    print(f"  eta_values : {eta_values}")
    print(f"  N          : {args.n_particles}")
    print(f"  n_equil    : {args.n_equil}")
    print(f"  n_prod     : {args.n_prod}")
    print(f"  n_sample   : {args.n_sample}")
    print(f"  n_widom    : {args.n_widom}")
    print(f"  output_dir : {output_dir}")

    all_results = []
    t_start = time.time()

    for i, eta in enumerate(eta_values):
        print(f"\n[{i+1}/{len(eta_values)}] η = {eta:.4f}")
        result = run_one_eta(
            eta=eta,
            n_particles=args.n_particles,
            n_equil=args.n_equil,
            n_prod=args.n_prod,
            n_sample=args.n_sample,
            n_widom=args.n_widom,
            seed=args.seed + i * 1000,
            sigma=args.sigma,
            rdf_bins=args.rdf_bins,
            rdf_rmax=args.rdf_rmax,
            output_dir=output_dir,
            verbose=verbose,
        )
        all_results.append(result)

    total_time = time.time() - t_start

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'η':>8}  {'Z_MC':>8}  {'Z_CS':>8}  {'err_Z%':>8}  {'μ_MC':>8}  {'μ_CS':>8}")
    print(f"{'-'*70}")
    for r in all_results:
        err_z = abs(r["Z_mc"] - r["Z_cs"]) / r["Z_cs"] * 100
        print(
            f"{r['eta']:8.4f}  {r['Z_mc']:8.4f}  {r['Z_cs']:8.4f}  "
            f"{err_z:8.2f}  {r['mu_ex_mc']:8.4f}  {r['mu_ex_cs']:8.4f}"
        )
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
