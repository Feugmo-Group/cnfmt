"""
LJ Vapor-Liquid Coexistence via Gibbs Ensemble MC
==================================================

Computes the Lennard-Jones phase diagram (coexistence curve) using the
Gibbs Ensemble Monte Carlo method (Panagiotopoulos 1987).

For each temperature T*, two simulation boxes are initialised:
    - Box 1: liquid guess at ρ* ≈ 0.7  (dense)
    - Box 2: vapor  guess at ρ* ≈ 0.05 (sparse)

After equilibration and production, the mean densities of the two boxes
give ρ_liquid(T*) and ρ_vapor(T*) on the coexistence curve.

Known LJ coexistence points (literature values for validation):
    T*=0.7: ρ_liq≈0.84, ρ_vap≈0.002
    T*=0.9: ρ_liq≈0.77, ρ_vap≈0.015
    T*=1.0: ρ_liq≈0.72, ρ_vap≈0.035
    T*=1.1: ρ_liq≈0.65, ρ_vap≈0.075
    T*=1.2: ρ_liq≈0.55, ρ_vap≈0.14

Usage
-----
    # Single temperature
    python -m mc.scripts.lj_coexistence --temperature 1.0 --n-particles 500

    # Full coexistence curve
    python -m mc.scripts.lj_coexistence --temperatures 0.7,0.8,0.9,1.0,1.1,1.2

    # Quick test (fewer steps)
    python -m mc.scripts.lj_coexistence --temperature 1.0 --n-particles 200 \\
           --n-equil 5000 --n-prod 20000

References
----------
Panagiotopoulos, A.Z. (1987). Molecular Physics, 61, 813–826.
Vrabec, J. & Fischer, J. (1995). Mol. Phys., 85, 781–792. (reference data)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mc.core.potentials import LennardJones
from mc.core.state import MCState
from mc.ensembles.gibbs import GibbsSampler


# ── Known LJ coexistence data for validation ──────────────────────────

# (T*, rho_liq*, rho_vap*) — approximate literature values
_LJ_COEX_REF = [
    (0.70, 0.84, 0.002),
    (0.80, 0.80, 0.008),
    (0.90, 0.77, 0.015),
    (1.00, 0.72, 0.035),
    (1.10, 0.65, 0.075),
    (1.20, 0.55, 0.140),
    (1.25, 0.48, 0.200),
    (1.30, 0.35, 0.290),
]


# ── Initialisation helpers ────────────────────────────────────────────


def _fcc_positions(N: int, box_length: float) -> np.ndarray:
    """Return FCC lattice positions for N particles in a cubic box."""
    n_cells = int(math.ceil((N / 4) ** (1.0 / 3.0)))
    a = box_length / n_cells
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                corner = np.array([ix, iy, iz], dtype=float) * a
                for b in basis:
                    positions.append(corner + b * a)
    positions = np.array(positions)
    return (positions[:N] % box_length)


def _random_positions(N: int, box_length: float, rng: np.random.Generator) -> np.ndarray:
    """Return N random positions in [0, box_length)^3 (no overlap check)."""
    return rng.random((N, 3)) * box_length


def _init_boxes(
    N_total: int,
    rho_liq: float,
    rho_vap: float,
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> Tuple[MCState, MCState]:
    """Initialise two simulation boxes at given densities.

    Splits N_total particles between the two boxes according to the
    density ratio: N1/N2 ≈ rho_liq/rho_vap.  Box 1 (liquid) is
    initialised on an FCC lattice; box 2 (vapor) uses random placement.

    Parameters
    ----------
    N_total : int
        Total number of particles (conserved).
    rho_liq : float
        Initial liquid density ρ* (reduced units, σ=1).
    rho_vap : float
        Initial vapor density ρ*.
    rng : np.random.Generator
    sigma : float
        Hard-sphere diameter (σ* = 1 in reduced units).

    Returns
    -------
    state1, state2 : MCState
        Box 1 (liquid) and box 2 (vapor).
    """
    # Split N proportionally to density ratio
    frac_liq = rho_liq / (rho_liq + rho_vap)
    N1 = max(4, min(N_total - 4, int(round(frac_liq * N_total))))
    N2 = N_total - N1

    # Box sizes from target densities
    V1 = N1 / rho_liq
    V2 = N2 / rho_vap
    L1 = V1 ** (1.0 / 3.0)
    L2 = V2 ** (1.0 / 3.0)

    # Liquid box: FCC lattice
    pos1 = _fcc_positions(N1, L1)

    # Vapor box: random (low density, overlaps very unlikely)
    pos2 = _random_positions(N2, L2, rng)

    import jax.numpy as jnp
    state1 = MCState(
        positions=jnp.array(pos1),
        box_length=L1,
        n_particles=N1,
    )
    state2 = MCState(
        positions=jnp.array(pos2),
        box_length=L2,
        n_particles=N2,
    )
    return state1, state2


# ── Single temperature run ────────────────────────────────────────────


def run_gibbs_coexistence(
    temperature: float,
    n_particles: int = 500,
    n_equil: int = 50_000,
    n_prod: int = 200_000,
    r_cut: float = 2.5,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run Gibbs Ensemble MC at a single temperature T*.

    Parameters
    ----------
    temperature : float
        Reduced temperature T* = kT/ε.
    n_particles : int
        Total number of particles (split between two boxes).
    n_equil : int
        Number of equilibration sweeps.
    n_prod : int
        Number of production sweeps.
    r_cut : float
        LJ cutoff radius (in σ units).
    seed : int
        Random seed.
    verbose : bool
        Print progress during run.

    Returns
    -------
    dict with keys:
        'temperature', 'rho_liquid', 'rho_vapor',
        'std_rho_liquid', 'std_rho_vapor',
        'mean_N1', 'mean_N2', 'acc_xfer', 'acc_vol',
        'runtime_seconds'
    """
    t_start = time.time()
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  T* = {temperature:.3f}   N = {n_particles}")
        print(f"  n_equil = {n_equil}   n_prod = {n_prod}")
        print(f"{'='*60}")

    # Potential
    pot = LennardJones(epsilon=1.0, sigma=1.0, r_cut=r_cut)
    beta = 1.0 / temperature

    # Initial densities (liquid guess: ρ*=0.7, vapor guess: ρ*=0.05)
    rho_liq_init = 0.7
    rho_vap_init = 0.05

    # Adjust for very low or very high temperatures
    if temperature <= 0.75:
        rho_liq_init = 0.82
        rho_vap_init = 0.005
    elif temperature >= 1.25:
        rho_liq_init = 0.50
        rho_vap_init = 0.20

    if verbose:
        print(f"  Initial densities: ρ_liq={rho_liq_init:.3f}, ρ_vap={rho_vap_init:.3f}")

    state1, state2 = _init_boxes(n_particles, rho_liq_init, rho_vap_init, rng)

    if verbose:
        print(f"  Box 1: N={state1.n_particles}, L={state1.box_length:.3f}, "
              f"ρ={state1.density():.4f}")
        print(f"  Box 2: N={state2.n_particles}, L={state2.box_length:.3f}, "
              f"ρ={state2.density():.4f}")

    # Gibbs sampler
    sampler = GibbsSampler(
        state1=state1,
        state2=state2,
        potential=pot,
        beta=beta,
        seed=seed + 1,
    )

    # Run MC
    results = sampler.run(
        n_equil=n_equil,
        n_prod=n_prod,
        max_disp=0.15,
        max_vol_step=0.05,
        n_sample=max(1, n_prod // 2000),
        verbose=verbose,
    )

    runtime = time.time() - t_start

    # Compute statistics
    rho1 = np.array(results["rho1"])
    rho2 = np.array(results["rho2"])

    mean_rho1 = results["mean_rho1"]
    mean_rho2 = results["mean_rho2"]
    std_rho1 = float(np.std(rho1)) if len(rho1) > 1 else 0.0
    std_rho2 = float(np.std(rho2)) if len(rho2) > 1 else 0.0

    if mean_rho1 >= mean_rho2:
        rho_liquid = mean_rho1
        rho_vapor = mean_rho2
        std_liq = std_rho1
        std_vap = std_rho2
    else:
        rho_liquid = mean_rho2
        rho_vapor = mean_rho1
        std_liq = std_rho2
        std_vap = std_rho1

    if verbose:
        print(f"\n  Results at T* = {temperature:.3f}:")
        print(f"    ρ_liquid = {rho_liquid:.4f} ± {std_liq:.4f}")
        print(f"    ρ_vapor  = {rho_vapor:.4f} ± {std_vap:.4f}")
        print(f"    Transfer acceptance: {results['acc_xfer']:.3f}")
        print(f"    Volume acceptance:   {results['acc_vol']:.3f}")
        print(f"    Runtime: {runtime:.1f} s")

    return {
        "temperature": temperature,
        "rho_liquid": rho_liquid,
        "rho_vapor": rho_vapor,
        "std_rho_liquid": std_liq,
        "std_rho_vapor": std_vap,
        "mean_N1": results["mean_N1"],
        "mean_N2": results["mean_N2"],
        "acc_xfer": results["acc_xfer"],
        "acc_vol": results["acc_vol"],
        "acc_disp1": results["acc_disp1"],
        "acc_disp2": results["acc_disp2"],
        "runtime_seconds": runtime,
        "rho1_timeseries": rho1.tolist(),
        "rho2_timeseries": rho2.tolist(),
    }


# ── Output / plotting ─────────────────────────────────────────────────


def save_results(results: List[Dict], output_dir: Path) -> None:
    """Save coexistence data to JSON and plain-text table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "lj_coexistence.json"
    with open(json_path, "w") as f:
        # Remove large timeseries from JSON (keep summary only)
        summary = []
        for r in results:
            rec = {k: v for k, v in r.items() if "timeseries" not in k}
            summary.append(rec)
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plain-text table
    txt_path = output_dir / "lj_coexistence.txt"
    with open(txt_path, "w") as f:
        f.write("# LJ Vapor-Liquid Coexistence (Gibbs Ensemble MC)\n")
        f.write("#\n")
        f.write("# T*      rho_liq   std_liq   rho_vap   std_vap   "
                "acc_xfer  acc_vol   runtime(s)\n")
        f.write("# " + "-" * 80 + "\n")
        for r in results:
            f.write(
                f"  {r['temperature']:.4f}  "
                f"{r['rho_liquid']:.5f}  {r['std_rho_liquid']:.5f}  "
                f"{r['rho_vapor']:.5f}  {r['std_rho_vapor']:.5f}  "
                f"{r['acc_xfer']:.4f}  {r['acc_vol']:.4f}  "
                f"{r['runtime_seconds']:.1f}\n"
            )
    print(f"Saved: {txt_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  LJ Coexistence Curve Summary")
    print(f"{'='*70}")
    print(f"  {'T*':>6}  {'ρ_liquid':>10}  {'ρ_vapor':>10}  "
          f"{'acc_xfer':>10}  {'acc_vol':>8}")
    print(f"  {'-'*60}")
    for r in results:
        print(
            f"  {r['temperature']:>6.3f}  "
            f"{r['rho_liquid']:>10.4f}  "
            f"{r['rho_vapor']:>10.4f}  "
            f"{r['acc_xfer']:>10.3f}  "
            f"{r['acc_vol']:>8.3f}"
        )
    print(f"{'='*70}")

    # Validation against reference data
    _validate_against_reference(results)


def _validate_against_reference(results: List[Dict]) -> None:
    """Compare results against known LJ coexistence data."""
    ref_dict = {T: (rl, rv) for T, rl, rv in _LJ_COEX_REF}

    print(f"\n  Validation against reference data:")
    print(f"  {'T*':>6}  {'ρ_liq (MC)':>12}  {'ρ_liq (ref)':>12}  "
          f"{'err_liq%':>9}  {'ρ_vap (MC)':>12}  {'ρ_vap (ref)':>12}  {'err_vap%':>9}")
    print(f"  {'-'*80}")

    for r in results:
        T = r["temperature"]
        # Find closest reference temperature
        closest = min(ref_dict.keys(), key=lambda t: abs(t - T))
        if abs(closest - T) < 0.051:
            rl_ref, rv_ref = ref_dict[closest]
            err_liq = abs(r["rho_liquid"] - rl_ref) / rl_ref * 100
            err_vap = abs(r["rho_vapor"] - rv_ref) / max(rv_ref, 1e-6) * 100
            status_liq = "OK" if err_liq < 5.0 else "WARN"
            status_vap = "OK" if err_vap < 20.0 else "WARN"
            print(
                f"  {T:>6.3f}  "
                f"{r['rho_liquid']:>12.4f}  {rl_ref:>12.4f}  "
                f"{err_liq:>8.1f}%  "
                f"{r['rho_vapor']:>12.4f}  {rv_ref:>12.4f}  "
                f"{err_vap:>8.1f}%  [{status_liq}/{status_vap}]"
            )


def plot_coexistence(results: List[Dict], output_dir: Path) -> None:
    """Plot the coexistence curve if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    T_vals = [r["temperature"] for r in results]
    rho_liq = [r["rho_liquid"] for r in results]
    rho_vap = [r["rho_vapor"] for r in results]
    err_liq = [r["std_rho_liquid"] for r in results]
    err_vap = [r["std_rho_vapor"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Coexistence curve (ρ vs T)
    ax1.errorbar(rho_liq, T_vals, xerr=err_liq, fmt="o-", color="steelblue",
                 label="Liquid (MC)", capsize=3)
    ax1.errorbar(rho_vap, T_vals, xerr=err_vap, fmt="s-", color="tomato",
                 label="Vapor (MC)", capsize=3)

    # Reference data
    ref_T = [d[0] for d in _LJ_COEX_REF]
    ref_rl = [d[1] for d in _LJ_COEX_REF]
    ref_rv = [d[2] for d in _LJ_COEX_REF]
    ax1.plot(ref_rl, ref_T, "b--", alpha=0.5, label="Liquid (ref)")
    ax1.plot(ref_rv, ref_T, "r--", alpha=0.5, label="Vapor (ref)")

    ax1.set_xlabel(r"$\rho^*$", fontsize=13)
    ax1.set_ylabel(r"$T^*$", fontsize=13)
    ax1.set_title("LJ Vapor-Liquid Coexistence", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Density difference (order parameter) vs T
    delta_rho = [rl - rv for rl, rv in zip(rho_liq, rho_vap)]
    ax2.plot(T_vals, delta_rho, "o-", color="purple", label=r"$\Delta\rho$")
    ax2.set_xlabel(r"$T^*$", fontsize=13)
    ax2.set_ylabel(r"$\rho_\mathrm{liq} - \rho_\mathrm{vap}$", fontsize=13)
    ax2.set_title("Density Difference (Order Parameter)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "lj_coexistence_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LJ Vapor-Liquid Coexistence via Gibbs Ensemble MC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Temperature specification (single or multiple)
    temp_group = parser.add_mutually_exclusive_group()
    temp_group.add_argument(
        "--temperature", type=float, default=1.0,
        help="Single reduced temperature T* = kT/ε",
    )
    temp_group.add_argument(
        "--temperatures", type=str, default=None,
        help="Comma-separated list of T* values, e.g. '0.7,0.8,0.9,1.0,1.1,1.2'",
    )

    parser.add_argument(
        "--n-particles", type=int, default=500,
        help="Total number of particles (split between two boxes)",
    )
    parser.add_argument(
        "--n-equil", type=int, default=50_000,
        help="Number of equilibration sweeps",
    )
    parser.add_argument(
        "--n-prod", type=int, default=200_000,
        help="Number of production sweeps",
    )
    parser.add_argument(
        "--r-cut", type=float, default=2.5,
        help="LJ cutoff radius in units of σ",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (incremented for each temperature)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/mc_coex",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting even if matplotlib is available",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-sweep progress output",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse temperature list
    if args.temperatures is not None:
        temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    else:
        temperatures = [args.temperature]

    output_dir = Path(args.output_dir)

    print(f"\nLJ Coexistence — Gibbs Ensemble MC")
    print(f"  Temperatures: {temperatures}")
    print(f"  N_total = {args.n_particles}")
    print(f"  n_equil = {args.n_equil},  n_prod = {args.n_prod}")
    print(f"  r_cut = {args.r_cut} σ")
    print(f"  Output: {output_dir}\n")

    all_results = []
    for k, T in enumerate(sorted(temperatures)):
        result = run_gibbs_coexistence(
            temperature=T,
            n_particles=args.n_particles,
            n_equil=args.n_equil,
            n_prod=args.n_prod,
            r_cut=args.r_cut,
            seed=args.seed + k * 17,
            verbose=not args.quiet,
        )
        all_results.append(result)

    save_results(all_results, output_dir)

    if not args.no_plot:
        plot_coexistence(all_results, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
