"""
Benchmark: NVTSampler (O(N²)) vs CellListNVTSampler (O(N))
============================================================

Runs both samplers at several system sizes, reports timing per sweep,
speedup ratio, and validates that g(r) from both methods agree within
statistical noise.

Usage
-----
    python -m mc.scripts.benchmark --n-values 100,500,1000 --n-sweeps 100 --eta 0.3
"""

from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np

from mc.core.state import MCState
from mc.ensembles.nvt import NVTSampler
from mc.ensembles.nvt_cell import CellListNVTSampler
from mc.observables.rdf import RDFAccumulator


# ── Helper: box length from N and eta ─────────────────────────────────


def box_from_eta(N: int, eta: float, sigma: float = 1.0) -> float:
    """Compute cubic box side L such that packing fraction = eta."""
    V = N * np.pi * sigma ** 3 / (6.0 * eta)
    return V ** (1.0 / 3.0)


# ── g(r) comparison ───────────────────────────────────────────────────


def compare_gr(
    r1: np.ndarray,
    g1: np.ndarray,
    r2: np.ndarray,
    g2: np.ndarray,
    r_max_compare: float = 4.0,
) -> float:
    """Return mean absolute deviation |g1(r) - g2(r)| for r in [1, r_max_compare].

    Both g(r) arrays are assumed to share the same r grid (same n_bins, r_max).
    """
    mask = (r1 >= 1.0) & (r1 <= r_max_compare)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(g1[mask] - g2[mask])))


# ── Single benchmark run for one N ────────────────────────────────────


def run_one(
    N: int,
    eta: float,
    n_sweeps: int,
    sigma: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run NVTSampler and CellListNVTSampler for N particles, n_sweeps.

    Returns timing, acceptance rates, and g(r) MAD.
    """
    L = box_from_eta(N, eta, sigma)

    print(f"\n{'='*60}")
    print(f"  N={N}, η={eta:.2f}, L={L:.3f}, n_sweeps={n_sweeps}")
    print(f"{'='*60}")

    # ── Shared initial configuration (FCC lattice) ──
    mc_state = MCState.from_fcc(N, L)

    # ── NVTSampler (O(N²) all-pairs) ──
    print(f"  [O(N²)] Building NVTSampler...")
    rdf_naive = RDFAccumulator(n_bins=200, r_max=min(L / 2.0, 5.0), sigma=sigma)

    sampler_naive = NVTSampler(mc_state, seed=seed, sigma=sigma)

    t0 = time.perf_counter()
    final_state_naive, acc_naive, _ = sampler_naive.run(
        n_equil=n_sweeps // 4,
        n_prod=n_sweeps,
        max_disp=0.2,
        adjust_disp=True,
        verbose=verbose,
    )
    t_naive = time.perf_counter() - t0

    # Collect g(r) from production positions
    rdf_naive.update(np.asarray(final_state_naive.mc_state.positions), L)

    time_per_sweep_naive = t_naive / n_sweeps
    print(
        f"  [O(N²)] done: {t_naive:.2f}s total, "
        f"{time_per_sweep_naive*1000:.2f} ms/sweep, acc={acc_naive:.3f}"
    )

    # ── CellListNVTSampler (O(N)) ──
    print(f"  [O(N)]  Building CellListNVTSampler...")
    rdf_cell = RDFAccumulator(n_bins=200, r_max=min(L / 2.0, 5.0), sigma=sigma)

    sampler_cell = CellListNVTSampler(mc_state, seed=seed, sigma=sigma)

    t0 = time.perf_counter()
    final_state_cell, acc_cell, _ = sampler_cell.run(
        n_equil=n_sweeps // 4,
        n_prod=n_sweeps,
        max_disp=0.2,
        adjust_disp=True,
        verbose=verbose,
    )
    t_cell = time.perf_counter() - t0

    rdf_cell.update(np.asarray(final_state_cell.positions), L)

    time_per_sweep_cell = t_cell / n_sweeps
    speedup = t_naive / max(t_cell, 1e-9)
    print(
        f"  [O(N)]  done: {t_cell:.2f}s total, "
        f"{time_per_sweep_cell*1000:.2f} ms/sweep, acc={acc_cell:.3f}"
    )
    print(f"  Speedup (O(N²) / O(N)): {speedup:.2f}×")

    # ── g(r) agreement ──
    r_naive, g_naive = rdf_naive.get()
    r_cell, g_cell = rdf_cell.get()
    mad = compare_gr(r_naive, g_naive, r_cell, g_cell)
    print(f"  g(r) MAD (r∈[1,4]): {mad:.4f}")

    # ── Cell list stats ──
    cl = sampler_cell.cell_list
    occ = cl.occupancy_stats()
    print(
        f"  CellList: {cl.n_cells}^3 cells, cell_size={cl.cell_size:.3f}, "
        f"mean occupancy={occ['mean']:.1f}, max={occ['max']}"
    )

    return {
        "N": N,
        "eta": eta,
        "L": L,
        "n_sweeps": n_sweeps,
        "t_naive_s": t_naive,
        "t_cell_s": t_cell,
        "ms_per_sweep_naive": time_per_sweep_naive * 1000,
        "ms_per_sweep_cell": time_per_sweep_cell * 1000,
        "speedup": speedup,
        "acc_naive": acc_naive,
        "acc_cell": acc_cell,
        "gr_mad": mad,
        "r": r_naive,
        "g_naive": g_naive,
        "g_cell": g_cell,
    }


# ── Summary table ─────────────────────────────────────────────────────


def print_summary(results: List[dict]) -> None:
    header = (
        f"{'N':>6}  {'ms/sw (N²)':>12}  {'ms/sw (N)':>10}  "
        f"{'Speedup':>8}  {'acc N²':>7}  {'acc N':>7}  {'g(r) MAD':>9}"
    )
    print(f"\n{'='*75}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*75}")
    print(header)
    print("-" * 75)
    for r in results:
        print(
            f"{r['N']:>6}  {r['ms_per_sweep_naive']:>12.2f}  "
            f"{r['ms_per_sweep_cell']:>10.2f}  {r['speedup']:>8.2f}×  "
            f"{r['acc_naive']:>7.3f}  {r['acc_cell']:>7.3f}  "
            f"{r['gr_mad']:>9.4f}"
        )
    print(f"{'='*75}")

    # g(r) validation verdict
    print("\n  g(r) validation:")
    for r in results:
        mad = r["gr_mad"]
        verdict = "PASS" if mad < 0.05 else "WARN (MAD > 0.05)"
        print(f"    N={r['N']}: MAD={mad:.4f}  [{verdict}]")
    print()


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NVTSampler vs CellListNVTSampler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="100,500,1000",
        help="Comma-separated list of particle counts",
    )
    parser.add_argument(
        "--n-sweeps",
        type=int,
        default=100,
        help="Number of production sweeps per run",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.3,
        help="Target packing fraction",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Hard-sphere diameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-block progress",
    )
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_values.split(",")]

    results = []
    for N in n_values:
        result = run_one(
            N=N,
            eta=args.eta,
            n_sweeps=args.n_sweeps,
            sigma=args.sigma,
            seed=args.seed,
            verbose=args.verbose,
        )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
