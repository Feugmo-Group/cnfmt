"""
Generate Equation of State (EOS) Data via Monte Carlo
======================================================

Runs NVT (hard sphere) or NVT (LJ) Monte Carlo at specified state points,
computes thermodynamic observables, and compares to reference equations of
state (Carnahan-Starling for HS; Lennard-Jones reference data).

Usage
-----
Hard spheres (Z vs η):
    python -m mc.scripts.generate_eos --system hs \\
           --n-particles 256 --eta-values 0.1,0.2,0.3,0.4 \\
           --output-dir outputs/mc_eos

Lennard-Jones (P* vs ρ* at fixed T*):
    python -m mc.scripts.generate_eos --system lj \\
           --temperature 1.5 --density-values 0.3,0.5,0.7,0.8 \\
           --output-dir outputs/mc_eos

Physics
-------
Hard spheres:
  - Run NVT MC to equilibrate configuration.
  - Measure contact value g(σ⁺) via RDF → Z = 1 + 4η·g(σ⁺).
  - Compare Z_MC to Carnahan-Starling: Z_CS = (1+η+η²-η³)/(1-η)³.

Lennard-Jones:
  - Run NVT MC at specified (T*, ρ*).
  - Measure ⟨U⟩/N via direct energy accumulation.
  - Compute P* via virial theorem.
  - Reference: LJ EOS from Johnson et al. (1993).

References
----------
Carnahan & Starling, J. Chem. Phys. 51, 635 (1969).
Johnson, Zollweg & Gubbins, Mol. Phys. 78, 591 (1993).
Davidchack, Laird & Roth, Condens. Matter Phys. 19 (2016).
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

# ── Repo root on path ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mc.core.state import MCState
from mc.core.potentials import HardSphere, LennardJones
from mc.ensembles.nvt import NVTSampler
from mc.observables.rdf import RDFAccumulator


# ── Reference EOS ─────────────────────────────────────────────────────


def z_carnahan_starling(eta: float) -> float:
    """Compressibility factor from Carnahan-Starling EOS.

    Z_CS = (1 + η + η² - η³) / (1 - η)³
    """
    return (1.0 + eta + eta ** 2 - eta ** 3) / (1.0 - eta) ** 3


def mu_ex_cs(eta: float) -> float:
    """Excess chemical potential from CS EOS (βμ_ex).

    βμ_ex = η(8 - 9η + 3η²) / (1 - η)³
    """
    return eta * (8.0 - 9.0 * eta + 3.0 * eta ** 2) / (1.0 - eta) ** 3


# ── Hard-sphere NVT run ───────────────────────────────────────────────


def run_hs_nvt(
    eta: float,
    n_particles: int,
    n_equil: int,
    n_prod: int,
    n_rdf_bins: int = 300,
    sigma: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run NVT MC for hard spheres at packing fraction η.

    Returns
    -------
    dict with keys:
        'eta'        : packing fraction (input)
        'Z_mc'       : compressibility factor from virial (contact RDF)
        'Z_cs'       : Carnahan-Starling reference
        'rel_err'    : |(Z_mc - Z_cs)| / Z_cs
        'g_contact'  : g(σ⁺) contact value
        'r'          : RDF bin centres
        'g'          : g(r) values
    """
    # Box length from η = N π σ³ / (6 V)
    V = n_particles * math.pi * sigma ** 3 / (6.0 * eta)
    box_length = V ** (1.0 / 3.0)
    density = n_particles / V

    if verbose:
        print(f"  η={eta:.3f}  N={n_particles}  L={box_length:.4f}  ρ={density:.4f}")

    # Initialise: FCC for η > 0.25, random otherwise
    if eta > 0.25:
        mc_state = MCState.from_fcc(n_particles, box_length)
    else:
        mc_state = MCState.from_random(n_particles, box_length, seed=seed, sigma=sigma)

    sampler = NVTSampler(mc_state, seed=seed, sigma=sigma)

    # RDF accumulator: r_max = min(L/2, 3.0σ) — captures first few peaks
    r_max = min(box_length / 2.0 - 0.01, 3.5 * sigma)
    rdf_acc = RDFAccumulator(n_bins=n_rdf_bins, r_max=r_max)

    # Equilibrate
    sampler.run(n_equil=n_equil, n_prod=0, max_disp=0.2, verbose=False)

    # Production: sample RDF every sweep
    N = n_particles
    from mc.ensembles.nvt import mc_sweep, NVTState
    import jax
    state = sampler.nvt_state

    for sweep in range(n_prod):
        state, _ = mc_sweep(state, N, max_disp=0.2, sigma=float(sampler.sigma))
        # Sample every 10 sweeps to reduce correlation
        if (sweep + 1) % 10 == 0:
            pos_np = np.array(state.mc_state.positions)
            rdf_acc.update(pos_np, box_length)

    r, g = rdf_acc.get()

    # Extract g(σ⁺): average g(r) in first bin above σ
    # Contact value at r = σ (first peak of HS g(r))
    contact_mask = (r >= sigma) & (r <= sigma + 0.1)
    if np.any(contact_mask):
        g_contact = float(np.mean(g[contact_mask]))
    else:
        # Fallback: value in first bin above sigma
        idx = int(np.searchsorted(r, sigma))
        g_contact = float(g[min(idx, len(g) - 1)])

    # Virial route: Z = 1 + 4η g(σ⁺) for hard spheres
    Z_mc = 1.0 + 4.0 * eta * g_contact
    Z_cs = z_carnahan_starling(eta)
    rel_err = abs(Z_mc - Z_cs) / abs(Z_cs)

    if verbose:
        print(f"    g(σ⁺)={g_contact:.4f}  Z_MC={Z_mc:.4f}  Z_CS={Z_cs:.4f}  err={rel_err*100:.2f}%")

    return {
        "eta": eta,
        "Z_mc": Z_mc,
        "Z_cs": Z_cs,
        "rel_err": rel_err,
        "g_contact": g_contact,
        "density": density,
        "r": r,
        "g": g,
    }


# ── LJ NVT run ────────────────────────────────────────────────────────


def run_lj_nvt(
    rho: float,
    temperature: float,
    n_particles: int,
    n_equil: int,
    n_prod: int,
    r_cut: float = 2.5,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run NVT MC for Lennard-Jones fluid at (ρ*, T*).

    Returns
    -------
    dict with keys:
        'density'    : number density ρ* (input)
        'temperature': T* (input)
        'U_per_N'    : ⟨U/N⟩ potential energy per particle
        'P_virial'   : pressure from virial theorem
        'P_ideal'    : ideal gas contribution ρkT
        'Z'          : compressibility factor P/(ρkT)
    """
    V = n_particles / rho
    box_length = V ** (1.0 / 3.0)
    beta = 1.0 / temperature

    if verbose:
        print(f"  ρ={rho:.3f}  T={temperature:.3f}  N={n_particles}  L={box_length:.4f}")

    lj = LennardJones(epsilon=epsilon, sigma=sigma, r_cut=r_cut)

    # Initialise: FCC for high density, random for low
    eta_approx = rho * math.pi * sigma ** 3 / 6.0
    if eta_approx > 0.25:
        mc_state = MCState.from_fcc(n_particles, box_length)
    else:
        mc_state = MCState.from_random(n_particles, box_length, seed=seed, sigma=sigma)

    # NVT MC with Metropolis criterion for LJ
    # We run manually using NumPy for soft potentials
    positions = np.array(mc_state.positions, dtype=float)
    rng = np.random.default_rng(seed)

    def lj_single(pos_i, idx):
        return lj.single_particle_energy(pos_i, positions, idx, box_length)

    max_disp = 0.1
    n_acc = 0
    n_att = 0

    def sweep():
        nonlocal n_acc, n_att, max_disp
        for _ in range(n_particles):
            n_att += 1
            i = int(rng.integers(0, n_particles))
            old_pos = positions[i].copy()
            delta = max_disp * (rng.random(3) - 0.5)
            new_pos = (old_pos + delta) % box_length

            dU = lj_single(new_pos, i) - lj_single(old_pos, i)
            if dU <= 0.0 or rng.random() < math.exp(-beta * dU):
                positions[i] = new_pos
                n_acc += 1

    def tune():
        nonlocal max_disp, n_acc, n_att
        if n_att > 0:
            rate = n_acc / n_att
            if rate > 0.40:
                max_disp *= 1.05
            elif rate < 0.30:
                max_disp *= 0.95
            max_disp = float(np.clip(max_disp, 0.001, box_length / 2.0))
        n_acc = n_att = 0

    # Equilibrate
    for s in range(n_equil):
        sweep()
        if (s + 1) % 100 == 0:
            tune()

    n_acc = n_att = 0

    # Production: accumulate energy and virial
    energies = []
    virials = []
    SAMPLE_EVERY = 10

    for s in range(n_prod):
        sweep()
        if (s + 1) % 100 == 0:
            tune()
        if (s + 1) % SAMPLE_EVERY == 0:
            U = lj.pair_energy(positions, box_length)
            # Add tail correction
            U += lj.tail_correction_energy(n_particles, rho)
            energies.append(U / n_particles)

            # Virial pressure: P = ρkT + W/V
            # W = (1/3) Σ r_ij · f_ij for LJ
            # For LJ 12-6: W = (1/3V) Σ [48ε((σ/r)^12 - 0.5(σ/r)^6)]
            W = _lj_virial(positions, box_length, lj)
            # Add tail correction to pressure
            P_tail = lj.tail_correction_pressure(rho)
            P_virial = rho / beta + W / V + P_tail
            virials.append(float(P_virial))

    energies = np.array(energies)
    virials = np.array(virials)

    U_per_N = float(np.mean(energies))
    P_virial = float(np.mean(virials))
    P_ideal = rho / beta
    Z = P_virial / (rho / beta) if rho > 0 else 0.0

    if verbose:
        print(
            f"    U/N={U_per_N:.4f}  P*={P_virial:.4f}  "
            f"P_ideal={P_ideal:.4f}  Z={Z:.4f}"
        )

    return {
        "density": rho,
        "temperature": temperature,
        "U_per_N": U_per_N,
        "P_virial": P_virial,
        "P_ideal": P_ideal,
        "Z": Z,
    }


def _lj_virial(positions: np.ndarray, box_length: float, lj: LennardJones) -> float:
    """Virial W = (1/(3)) Σ_{i<j} r_ij · (dU/dr_ij).

    For LJ: dU/dr = (24ε/r)[2(σ/r)^12 - (σ/r)^6]
    So r · dU/dr = 24ε[2(σ/r)^12 - (σ/r)^6]
    """
    positions = np.asarray(positions)
    N = len(positions)
    rc2 = lj.r_cut_abs ** 2
    sigma2 = lj.sigma ** 2
    W = 0.0

    for i in range(N - 1):
        dr = positions[i + 1:] - positions[i]
        dr -= box_length * np.round(dr / box_length)
        r2 = np.sum(dr ** 2, axis=1)
        mask = r2 < rc2
        if not np.any(mask):
            continue
        inv_r2 = sigma2 / r2[mask]
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        # w_ij = 24ε [2(σ/r)^12 - (σ/r)^6]
        W += float(np.sum(24.0 * lj.epsilon * (2.0 * inv_r12 - inv_r6)))

    return W / 3.0


# ── Table printing ────────────────────────────────────────────────────


def print_hs_table(results: List[dict]) -> None:
    print("\n" + "=" * 65)
    print(f"{'η':>8}  {'ρ':>8}  {'g(σ+)':>8}  {'Z_MC':>8}  {'Z_CS':>8}  {'err%':>7}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['eta']:>8.4f}  {r['density']:>8.4f}  {r['g_contact']:>8.4f}  "
            f"{r['Z_mc']:>8.4f}  {r['Z_cs']:>8.4f}  {r['rel_err']*100:>7.2f}"
        )
    print("=" * 65)


def print_lj_table(results: List[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{'ρ*':>7}  {'T*':>7}  {'U/N':>8}  {'P*':>8}  {'P_id':>8}  {'Z':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['density']:>7.4f}  {r['temperature']:>7.4f}  {r['U_per_N']:>8.4f}  "
            f"{r['P_virial']:>8.4f}  {r['P_ideal']:>8.4f}  {r['Z']:>8.4f}"
        )
    print("=" * 70)


# ── Plotting ──────────────────────────────────────────────────────────


def _plot_hs_eos(results: List[dict], output_dir: Path) -> None:
    """Plot Z(η) comparison: MC vs Carnahan-Starling."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping plot")
        return

    eta_vals = np.linspace(0.01, 0.50, 200)
    Z_cs_ref = np.array([z_carnahan_starling(e) for e in eta_vals])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(eta_vals, Z_cs_ref, "k-", lw=1.5, label="Carnahan-Starling")
    eta_mc = [r["eta"] for r in results]
    Z_mc = [r["Z_mc"] for r in results]
    Z_cs_pt = [r["Z_cs"] for r in results]
    ax.plot(eta_mc, Z_mc, "rs", ms=8, label="NVT MC (virial)")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$Z = \beta P / \rho$")
    ax.set_title("Hard Sphere EOS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    errs = [abs(r["Z_mc"] - r["Z_cs"]) / r["Z_cs"] * 100 for r in results]
    ax2.bar(eta_mc, errs, width=0.01, color="steelblue", alpha=0.7)
    ax2.axhline(1.0, color="r", ls="--", lw=1, label="1 % threshold")
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel(r"$|Z_{MC} - Z_{CS}| / Z_{CS}$ [%]")
    ax2.set_title("Relative Error vs CS")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "hs_eos.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved {out_path}")


def _plot_lj_eos(results: List[dict], output_dir: Path) -> None:
    """Plot P*(ρ*) for LJ NVT runs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping plot")
        return

    rho_vals = [r["density"] for r in results]
    P_vals = [r["P_virial"] for r in results]
    T_val = results[0]["temperature"] if results else 1.0
    U_vals = [r["U_per_N"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(rho_vals, P_vals, "bs-", ms=8, label=f"NVT MC (T*={T_val})")
    ax.set_xlabel(r"$\rho^*$")
    ax.set_ylabel(r"$P^*$")
    ax.set_title("LJ EOS (NVT)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    ax2.plot(rho_vals, U_vals, "rs-", ms=8, label=f"T*={T_val}")
    ax2.axhline(-5.0, color="k", ls="--", lw=1, label="Ref: U/N≈-5 at ρ*=0.8")
    ax2.set_xlabel(r"$\rho^*$")
    ax2.set_ylabel(r"$\langle U \rangle / N$")
    ax2.set_title("LJ Energy per Particle")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = output_dir / f"lj_eos_T{T_val:.2f}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate EOS data via NVT Monte Carlo."
    )
    p.add_argument(
        "--system",
        choices=["hs", "lj"],
        default="hs",
        help="System type: 'hs' (hard sphere) or 'lj' (Lennard-Jones).",
    )
    p.add_argument(
        "--n-particles",
        type=int,
        default=256,
        help="Number of particles (default: 256).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Reduced temperature T* for LJ (default: 1.5).",
    )
    p.add_argument(
        "--eta-values",
        type=str,
        default="0.1,0.2,0.3,0.4",
        help="Comma-separated packing fractions for HS (default: 0.1,0.2,0.3,0.4).",
    )
    p.add_argument(
        "--density-values",
        type=str,
        default="0.3,0.5,0.7,0.8",
        help="Comma-separated densities ρ* for LJ (default: 0.3,0.5,0.7,0.8).",
    )
    p.add_argument(
        "--n-equil",
        type=int,
        default=2000,
        help="Number of equilibration sweeps (default: 2000).",
    )
    p.add_argument(
        "--n-prod",
        type=int,
        default=10000,
        help="Number of production sweeps (default: 10000).",
    )
    p.add_argument(
        "--r-cut",
        type=float,
        default=2.5,
        help="LJ cutoff in units of σ (default: 2.5).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mc_eos",
        help="Output directory (default: outputs/mc_eos).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-state-point progress.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MC EOS generation — system: {args.system.upper()}")
    print(f"  N={args.n_particles}  n_equil={args.n_equil}  n_prod={args.n_prod}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    if args.system == "hs":
        eta_values = [float(x.strip()) for x in args.eta_values.split(",")]
        results = []
        for eta in eta_values:
            print(f"Running HS NVT  η={eta:.4f} ...")
            res = run_hs_nvt(
                eta=eta,
                n_particles=args.n_particles,
                n_equil=args.n_equil,
                n_prod=args.n_prod,
                seed=args.seed,
                verbose=args.verbose,
            )
            results.append(res)

        print_hs_table(results)

        # Save NPZ
        out_npz = output_dir / "hs_eos.npz"
        np.savez(
            out_npz,
            eta=np.array([r["eta"] for r in results]),
            Z_mc=np.array([r["Z_mc"] for r in results]),
            Z_cs=np.array([r["Z_cs"] for r in results]),
            g_contact=np.array([r["g_contact"] for r in results]),
            density=np.array([r["density"] for r in results]),
        )
        print(f"\n  Saved: {out_npz}")
        _plot_hs_eos(results, output_dir)

    else:  # lj
        density_values = [float(x.strip()) for x in args.density_values.split(",")]
        results = []
        for rho in density_values:
            print(f"Running LJ NVT  ρ*={rho:.4f}  T*={args.temperature:.4f} ...")
            res = run_lj_nvt(
                rho=rho,
                temperature=args.temperature,
                n_particles=args.n_particles,
                n_equil=args.n_equil,
                n_prod=args.n_prod,
                r_cut=args.r_cut,
                seed=args.seed,
                verbose=args.verbose,
            )
            results.append(res)

        print_lj_table(results)

        out_npz = output_dir / f"lj_eos_T{args.temperature:.2f}.npz"
        np.savez(
            out_npz,
            density=np.array([r["density"] for r in results]),
            temperature=np.array([r["temperature"] for r in results]),
            U_per_N=np.array([r["U_per_N"] for r in results]),
            P_virial=np.array([r["P_virial"] for r in results]),
            Z=np.array([r["Z"] for r in results]),
        )
        print(f"\n  Saved: {out_npz}")
        _plot_lj_eos(results, output_dir)

    elapsed = time.perf_counter() - t0
    print(f"\n  Done in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
