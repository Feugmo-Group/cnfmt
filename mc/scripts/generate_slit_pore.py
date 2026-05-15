"""
Generate Slit Pore Density Profiles via GCMC
=============================================

Simulates hard spheres confined between two parallel hard walls (a slit
pore) using Grand Canonical Monte Carlo and accumulates the density
profile ρ(z).

Geometry
--------
Two hard walls at z = 0 and z = H.
Particles have radius R = σ/2 and are confined to:
    R < z < H - R   (accessible region of width H - σ)

PBC are applied in x and y only.

The bulk reservoir is characterised by:
  - Bulk packing fraction η_bulk → ρ_bulk = 6η/(πσ³)
  - Excess chemical potential βμ_ex(η) from Carnahan-Starling EOS
  - Activity z_act = ρ_bulk · exp(βμ_ex)

Usage
-----
Single width:
    python -m mc.scripts.generate_slit_pore --eta 0.367 --width 3.0 \\
           --output-dir outputs/mc_slit

Multiple widths:
    python -m mc.scripts.generate_slit_pore --eta 0.367 \\
           --widths 2.0,3.0,4.0,5.0 \\
           --output-dir outputs/mc_slit

Physics background
------------------
For a slit pore the density profile ρ(z) shows pronounced oscillations
from both walls that interfere for narrow widths (H < 4σ).  This is a
stringent test of any DFT functional because:

1. Contact theorem at BOTH walls must be satisfied:
   ρ(z = R) = ρ(z = H-R) = βP_bulk  (by symmetry if walls are identical)

2. Narrow pores show commensurability: density is enhanced at
   H = nσ (n integer) and reduced at H = (n + 1/2)σ.

3. The oscillation wavelength is ~σ regardless of pore width.

Output
------
For each width H, saves:
  <output-dir>/slit_eta{η:.3f}_H{H:.2f}.npz  — z, rho arrays
  <output-dir>/slit_eta{η:.3f}_H{H:.2f}.png  — plot of ρ(z)/ρ_bulk vs z/σ
  <output-dir>/slit_summary_eta{η:.3f}.npz   — all widths combined

References
----------
Snook & Henderson, J. Chem. Phys. 68, 2134 (1978).
Groot, Faber & van der Eerden, Mol. Phys. 62, 861 (1987).
González et al., J. Chem. Phys. 107, 4349 (1997).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Repo root on path ─────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mc.ensembles.gcmc import GCMCSampler
from mc.observables.density_profile import DensityProfileAccumulator


# ── Reference thermodynamics ──────────────────────────────────────────


def mu_ex_cs(eta: float) -> float:
    """Excess chemical potential βμ_ex from Carnahan-Starling EOS.

    βμ_ex = η(8 - 9η + 3η²) / (1 - η)³
    """
    return eta * (8.0 - 9.0 * eta + 3.0 * eta ** 2) / (1.0 - eta) ** 3


def z_cs(eta: float) -> float:
    """Compressibility factor Z_CS = (1 + η + η² - η³) / (1 - η)³."""
    return (1.0 + eta + eta ** 2 - eta ** 3) / (1.0 - eta) ** 3


# ── Slit pore GCMC ────────────────────────────────────────────────────


class SlitPoreGCMC:
    """GCMC simulation for hard spheres in a slit pore.

    Extends the GCMCSampler from Phase 2 to handle two walls (at z = 0
    and z = H) instead of one.

    Parameters
    ----------
    Lx, Ly : float
        Box dimensions in the periodic (xy) directions.
    H : float
        Slit width H (wall-to-wall distance, in units of σ).
    mu_ex : float
        Excess chemical potential βμ_ex (from CS EOS).
    rho_bulk : float
        Target bulk density ρ_bulk = 6η/(πσ³).
    sigma : float
        Hard-sphere diameter (default 1.0).
    seed : int
        NumPy random seed.
    """

    def __init__(
        self,
        Lx: float,
        Ly: float,
        H: float,
        mu_ex: float,
        rho_bulk: float,
        sigma: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.H = float(H)
        self.sigma = float(sigma)
        self.R = sigma / 2.0             # particle radius
        self.sigma2 = sigma ** 2
        self.rho_bulk = float(rho_bulk)
        self.mu_ex = float(mu_ex)
        self.rng = np.random.default_rng(seed)

        # Volume of the accessible region (between the two walls)
        self.V_acc = Lx * Ly * max(H - sigma, 0.0)
        # Activity
        self.z_act = rho_bulk * math.exp(mu_ex)

        # Particle list
        self._positions: List[np.ndarray] = []
        self._initialise()

        # Move counters
        self.n_disp_acc = self.n_disp_att = 0
        self.n_ins_acc = self.n_ins_att = 0
        self.n_del_acc = self.n_del_att = 0

    # ── Initialisation ────────────────────────────────────────────────

    def _initialise(self) -> None:
        """Place initial particles at ~10 % of target density."""
        if self.H <= self.sigma:
            # Pore too narrow for any particle
            self._positions = []
            return

        N_target = max(5, int(0.10 * self.rho_bulk * self.V_acc))
        N_target = min(N_target, 300)
        placed: List[np.ndarray] = []
        z_min = self.R
        z_max = self.H - self.R

        for _ in range(N_target * 1000):
            if len(placed) >= N_target:
                break
            candidate = np.array([
                self.rng.uniform(0.0, self.Lx),
                self.rng.uniform(0.0, self.Ly),
                self.rng.uniform(z_min, z_max),
            ])
            if len(placed) == 0 or not self._overlap_pbc(candidate, np.array(placed)):
                placed.append(candidate)

        self._positions = placed

    def _overlap_pbc(self, pos: np.ndarray, others: np.ndarray) -> bool:
        """Check if *pos* overlaps any particle in *others* (PBC in x,y only)."""
        if len(others) == 0:
            return False
        dr = others - pos
        dr[:, 0] -= self.Lx * np.round(dr[:, 0] / self.Lx)
        dr[:, 1] -= self.Ly * np.round(dr[:, 1] / self.Ly)
        r2 = np.sum(dr ** 2, axis=1)
        return bool(np.any(r2 < self.sigma2))

    # ── Properties ────────────────────────────────────────────────────

    @property
    def N(self) -> int:
        return len(self._positions)

    @property
    def positions(self) -> np.ndarray:
        if self.N == 0:
            return np.empty((0, 3), dtype=float)
        return np.array(self._positions)

    # ── Move implementations ──────────────────────────────────────────

    def _try_displace(self, max_disp: float) -> None:
        if self.N == 0:
            return
        self.n_disp_att += 1
        idx = int(self.rng.integers(0, self.N))
        old_pos = self._positions[idx].copy()

        delta = max_disp * (self.rng.random(3) - 0.5)
        new_pos = old_pos + delta
        new_pos[0] %= self.Lx
        new_pos[1] %= self.Ly

        # Slit pore wall constraints: R < z < H - R
        if new_pos[2] < self.R or new_pos[2] > self.H - self.R:
            return

        others = np.array(self._positions[:idx] + self._positions[idx + 1:])
        if len(others) > 0 and self._overlap_pbc(new_pos, others):
            return

        self._positions[idx] = new_pos
        self.n_disp_acc += 1

    def _try_insert(self) -> None:
        if self.H <= self.sigma:
            return
        self.n_ins_att += 1

        z_min = self.R
        z_max = self.H - self.R
        candidate = np.array([
            self.rng.uniform(0.0, self.Lx),
            self.rng.uniform(0.0, self.Ly),
            self.rng.uniform(z_min, z_max),
        ])

        if self.N > 0 and self._overlap_pbc(candidate, np.array(self._positions)):
            return

        # Acceptance probability: V_acc * z_act / (N + 1)
        acc = min(1.0, self.V_acc * self.z_act / (self.N + 1))
        if self.rng.random() < acc:
            self._positions.append(candidate)
            self.n_ins_acc += 1

    def _try_delete(self) -> None:
        if self.N == 0:
            return
        self.n_del_att += 1

        idx = int(self.rng.integers(0, self.N))
        acc = min(1.0, self.N / (self.V_acc * self.z_act))
        if self.rng.random() < acc:
            del self._positions[idx]
            self.n_del_acc += 1

    # ── Run ───────────────────────────────────────────────────────────

    def run(
        self,
        n_equil: int,
        n_prod: int,
        n_bins: int,
        max_disp: float = 0.15,
        adjust_disp: bool = True,
        n_adjust: int = 500,
        n_sample: int = 10,
        verbose: bool = False,
        n_report: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run GCMC and return ρ(z) profile.

        Parameters
        ----------
        n_equil : int
            Equilibration steps.
        n_prod : int
            Production steps.
        n_bins : int
            Number of histogram bins along z.
        max_disp : float
            Initial max displacement.
        adjust_disp : bool
            Tune max_disp to target 30-40 % acceptance.
        n_adjust : int
            Tune every this many steps.
        n_sample : int
            Accumulate density profile every *n_sample* steps.
        verbose : bool
            Print progress.
        n_report : int
            Report interval when verbose.

        Returns
        -------
        z_centers : (n_bins,) array
            Bin centres in units of σ.
        rho : (n_bins,) array
            Mean density ρ(z) in units of σ⁻³.
        """
        acc = DensityProfileAccumulator(
            n_bins=n_bins,
            Lz=self.H,
            Lx=self.Lx,
            Ly=self.Ly,
            sigma=self.sigma,
        )

        def _step(disp: float) -> None:
            move = self.rng.integers(0, 3)
            if move == 0:
                self._try_displace(disp)
            elif move == 1:
                self._try_insert()
            else:
                self._try_delete()

        def _tune(disp: float) -> float:
            if self.n_disp_att > 0:
                rate = self.n_disp_acc / self.n_disp_att
                if rate > 0.40:
                    disp *= 1.05
                elif rate < 0.30:
                    disp *= 0.95
                disp = float(np.clip(disp, 0.01, min(self.Lx, self.Ly, self.H) / 2.0))
            self.n_disp_acc = self.n_disp_att = 0
            return disp

        disp = float(max_disp)

        # Equilibration
        for step in range(n_equil):
            _step(disp)
            if adjust_disp and (step + 1) % n_adjust == 0:
                disp = _tune(disp)
            if verbose and (step + 1) % n_report == 0:
                print(
                    f"  [equil] {step+1:7d}/{n_equil}  N={self.N:4d}  "
                    f"ρ={self.N/self.V_acc if self.V_acc > 0 else 0:.4f}"
                )

        # Reset
        self.n_disp_acc = self.n_disp_att = 0
        self.n_ins_acc = self.n_ins_att = 0
        self.n_del_acc = self.n_del_att = 0

        # Production
        for step in range(n_prod):
            _step(disp)
            if adjust_disp and (step + 1) % n_adjust == 0:
                disp = _tune(disp)
            if (step + 1) % n_sample == 0:
                acc.update(self.positions)
            if verbose and (step + 1) % n_report == 0:
                acc_d = self.n_disp_acc / max(self.n_disp_att, 1)
                acc_i = self.n_ins_acc / max(self.n_ins_att, 1)
                print(
                    f"  [prod]  {step+1:7d}/{n_prod}  N={self.N:4d}  "
                    f"acc_d={acc_d:.3f}  acc_i={acc_i:.3f}"
                )

        z, rho, _ = acc.get()
        return z, rho


# ── Single-width simulation ────────────────────────────────────────────


def run_slit(
    eta: float,
    H: float,
    Lxy: float = 10.0,
    n_equil: int = 50000,
    n_prod: int = 200000,
    n_bins: int = 200,
    sigma: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Run one slit pore GCMC simulation.

    Parameters
    ----------
    eta : float
        Bulk packing fraction.
    H : float
        Slit width in units of σ.
    Lxy : float
        Box dimensions in x and y (must be ≥ σ, typically 10σ).
    n_equil, n_prod : int
        Number of equilibration and production steps.
    n_bins : int
        Number of z-bins for density profile.
    sigma, seed, verbose : ...

    Returns
    -------
    z : (n_bins,) array  — bin centres in units of σ
    rho : (n_bins,) array — ρ(z) in units of σ⁻³
    rho_bulk : float — bulk number density
    beta_P : float — bulk pressure βP = Z_CS * ρ_bulk
    """
    rho_bulk = 6.0 * eta / (math.pi * sigma ** 3)
    mu_ex = mu_ex_cs(eta)
    Z = z_cs(eta)
    beta_P = Z * rho_bulk   # exact for hard spheres (βP from CS)

    sim = SlitPoreGCMC(
        Lx=Lxy,
        Ly=Lxy,
        H=H,
        mu_ex=mu_ex,
        rho_bulk=rho_bulk,
        sigma=sigma,
        seed=seed,
    )

    z, rho = sim.run(
        n_equil=n_equil,
        n_prod=n_prod,
        n_bins=n_bins,
        verbose=verbose,
    )

    return z, rho, rho_bulk, beta_P


# ── Plotting ──────────────────────────────────────────────────────────


def _plot_single(
    z: np.ndarray,
    rho: np.ndarray,
    rho_bulk: float,
    H: float,
    eta: float,
    beta_P: float,
    out_path: Path,
) -> None:
    """Save a ρ(z)/ρ_bulk plot for one slit width."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    g = rho / rho_bulk if rho_bulk > 0 else rho
    ax.plot(z, g, "b-", lw=1.5, label=r"GCMC $\rho(z)/\rho_b$")

    # Mark wall positions
    sigma = 1.0
    ax.axvline(sigma / 2.0, color="k", ls="--", lw=1.2, label="Wall contact (z=σ/2)")
    ax.axvline(H / sigma - 0.5, color="k", ls="--", lw=1.2)
    ax.axhline(1.0, color="gray", ls=":", lw=1)

    # Contact theorem value βP/ρ_bulk = Z_CS
    Z_val = beta_P / rho_bulk if rho_bulk > 0 else 0.0
    ax.axhline(Z_val, color="r", ls=":", lw=1, label=f"βP/ρ_b = {Z_val:.3f}")

    ax.set_xlabel(r"$z / \sigma$")
    ax.set_ylabel(r"$\rho(z) / \rho_\mathrm{bulk}$")
    ax.set_title(f"Slit pore: η={eta:.3f},  H={H:.2f}σ")
    ax.legend(fontsize=9)
    ax.set_xlim(0, H / sigma)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved {out_path}")


def _plot_multi(
    results: List[dict],
    eta: float,
    out_path: Path,
) -> None:
    """Plot ρ(z)/ρ_bulk for multiple slit widths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(results)))

    for res, c in zip(results, colors):
        H = res["H"]
        z = res["z"]
        rho = res["rho"]
        rho_bulk = res["rho_bulk"]
        g = rho / rho_bulk if rho_bulk > 0 else rho
        ax.plot(z, g, color=c, lw=1.5, label=f"H={H:.1f}σ")

    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$z / \sigma$")
    ax.set_ylabel(r"$\rho(z) / \rho_\mathrm{bulk}$")
    ax.set_title(f"Slit pore density profiles  (η={eta:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GCMC slit pore density profiles for hard spheres."
    )
    p.add_argument(
        "--eta",
        type=float,
        default=0.367,
        help="Bulk packing fraction η (default: 0.367).",
    )
    p.add_argument(
        "--width",
        type=float,
        default=None,
        help="Single slit width H in units of σ.",
    )
    p.add_argument(
        "--widths",
        type=str,
        default=None,
        help="Comma-separated slit widths H/σ (e.g. 2.0,3.0,4.0,5.0).",
    )
    p.add_argument(
        "--Lxy",
        type=float,
        default=10.0,
        help="Lateral box dimension Lx=Ly in units of σ (default: 10.0).",
    )
    p.add_argument(
        "--n-equil",
        type=int,
        default=50000,
        help="Equilibration steps (default: 50000).",
    )
    p.add_argument(
        "--n-prod",
        type=int,
        default=200000,
        help="Production steps (default: 200000).",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=200,
        help="Number of histogram bins along z (default: 200).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mc_slit",
        help="Output directory (default: outputs/mc_slit).",
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
        help="Print per-step progress.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine list of widths to simulate
    if args.widths is not None:
        widths = [float(x.strip()) for x in args.widths.split(",")]
    elif args.width is not None:
        widths = [float(args.width)]
    else:
        widths = [3.0]  # sensible default

    eta = args.eta

    print(f"\n{'='*60}")
    print(f"  Slit pore GCMC  η={eta:.4f}")
    print(f"  Widths: {widths}")
    print(f"  Lxy={args.Lxy}  n_equil={args.n_equil}  n_prod={args.n_prod}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    results = []

    for H in widths:
        print(f"Running slit pore: η={eta:.4f}  H={H:.2f}σ ...")
        t1 = time.perf_counter()

        z, rho, rho_bulk, beta_P = run_slit(
            eta=eta,
            H=H,
            Lxy=args.Lxy,
            n_equil=args.n_equil,
            n_prod=args.n_prod,
            n_bins=args.n_bins,
            seed=args.seed,
            verbose=args.verbose,
        )
        elapsed = time.perf_counter() - t1

        # Contact density check (at z = σ/2 and z = H - σ/2)
        sigma = 1.0
        idx_lo = int(np.argmin(np.abs(z - 0.5)))          # z = 0.5σ
        idx_hi = int(np.argmin(np.abs(z - (H / sigma - 0.5))))
        rho_contact_lo = float(rho[idx_lo])
        rho_contact_hi = float(rho[idx_hi])
        contact_ref = beta_P  # contact theorem: ρ(R) = βP

        print(
            f"  H={H:.2f}σ  ρ_contact(lo)={rho_contact_lo:.4f}  "
            f"ρ_contact(hi)={rho_contact_hi:.4f}  "
            f"βP_ref={contact_ref:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Save NPZ
        tag = f"eta{eta:.3f}_H{H:.2f}"
        out_npz = output_dir / f"slit_{tag}.npz"
        np.savez(
            out_npz,
            z=z,
            rho=rho,
            rho_bulk=rho_bulk,
            beta_P=beta_P,
            H=H,
            eta=eta,
        )
        print(f"  Saved: {out_npz}")

        # Plot individual profile
        out_png = output_dir / f"slit_{tag}.png"
        _plot_single(z, rho, rho_bulk, H, eta, beta_P, out_png)

        results.append({
            "H": H,
            "z": z,
            "rho": rho,
            "rho_bulk": rho_bulk,
            "beta_P": beta_P,
            "rho_contact_lo": rho_contact_lo,
            "rho_contact_hi": rho_contact_hi,
        })

    # If multiple widths, also save a combined figure
    if len(widths) > 1:
        out_png_multi = output_dir / f"slit_multi_eta{eta:.3f}.png"
        _plot_multi(results, eta, out_png_multi)

        out_npz_summary = output_dir / f"slit_summary_eta{eta:.3f}.npz"
        np.savez(
            out_npz_summary,
            eta=eta,
            widths=np.array(widths),
            rho_bulk=results[0]["rho_bulk"],
        )
        print(f"\n  Summary saved: {out_npz_summary}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
