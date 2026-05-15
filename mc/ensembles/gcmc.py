"""
Grand Canonical Monte Carlo for Hard Spheres at a Planar Hard Wall
===================================================================

The system: N hard spheres (σ=1) in a semi-infinite geometry.
  - Box: Lx × Ly × Lz where z is the wall-normal direction
  - Hard wall at z=0 (particles must have z > σ/2)
  - Periodic in x, y
  - Open in z: particles inserted/deleted to maintain chemical potential μ

Moves (chosen uniformly at random each "step"):
  1. Displacement: pick random particle, displace by δr (uniform cube),
     accept if no overlap and z > σ/2.  Identical to NVT hard-sphere MC.
  2. Insertion: place at uniformly random position in box, accept if
     no overlap and z > σ/2:
         acc = min(1, V · z_act / (N + 1))
     where z_act = ρ_bulk · exp(βμ_ex) is the activity.
  3. Deletion: pick random particle, remove with probability:
         acc = min(1, N / (V · z_act))

Activity derivation
-------------------
Setting the thermal de Broglie wavelength Λ = 1 and using kT = 1:
    βμ      = ln(ρ_bulk) + βμ_ex(η_bulk)          (CS EOS)
    z_act   = exp(βμ) = ρ_bulk · exp(βμ_ex)

Insertion acceptance (hard spheres, no soft energy):
    acc_ins = min(1, V / (N+1) · exp(βμ))
            = min(1, V · z_act / (N+1))            if no overlap and z > σ/2

Deletion acceptance:
    acc_del = min(1, (N+1) / V · exp(-βμ))         ← standard Metropolis
            = min(1, N / (V · z_act))               ← pick particle first

Reference
---------
Frenkel & Smit, "Understanding Molecular Simulation", 2nd ed., Ch. 5.
Panagiotopoulos, Mol. Phys. 61, 813 (1987).
Davidchack, Laird, Roth, Cond. Matt. Phys. (2016) — validation target.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Utility: hard-sphere overlap check (NumPy, O(N)) ──────────────────


def _has_overlap_with_existing(
    new_pos: np.ndarray,
    positions: np.ndarray,
    sigma2: float,
) -> bool:
    """Return True if *new_pos* overlaps any particle in *positions*.

    Parameters
    ----------
    new_pos : (3,) array
        Candidate position.
    positions : (N, 3) array
        Current particle positions. May be empty.
    sigma2 : float
        Squared hard-sphere diameter (σ²).

    Notes
    -----
    No periodic boundary conditions in z (wall geometry).
    PBC only in x and y are applied via minimum-image convention.
    For GCMC at a wall the box is large enough that z PBC is not needed
    (Lz ≫ σ, and particles are confined near the wall).  This simplifies
    the check significantly.
    """
    if len(positions) == 0:
        return False
    dr = positions - new_pos          # (N, 3)
    r2 = np.sum(dr * dr, axis=1)     # (N,)
    return bool(np.any(r2 < sigma2))


def _has_overlap_pbc(
    new_pos: np.ndarray,
    positions: np.ndarray,
    Lx: float,
    Ly: float,
    sigma2: float,
) -> bool:
    """Overlap check with PBC in x, y only (wall geometry).

    Parameters
    ----------
    new_pos : (3,)
    positions : (N, 3)
    Lx, Ly : float
        Box dimensions in periodic directions.
    sigma2 : float
    """
    if len(positions) == 0:
        return False
    dr = positions - new_pos          # (N, 3)
    # Minimum image in x, y only
    dr[:, 0] -= Lx * np.round(dr[:, 0] / Lx)
    dr[:, 1] -= Ly * np.round(dr[:, 1] / Ly)
    # No wrapping in z (semi-infinite / slab geometry)
    r2 = np.sum(dr * dr, axis=1)
    return bool(np.any(r2 < sigma2))


# ── GCMCSampler ────────────────────────────────────────────────────────


class GCMCSampler:
    """Grand Canonical Monte Carlo sampler for hard spheres at a hard wall.

    Parameters
    ----------
    Lx, Ly : float
        Box dimensions in the periodic (xy) directions.
    Lz : float
        Box depth in the wall-normal direction.  The wall is at z = 0;
        the open reservoir is effectively at z → ∞.
    mu_ex : float
        Excess chemical potential βμ_ex (dimensionless).  Use
        ``BulkThermodynamics.mu_ex_CS(eta)`` for the CS value.
    rho_bulk : float
        Target bulk number density ρ_bulk = 6η/(π σ³).
    sigma : float
        Hard-sphere diameter (default 1.0).
    seed : int
        NumPy random seed.

    Notes
    -----
    Uses pure NumPy (no JAX) because the particle number varies —
    JAX arrays have static shapes and are not convenient for GCMC.
    An O(N²) overlap test is used (sufficient for N ~ 300–500).
    Cell lists (Phase 3) will make this O(N).
    """

    def __init__(
        self,
        Lx: float,
        Ly: float,
        Lz: float,
        mu_ex: float,
        rho_bulk: float,
        sigma: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.mu_ex = float(mu_ex)
        self.rho_bulk = float(rho_bulk)
        self.sigma = float(sigma)
        self.sigma2 = sigma ** 2
        self.R = sigma / 2.0          # particle radius / wall exclusion depth

        # Volume of the simulation box
        self.V = Lx * Ly * Lz

        # Activity: z_act = ρ_bulk · exp(βμ_ex)
        # This encodes the reservoir chemical potential with Λ = 1.
        self.z_act = rho_bulk * np.exp(mu_ex)

        # RNG
        self.rng = np.random.default_rng(seed)

        # Initialise positions from a dilute random placement
        self._positions: List[np.ndarray] = []
        self._initialise()

        # Counters
        self.n_disp_acc = 0
        self.n_disp_att = 0
        self.n_ins_acc = 0
        self.n_ins_att = 0
        self.n_del_acc = 0
        self.n_del_att = 0

    # ── Initialisation ────────────────────────────────────────────────

    def _initialise(self) -> None:
        """Place an initial configuration at ~10 % of the target density."""
        N_target = max(10, int(0.10 * self.rho_bulk * self.V))
        N_target = min(N_target, 500)  # cap for safety during init
        placed: List[np.ndarray] = []
        max_attempts = N_target * 1000

        for _ in range(max_attempts):
            if len(placed) >= N_target:
                break
            candidate = np.array([
                self.rng.uniform(0.0, self.Lx),
                self.rng.uniform(0.0, self.Ly),
                self.rng.uniform(self.R, self.Lz),
            ])
            if len(placed) == 0:
                placed.append(candidate)
                continue
            arr = np.array(placed)
            if not _has_overlap_pbc(candidate, arr, self.Lx, self.Ly, self.sigma2):
                placed.append(candidate)

        self._positions = placed

    # ── Properties ────────────────────────────────────────────────────

    @property
    def N(self) -> int:
        """Current number of particles."""
        return len(self._positions)

    @property
    def positions(self) -> np.ndarray:
        """Particle positions as (N, 3) array (copy)."""
        if self.N == 0:
            return np.empty((0, 3), dtype=float)
        return np.array(self._positions)

    # ── Move implementations ──────────────────────────────────────────

    def _try_displace(self, max_disp: float) -> None:
        """Attempt one displacement move."""
        if self.N == 0:
            return
        self.n_disp_att += 1

        idx = int(self.rng.integers(0, self.N))
        old_pos = self._positions[idx].copy()

        delta = max_disp * (self.rng.random(3) - 0.5)
        new_pos = old_pos + delta
        # Wrap x, y only
        new_pos[0] %= self.Lx
        new_pos[1] %= self.Ly

        # Reject if particle enters the wall or leaves the box
        if new_pos[2] < self.R or new_pos[2] > self.Lz:
            return

        # Overlap check: build array without particle idx
        others = self._positions[:idx] + self._positions[idx + 1:]
        if others:
            arr = np.array(others)
            if _has_overlap_pbc(new_pos, arr, self.Lx, self.Ly, self.sigma2):
                return

        # Accept
        self._positions[idx] = new_pos
        self.n_disp_acc += 1

    def _try_insert(self) -> None:
        """Attempt one particle insertion."""
        self.n_ins_att += 1

        # Random position in box, respecting hard wall
        candidate = np.array([
            self.rng.uniform(0.0, self.Lx),
            self.rng.uniform(0.0, self.Ly),
            self.rng.uniform(self.R, self.Lz),
        ])

        # Overlap check with all existing particles
        if self.N > 0:
            arr = np.array(self._positions)
            if _has_overlap_pbc(candidate, arr, self.Lx, self.Ly, self.sigma2):
                return

        # Acceptance probability (no energy cost for hard spheres):
        #   acc = min(1, V_avail * z_act / (N + 1))
        # where V_avail = Lx * Ly * (Lz - R) (accessible volume above wall)
        V_avail = self.Lx * self.Ly * (self.Lz - self.R)
        acc = min(1.0, V_avail * self.z_act / (self.N + 1))

        if self.rng.random() < acc:
            self._positions.append(candidate)
            self.n_ins_acc += 1

    def _try_delete(self) -> None:
        """Attempt one particle deletion."""
        if self.N == 0:
            return
        self.n_del_att += 1

        idx = int(self.rng.integers(0, self.N))

        # Acceptance probability:
        #   acc = min(1, N / (V_avail * z_act))
        V_avail = self.Lx * self.Ly * (self.Lz - self.R)
        acc = min(1.0, self.N / (V_avail * self.z_act))

        if self.rng.random() < acc:
            del self._positions[idx]
            self.n_del_acc += 1

    # ── Main run loop ─────────────────────────────────────────────────

    def run(
        self,
        n_steps: int,
        n_equil: int,
        max_disp: float = 0.15,
        adjust_disp: bool = True,
        n_adjust: int = 500,
        verbose: bool = False,
        n_report: int = 10000,
    ) -> Dict:
        """Run GCMC for *n_equil* + *n_steps* MC steps.

        One "step" attempts one of {displace, insert, delete} chosen with
        equal probability (1/3 each).  This is the standard GCMC protocol.

        Parameters
        ----------
        n_steps : int
            Number of production steps (after equilibration).
        n_equil : int
            Number of equilibration steps (counters reset afterwards).
        max_disp : float
            Initial maximum displacement in each direction.
        adjust_disp : bool
            If True, tune max_disp to target 30–40 % displacement acceptance.
        n_adjust : int
            Adjust displacement every this many steps.
        verbose : bool
            Print progress.
        n_report : int
            Report interval when verbose=True.

        Returns
        -------
        dict with keys:
            'positions'     : final (N, 3) array
            'N_mean'        : mean particle count during production
            'N_history'     : array of N values sampled during production
            'acc_disp'      : displacement acceptance rate
            'acc_ins'       : insertion acceptance rate
            'acc_del'       : deletion acceptance rate
        """
        # ── Equilibration ──────────────────────────────────────────────
        if verbose:
            print(f"[GCMC] Equilibration: {n_equil} steps, N_init={self.N}")

        self._reset_counters()
        disp = float(max_disp)

        for step in range(n_equil):
            move = self.rng.integers(0, 3)
            if move == 0:
                self._try_displace(disp)
            elif move == 1:
                self._try_insert()
            else:
                self._try_delete()

            if adjust_disp and self.n_disp_att > 0 and (step + 1) % n_adjust == 0:
                rate = self.n_disp_acc / self.n_disp_att
                if rate > 0.40:
                    disp *= 1.05
                elif rate < 0.30:
                    disp *= 0.95
                disp = float(np.clip(disp, 0.01, min(self.Lx, self.Ly, self.Lz) / 2.0))
                self.n_disp_acc = 0
                self.n_disp_att = 0

            if verbose and (step + 1) % n_report == 0:
                acc_d = (self.n_disp_acc / max(self.n_disp_att, 1))
                acc_i = (self.n_ins_acc / max(self.n_ins_att, 1))
                acc_del = (self.n_del_acc / max(self.n_del_att, 1))
                rho_cur = self.N / self.V
                print(
                    f"  equil step {step+1:7d}/{n_equil}  "
                    f"N={self.N:4d}  ρ={rho_cur:.4f}  "
                    f"acc_d={acc_d:.3f}  acc_i={acc_i:.3f}  acc_del={acc_del:.3f}  "
                    f"disp={disp:.4f}"
                )

        # Reset counters for production
        self._reset_counters()

        # ── Production ─────────────────────────────────────────────────
        if verbose:
            print(f"[GCMC] Production: {n_steps} steps, N={self.N}")

        N_history: List[int] = []

        for step in range(n_steps):
            move = self.rng.integers(0, 3)
            if move == 0:
                self._try_displace(disp)
            elif move == 1:
                self._try_insert()
            else:
                self._try_delete()

            N_history.append(self.N)

            if verbose and (step + 1) % n_report == 0:
                acc_d = self.n_disp_acc / max(self.n_disp_att, 1)
                acc_i = self.n_ins_acc / max(self.n_ins_att, 1)
                acc_del = self.n_del_acc / max(self.n_del_att, 1)
                rho_cur = self.N / self.V
                print(
                    f"  prod  step {step+1:7d}/{n_steps}  "
                    f"N={self.N:4d}  ρ={rho_cur:.4f}  "
                    f"acc_d={acc_d:.3f}  acc_i={acc_i:.3f}  acc_del={acc_del:.3f}"
                )

        N_arr = np.array(N_history, dtype=float)

        return {
            "positions": self.positions,
            "N_mean": float(np.mean(N_arr)),
            "N_history": N_arr,
            "acc_disp": self.n_disp_acc / max(self.n_disp_att, 1),
            "acc_ins": self.n_ins_acc / max(self.n_ins_att, 1),
            "acc_del": self.n_del_acc / max(self.n_del_att, 1),
            "max_disp_final": disp,
        }

    # ── Density profile ───────────────────────────────────────────────

    def get_density_profile(
        self,
        positions: np.ndarray,
        n_bins: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ρ(z) from a single snapshot.

        Parameters
        ----------
        positions : (N, 3)
            Particle positions.
        n_bins : int
            Number of histogram bins along z.

        Returns
        -------
        z_centers : (n_bins,)
            Bin centres in units of σ.
        rho : (n_bins,)
            Number density profile ρ(z) in units of σ⁻³.
        """
        dz = self.Lz / n_bins
        z_edges = np.linspace(0.0, self.Lz, n_bins + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

        if len(positions) == 0:
            return z_centers / self.sigma, np.zeros(n_bins)

        counts, _ = np.histogram(positions[:, 2], bins=z_edges)
        # Normalise: ρ(z) = <N in slab> / (Lx * Ly * dz)
        rho = counts / (self.Lx * self.Ly * dz)

        return z_centers / self.sigma, rho * self.sigma ** 3

    # ── Helpers ───────────────────────────────────────────────────────

    def _reset_counters(self) -> None:
        self.n_disp_acc = 0
        self.n_disp_att = 0
        self.n_ins_acc = 0
        self.n_ins_att = 0
        self.n_del_acc = 0
        self.n_del_att = 0

    def acceptance_rates(self) -> Dict[str, float]:
        """Return current acceptance rates as a dict."""
        return {
            "displacement": self.n_disp_acc / max(self.n_disp_att, 1),
            "insertion": self.n_ins_acc / max(self.n_ins_att, 1),
            "deletion": self.n_del_acc / max(self.n_del_att, 1),
        }
