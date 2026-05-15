"""
Density Profile Accumulator
============================

Accumulates ρ(z) density profiles from GCMC (or NVT slab) snapshots.

Physics
-------
For a planar slab geometry with a hard wall at z = 0:

    ρ(z) = <N(z, z+Δz)> / (Lx · Ly · Δz)

The bin-averaged density is in units of σ⁻³ (with σ as the hard-sphere
diameter).  The normalised profile g(z) = ρ(z)/ρ_bulk oscillates around 1
far from the wall.

Contact theorem (exact for hard spheres at a hard wall):
    ρ(z = σ/2) = βP = ρ_bulk · Z_CS(η)

This provides a stringent validation of any MC simulation or DFT solver.

Output format
-------------
The ``get()`` method returns (z_centers, rho, rho_bulk_far) in a format
compatible with ``solvers/wall_profile.py``'s ``get_mc_data()`` which
returns dicts with keys 'z', 'rho', 'rho_bulk'.

  z            : bin centres in units of σ, from 0 to Lz/σ
  rho          : <ρ(z)> in units of σ⁻³
  rho_bulk_far : mean ρ in the outer half of the box (bulk estimate)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class DensityProfileAccumulator:
    """Accumulate ρ(z) profiles over multiple GCMC frames.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins along z.
    Lz : float
        Box length in the wall-normal direction (in units of σ).
    Lx : float
        Box length in x (in units of σ).
    Ly : float
        Box length in y (in units of σ).
    sigma : float
        Hard-sphere diameter (default 1.0).

    Notes
    -----
    Positions passed to ``update()`` must be in the same length units as
    Lx, Ly, Lz, sigma (typically all in σ = 1 units).
    """

    def __init__(
        self,
        n_bins: int,
        Lz: float,
        Lx: float,
        Ly: float,
        sigma: float = 1.0,
    ) -> None:
        self.n_bins = int(n_bins)
        self.Lz = float(Lz)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.sigma = float(sigma)

        self.dz = Lz / n_bins
        self.z_edges = np.linspace(0.0, Lz, n_bins + 1)
        # Bin centres in units of σ
        self.z_centers = 0.5 * (self.z_edges[:-1] + self.z_edges[1:]) / sigma

        # Accumulate counts and frames
        self._counts = np.zeros(n_bins, dtype=float)
        self._n_frames = 0

    # ── Core methods ──────────────────────────────────────────────────

    def update(self, positions: np.ndarray) -> None:
        """Add one configuration frame to the accumulator.

        Parameters
        ----------
        positions : (N, 3) array, or empty (0, 3)
            Particle positions in simulation-box units.
            Only the z-component (column 2) is used.
        """
        self._n_frames += 1
        if len(positions) == 0:
            return
        counts, _ = np.histogram(positions[:, 2], bins=self.z_edges)
        self._counts += counts

    def reset(self) -> None:
        """Clear accumulated data."""
        self._counts[:] = 0.0
        self._n_frames = 0

    # ── Results ───────────────────────────────────────────────────────

    def get(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return the accumulated density profile.

        Returns
        -------
        z_centers : (n_bins,)
            Bin centres in units of σ.
        rho_profile : (n_bins,)
            Mean number density <ρ(z)> in units of σ⁻³.
        rho_bulk_far : float
            Estimated bulk density from the outer half of the box
            (far from the wall), in units of σ⁻³.

        Notes
        -----
        If no frames have been accumulated, returns zeros with rho_bulk_far=0.
        """
        if self._n_frames == 0:
            return self.z_centers.copy(), np.zeros(self.n_bins), 0.0

        # Mean counts per bin → convert to density
        mean_counts = self._counts / self._n_frames
        rho = mean_counts / (self.Lx * self.Ly * self.dz)  # units: length⁻³

        # Convert to σ⁻³
        rho_sigma3 = rho * self.sigma ** 3

        # Estimate bulk density from outer half (far from wall)
        half_idx = self.n_bins // 2
        rho_bulk_far = float(np.mean(rho_sigma3[half_idx:]))

        return self.z_centers.copy(), rho_sigma3, rho_bulk_far

    def normalize(self, rho_bulk: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return ρ(z)/ρ_bulk (the reduced density profile).

        Parameters
        ----------
        rho_bulk : float, optional
            Bulk density for normalisation.  If None, uses ``rho_bulk_far``
            from ``get()`` (the measured bulk from the outer half of the box).

        Returns
        -------
        z_centers : (n_bins,)
            Bin centres in units of σ.
        g_z : (n_bins,)
            Normalised profile ρ(z)/ρ_bulk.
        """
        z, rho, rho_bulk_far = self.get()
        if rho_bulk is None:
            rho_bulk = rho_bulk_far
        if rho_bulk <= 0.0:
            return z, np.zeros_like(rho)
        return z, rho / rho_bulk

    def contact_density(self, sigma: Optional[float] = None) -> Tuple[float, float]:
        """Extract contact density ρ(z = σ/2).

        The contact bin is the first bin whose centre lies at or just above
        z = σ/2 (= R, the particle radius).

        Parameters
        ----------
        sigma : float, optional
            Hard-sphere diameter.  Defaults to ``self.sigma``.

        Returns
        -------
        rho_contact : float
            Density at the first bin above the wall (σ⁻³).
        z_contact : float
            z-position of that bin centre (in units of σ).
        """
        if sigma is None:
            sigma = self.sigma
        R = sigma / 2.0
        z, rho, _ = self.get()
        # Find bin closest to z = R (in units of sigma, R/sigma = 0.5)
        idx = int(np.argmin(np.abs(z - 0.5)))
        return float(rho[idx]), float(z[idx])

    def to_dict(self, rho_bulk: Optional[float] = None) -> dict:
        """Return results as a dict compatible with ``get_mc_data()`` format.

        Compatible with ``solvers/wall_profile.py::get_mc_data()`` which
        returns ``{'z': ..., 'rho': ..., 'rho_bulk': ...}``.

        Parameters
        ----------
        rho_bulk : float, optional
            If provided, used as the reported bulk density.  Otherwise the
            measured outer-half average is used.

        Returns
        -------
        dict with keys:
            'z'        : bin centres in units of σ
            'rho'      : mean density ρ(z) in units of σ⁻³
            'rho_bulk' : bulk density in units of σ⁻³
            'n_frames' : number of frames accumulated
        """
        z, rho, rho_bulk_far = self.get()
        if rho_bulk is None:
            rho_bulk = rho_bulk_far
        return {
            "z": z,
            "rho": rho,
            "rho_bulk": float(rho_bulk),
            "n_frames": self._n_frames,
        }

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        """Number of accumulated frames."""
        return self._n_frames

    def __repr__(self) -> str:
        return (
            f"DensityProfileAccumulator("
            f"n_bins={self.n_bins}, Lz={self.Lz:.2f}σ, "
            f"n_frames={self._n_frames})"
        )
