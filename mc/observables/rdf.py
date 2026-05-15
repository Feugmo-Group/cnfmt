"""
Radial Distribution Function g(r)
==================================

Accumulates pairwise distances into a histogram and normalises by the
ideal-gas pair count in each spherical shell:

    g(r) = histogram(r; r+dr) / (n_frames * N * rho * 4*pi*r^2 * dr)

where *rho* is the number density N/V.

The histogram accumulation is done in **NumPy** (outside JAX) for
simplicity; each call is O(N²) so N should be ≲ 2000 for fast runs.

Usage
-----
    acc = RDFAccumulator(n_bins=200, r_max=5.0)
    for snapshot in production_run:
        acc.update(snapshot.positions, snapshot.box_length)
    r, g = acc.get()
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import Tuple


# ── Low-level functions ───────────────────────────────────────────────


def accumulate_rdf(
    positions: np.ndarray,
    box_length: float,
    bins: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Accumulate pairwise distances into a histogram.

    Uses minimum-image convention and counts only r ≤ r_max.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Particle positions (NumPy, not JAX).
    box_length : float
    bins : np.ndarray, shape (n_bins+1,)
        Bin edges from 0 to r_max.
    sigma : float
        Hard-sphere diameter (unused here, kept for API consistency).

    Returns
    -------
    histogram : np.ndarray, shape (n_bins,)
        Raw pair counts in each bin.
    """
    positions = np.asarray(positions)
    N = len(positions)
    histogram = np.zeros(len(bins) - 1, dtype=np.float64)

    for i in range(N - 1):
        dr = positions[i + 1:] - positions[i]          # (N-i-1, 3)
        dr = dr - box_length * np.round(dr / box_length)
        r = np.sqrt(np.sum(dr ** 2, axis=1))            # (N-i-1,)
        counts, _ = np.histogram(r, bins=bins)
        histogram += counts

    # Each pair counted once above; g(r) formula needs N*(N-1)/2 pairs
    # but the normalization below handles this via rho
    return histogram


def normalize_rdf(
    histogram: np.ndarray,
    n_frames: int,
    n_particles: int,
    density: float,
    bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise raw histogram to g(r).

    Normalisation:
        g(r) = histogram / (n_frames * N * rho * 4*pi*r^2 * dr)

    This gives the correct g(r) = 1 for an ideal gas.

    Parameters
    ----------
    histogram : np.ndarray, shape (n_bins,)
        Raw pair counts (each pair counted once, i.e. i < j only).
    n_frames : int
        Number of accumulated configurations.
    n_particles : int
        Number of particles N.
    density : float
        Number density ρ = N/V.
    bins : np.ndarray, shape (n_bins+1,)
        Bin edges.

    Returns
    -------
    r : np.ndarray, shape (n_bins,)
        Bin centres.
    g : np.ndarray, shape (n_bins,)
        Radial distribution function.
    """
    r = 0.5 * (bins[:-1] + bins[1:])     # bin centres
    dr = bins[1:] - bins[:-1]            # bin widths

    # Shell volume (exact spherical shell)
    shell_volume = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    # Ideal number of pairs in each shell for ONE reference particle:
    #   ρ * shell_volume
    # Summed over N reference particles and n_frames snapshots:
    #   n_frames * N * rho * shell_volume
    # We counted i < j pairs only, so histogram has N*(N-1)/2 expected
    # pairs per frame, but the formula below handles that:
    # histogram(i<j) counts each pair once → normalise by N/2 reference
    # particles effectively (standard route: divide by N_frames * N/2 * rho * V_shell)
    norm = n_frames * (n_particles / 2.0) * density * shell_volume

    # Guard against empty bins
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(norm > 0, histogram / norm, 0.0)

    return r, g


# ── Accumulator class ─────────────────────────────────────────────────


class RDFAccumulator:
    """Accumulate g(r) over many MC snapshots.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    r_max : float
        Maximum distance to accumulate (should be ≤ L/2).
    sigma : float
        Hard-sphere diameter (for reference, not used in accumulation).
    """

    def __init__(self, n_bins: int = 200, r_max: float = 5.0, sigma: float = 1.0):
        self.n_bins = n_bins
        self.r_max = float(r_max)
        self.sigma = float(sigma)
        self.bins = np.linspace(0.0, r_max, n_bins + 1)
        self._histogram = np.zeros(n_bins, dtype=np.float64)
        self._n_frames = 0
        self._n_particles: int = 0
        self._box_length: float = 0.0

    def update(self, positions, box_length: float) -> None:
        """Add one configuration to the running histogram.

        Parameters
        ----------
        positions : array-like, shape (N, 3)
            Can be JAX or NumPy array.
        box_length : float
        """
        positions = np.asarray(positions)
        N = len(positions)
        self._n_particles = N
        self._box_length = float(box_length)

        hist = accumulate_rdf(positions, box_length, self.bins, self.sigma)
        self._histogram += hist
        self._n_frames += 1

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalised (r, g(r)).

        Returns
        -------
        r : np.ndarray
        g : np.ndarray
        """
        if self._n_frames == 0:
            raise RuntimeError("No frames accumulated yet.")

        density = self._n_particles / self._box_length ** 3
        r, g = normalize_rdf(
            self._histogram,
            self._n_frames,
            self._n_particles,
            density,
            self.bins,
        )
        return r, g

    def reset(self) -> None:
        """Clear accumulated data."""
        self._histogram[:] = 0.0
        self._n_frames = 0

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def r(self) -> np.ndarray:
        """Bin centres."""
        return 0.5 * (self.bins[:-1] + self.bins[1:])
