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


# ── Species-resolved g_αβ(r) for binary mixtures ─────────────────────


class MixtureRDFAccumulator:
    """Accumulate species-resolved radial distribution functions g_αβ(r).

    For a binary mixture with species labels 0 and 1, computes the three
    independent partial RDFs:
        g₀₀(r) — between particles of species 0
        g₀₁(r) — cross correlation (species 0 with species 1)
        g₁₁(r) — between particles of species 1

    The normalisation uses the partial number densities ρ_α = N_α / V:

        g_αβ(r) = V / (N_α N_β) * <Σᵢ∈α Σⱼ∈β,j≠i δ(r - rᵢⱼ)> / (4πr²Δr)

    For the cross correlation (α ≠ β), both (i∈α, j∈β) and (i∈β, j∈α)
    directions are counted and the result is divided by 2 (i.e., each
    pair is counted once).

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    r_max : float
        Maximum separation to accumulate (should be ≤ L/2).
    n_species : int
        Number of species (currently only 2 is supported).
    """

    def __init__(
        self,
        n_bins: int = 200,
        r_max: float = 5.0,
        n_species: int = 2,
    ) -> None:
        if n_species != 2:
            raise ValueError("MixtureRDFAccumulator currently supports exactly 2 species.")
        self.n_bins = n_bins
        self.r_max = float(r_max)
        self.n_species = n_species
        self.bins = np.linspace(0.0, r_max, n_bins + 1)

        # One histogram per species pair (α ≤ β)
        self._hist = {
            (0, 0): np.zeros(n_bins, dtype=np.float64),
            (0, 1): np.zeros(n_bins, dtype=np.float64),
            (1, 1): np.zeros(n_bins, dtype=np.float64),
        }
        self._n_frames = 0
        # Running sums of partial counts (to compute mean ρ_α)
        self._N_alpha_sum = np.zeros(n_species, dtype=np.float64)
        self._box_length: float = 0.0

    # ── Accumulation ──────────────────────────────────────────────────

    def update(
        self,
        positions: np.ndarray,
        species: np.ndarray,
        box_length: float,
    ) -> None:
        """Add one configuration to the running histograms.

        Parameters
        ----------
        positions : array-like, shape (N, 3)
            Particle positions (NumPy or JAX array).
        species : array-like, shape (N,) of int
            Species labels for each particle (0 or 1).
        box_length : float
            Cubic box side length.
        """
        positions = np.asarray(positions, dtype=float)
        species = np.asarray(species, dtype=int)
        N = len(positions)
        self._box_length = float(box_length)
        self._n_frames += 1

        # Partial counts
        mask0 = (species == 0)
        mask1 = (species == 1)
        N0 = int(np.sum(mask0))
        N1 = int(np.sum(mask1))
        self._N_alpha_sum[0] += N0
        self._N_alpha_sum[1] += N1

        pos0 = positions[mask0]
        pos1 = positions[mask1]

        # g₀₀: all pairs within species 0
        self._accumulate_pair(pos0, pos0, (0, 0), box_length, exclude_self=True)

        # g₁₁: all pairs within species 1
        self._accumulate_pair(pos1, pos1, (1, 1), box_length, exclude_self=True)

        # g₀₁: cross-species pairs (order doesn't matter; count i<j only)
        self._accumulate_pair(pos0, pos1, (0, 1), box_length, exclude_self=False)

    def _accumulate_pair(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        pair: Tuple,
        box_length: float,
        exclude_self: bool,
    ) -> None:
        """Accumulate distances between particles in pos_a and pos_b.

        If ``exclude_self=True`` (same-species pairs, pos_a is pos_b),
        only i < j pairs are counted to avoid double counting and
        self-interactions.
        """
        Na = len(pos_a)
        Nb = len(pos_b)
        if Na == 0 or Nb == 0:
            return

        hist = self._hist[pair]

        if exclude_self:
            # Same-species: count only i < j
            for i in range(Na - 1):
                dr = pos_b[i + 1:] - pos_a[i]          # (Nb-i-1, 3)
                dr = dr - box_length * np.round(dr / box_length)
                r = np.sqrt(np.sum(dr ** 2, axis=1))
                counts, _ = np.histogram(r, bins=self.bins)
                hist += counts
        else:
            # Cross-species: all (i, j) pairs with i∈A, j∈B
            for i in range(Na):
                dr = pos_b - pos_a[i]                   # (Nb, 3)
                dr = dr - box_length * np.round(dr / box_length)
                r = np.sqrt(np.sum(dr ** 2, axis=1))
                counts, _ = np.histogram(r, bins=self.bins)
                hist += counts

    # ── Retrieval ─────────────────────────────────────────────────────

    def get_gij(self, alpha: int, beta: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r, g_αβ(r)) for species pair (alpha, beta).

        Parameters
        ----------
        alpha : int
            Species index (0 or 1).
        beta : int
            Species index (0 or 1).

        Returns
        -------
        r : np.ndarray, shape (n_bins,)
            Bin centres.
        g_ab : np.ndarray, shape (n_bins,)
            Partial radial distribution function g_αβ(r).
        """
        if self._n_frames == 0:
            raise RuntimeError("No frames accumulated yet.")

        alpha, beta = int(alpha), int(beta)
        pair = (min(alpha, beta), max(alpha, beta))

        raw = self._hist[pair].copy()
        r_centres = 0.5 * (self.bins[:-1] + self.bins[1:])

        V = self._box_length ** 3
        N_alpha = self._N_alpha_sum[alpha] / self._n_frames   # mean N_alpha
        N_beta = self._N_alpha_sum[beta] / self._n_frames     # mean N_beta

        if N_alpha == 0 or N_beta == 0:
            return r_centres, np.zeros(self.n_bins)

        # Shell volume (exact spherical shell)
        shell_vol = (4.0 / 3.0) * np.pi * (self.bins[1:] ** 3 - self.bins[:-1] ** 3)

        if alpha == beta:
            # Same-species: i<j pairs counted; effective N*(N-1)/2 normalization
            # Histogram has (per frame) Σ_{i<j} count → normalize by N_alpha*(N_alpha-1)/2
            # Standard route: norm = n_frames * (N_alpha/2) * (N_alpha*rho_alpha) * shell_vol
            # which simplifies to n_frames * N_alpha*(N_alpha-1)/(2*V) * shell_vol
            rho_alpha = N_alpha / V
            norm = self._n_frames * (N_alpha / 2.0) * rho_alpha * shell_vol
        else:
            # Cross-species: all (i,j) pairs counted for i∈A, j∈B
            # norm = n_frames * N_alpha * rho_beta * shell_vol
            rho_beta = N_beta / V
            norm = self._n_frames * N_alpha * rho_beta * shell_vol

        with np.errstate(divide="ignore", invalid="ignore"):
            g_ab = np.where(norm > 0, raw / norm, 0.0)

        return r_centres, g_ab

    def get_all(self) -> dict:
        """Return all three partial g_αβ(r) functions.

        Returns
        -------
        dict with keys '00', '01', '11', each mapping to (r, g_ab).
        """
        r00, g00 = self.get_gij(0, 0)
        r01, g01 = self.get_gij(0, 1)
        r11, g11 = self.get_gij(1, 1)
        return {
            "r": r00,       # shared r array
            "g00": g00,
            "g01": g01,
            "g11": g11,
        }

    def reset(self) -> None:
        """Clear all accumulated data."""
        for key in self._hist:
            self._hist[key][:] = 0.0
        self._N_alpha_sum[:] = 0.0
        self._n_frames = 0

    @property
    def n_frames(self) -> int:
        """Number of accumulated configurations."""
        return self._n_frames

    @property
    def r(self) -> np.ndarray:
        """Bin centres."""
        return 0.5 * (self.bins[:-1] + self.bins[1:])
