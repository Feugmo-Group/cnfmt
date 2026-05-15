"""
Hard-Sphere Potentials
======================

Pure JAX implementations of the hard-sphere pair potential.

All functions use the **minimum-image convention**:
    dr_mic = dr - L * round(dr / L)

JIT-compiled where useful.  Single-particle overlap check is the
hot path for NVT MC; it is written to be easily JIT-compiled with
static `i`.

Physics
-------
u_HS(r) = ∞  if r < σ  (overlap)
u_HS(r) = 0  if r ≥ σ  (no interaction)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array

# Enable double precision
jax.config.update("jax_enable_x64", True)


# ── Scalar overlap energy ──────────────────────────────────────────────


def overlap_energy(r2: float, sigma: float = 1.0) -> float:
    """Energy for a single pair at squared distance *r2*.

    Returns ``jnp.inf`` if the particles overlap (r² < σ²), else 0.

    Parameters
    ----------
    r2 : float
        Squared centre-to-centre distance.
    sigma : float
        Hard-sphere diameter.
    """
    return jnp.where(r2 < sigma * sigma, jnp.inf, 0.0)


# ── All-pairs overlap (validation / initialisation) ───────────────────


@jax.jit
def pairwise_overlap(
    positions: Array,
    box_length: float,
    sigma: float = 1.0,
) -> bool:
    """Check whether *any* pair of particles overlaps.

    O(N²) — intended for validation and initialisation checks only.

    Parameters
    ----------
    positions : Array, shape (N, 3)
    box_length : float
    sigma : float

    Returns
    -------
    bool
        ``True`` if at least one overlap exists.
    """
    sigma2 = sigma * sigma

    def _check_row(i):
        dr = positions - positions[i]              # (N, 3)
        dr = dr - box_length * jnp.round(dr / box_length)
        r2 = jnp.sum(dr ** 2, axis=1)             # (N,)
        # Exclude self (set self distance to a large value)
        r2 = r2.at[i].set(sigma2 + 1.0)
        return jnp.any(r2 < sigma2)

    # vmap over all particle indices, then OR the results
    rows = jax.vmap(_check_row)(jnp.arange(len(positions)))
    return jnp.any(rows)


# ── Single-particle overlap (hot path for NVT MC) ─────────────────────


@jax.jit
def single_particle_overlap(
    pos_i: Array,
    positions: Array,
    i: int,
    box_length: float,
    sigma: float = 1.0,
) -> bool:
    """Check whether a **trial** position *pos_i* overlaps any other particle.

    Excludes particle *i* itself (the particle being displaced).

    Parameters
    ----------
    pos_i : Array, shape (3,)
        Trial position of particle *i*.
    positions : Array, shape (N, 3)
        Current positions of all particles.
    i : int
        Index of the particle being moved (excluded from overlap check).
    box_length : float
    sigma : float

    Returns
    -------
    bool
        ``True`` if *pos_i* overlaps any particle j ≠ i.
    """
    sigma2 = sigma * sigma
    dr = positions - pos_i                          # (N, 3)
    dr = dr - box_length * jnp.round(dr / box_length)
    r2 = jnp.sum(dr ** 2, axis=1)                 # (N,)
    # Mask out self
    r2 = r2.at[i].set(sigma2 + 1.0)
    return jnp.any(r2 < sigma2)


# ── HardSphere convenience class ──────────────────────────────────────


class HardSphere:
    """Hard-sphere potential wrapper.

    Exposes ``overlap``, ``pairwise_overlap``, and
    ``single_particle_overlap`` as instance methods, carrying the
    diameter *sigma* so callers don't need to pass it repeatedly.

    Parameters
    ----------
    sigma : float
        Hard-sphere diameter (default 1.0).
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)

    def overlap(self, r2: float) -> bool:
        """Single pair overlap test at squared distance *r2*."""
        return overlap_energy(r2, self.sigma) == jnp.inf

    def pairwise_overlap(self, positions: Array, box_length: float) -> bool:
        """Check all pairs for overlap."""
        return pairwise_overlap(positions, box_length, self.sigma)

    def single_particle_overlap(
        self,
        pos_i: Array,
        positions: Array,
        i: int,
        box_length: float,
    ) -> bool:
        """Check trial position *pos_i* against all particles except *i*."""
        return single_particle_overlap(pos_i, positions, i, box_length, self.sigma)


# ── Lennard-Jones potential (NumPy, O(N²)) ────────────────────────────


class LennardJones:
    """Lennard-Jones 12-6 pair potential with cutoff and energy shift.

    u(r) = 4ε[(σ/r)¹² - (σ/r)⁶] - u(r_cut)   for r < r_cut
    u(r) = 0                                      for r ≥ r_cut

    The potential is shifted so u(r_cut) = 0 (no energy discontinuity).

    Long-range tail correction (beyond r_cut):
        U_tail = (8π/3) N ρ ε [ (1/3)(σ/r_cut)⁹ - (σ/r_cut)³ ]

    Parameters
    ----------
    epsilon : float
        Well depth (reduced units: ε* = 1.0).
    sigma : float
        Particle diameter (reduced units: σ* = 1.0).
    r_cut : float
        Cutoff distance in units of σ (default 2.5σ).

    Notes
    -----
    Uses pure NumPy throughout (no JAX) because positions can be
    Python lists or NumPy arrays with variable length during GCMC.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        r_cut: float = 2.5,
    ) -> None:
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.r_cut = float(r_cut)      # in units of sigma
        self.r_cut_abs = self.r_cut * self.sigma  # absolute cutoff distance

        # Precompute energy shift: u_shift = 4ε[(σ/r_cut)¹² - (σ/r_cut)⁶]
        inv_rc = self.sigma / self.r_cut_abs
        inv_rc6 = inv_rc ** 6
        inv_rc12 = inv_rc6 ** 2
        self._u_shift = 4.0 * self.epsilon * (inv_rc12 - inv_rc6)

    # ── Single-pair energy ────────────────────────────────────────────

    def energy(self, r2: float) -> float:
        """LJ pair energy at squared distance *r2*.

        Parameters
        ----------
        r2 : float
            Squared centre-to-centre distance (in σ units if sigma=1).

        Returns
        -------
        float
            Pair energy.  Returns 0 if r² ≥ r_cut².
        """
        rc2 = self.r_cut_abs ** 2
        if r2 >= rc2:
            return 0.0
        sigma2 = self.sigma ** 2
        inv_r2 = sigma2 / r2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        return 4.0 * self.epsilon * (inv_r12 - inv_r6) - self._u_shift

    # ── All-pairs energy O(N²) ────────────────────────────────────────

    def pair_energy(
        self,
        positions: np.ndarray,
        box_length: float,
    ) -> float:
        """Total potential energy for all pairs (minimum-image convention).

        O(N²) — suitable for N ≲ 500.  Does NOT include the tail correction.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
        box_length : float
            Cubic box side length.

        Returns
        -------
        float
            Total potential energy U (in units of ε).
        """
        positions = np.asarray(positions, dtype=float)
        N = len(positions)
        rc2 = self.r_cut_abs ** 2
        sigma2 = self.sigma ** 2
        u_total = 0.0

        for i in range(N - 1):
            dr = positions[i + 1:] - positions[i]           # (N-i-1, 3)
            dr -= box_length * np.round(dr / box_length)     # min image
            r2 = np.sum(dr ** 2, axis=1)                    # (N-i-1,)
            mask = r2 < rc2
            if not np.any(mask):
                continue
            inv_r2 = sigma2 / r2[mask]
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2
            u_total += np.sum(
                4.0 * self.epsilon * (inv_r12 - inv_r6) - self._u_shift
            )

        return float(u_total)

    # ── Single-particle energy (hot path for MC) ──────────────────────

    def single_particle_energy(
        self,
        pos_i: np.ndarray,
        positions: np.ndarray,
        i: int,
        box_length: float,
    ) -> float:
        """Energy of particle *i* at trial position *pos_i* with all others.

        Excludes the self-interaction (particle *i* in *positions*).

        Parameters
        ----------
        pos_i : np.ndarray, shape (3,)
            Trial position of particle *i*.
        positions : np.ndarray, shape (N, 3)
            Current positions of all particles.
        i : int
            Index of the particle being moved (excluded from sum).
        box_length : float

        Returns
        -------
        float
            Sum of pair energies between pos_i and all j ≠ i.
        """
        positions = np.asarray(positions, dtype=float)
        pos_i = np.asarray(pos_i, dtype=float)
        N = len(positions)
        rc2 = self.r_cut_abs ** 2
        sigma2 = self.sigma ** 2

        # Build array of all positions except i
        dr = positions - pos_i                               # (N, 3)
        dr -= box_length * np.round(dr / box_length)         # min image
        r2 = np.sum(dr ** 2, axis=1)                        # (N,)

        # Mask: within cutoff and not self
        mask = (r2 < rc2)
        mask[i] = False

        if not np.any(mask):
            return 0.0

        inv_r2 = sigma2 / r2[mask]
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        return float(np.sum(
            4.0 * self.epsilon * (inv_r12 - inv_r6) - self._u_shift
        ))

    # ── Long-range tail corrections ───────────────────────────────────

    def tail_correction_energy(
        self,
        n_particles: int,
        density: float,
    ) -> float:
        """Long-range energy correction beyond r_cut.

        U_tail = (8π/3) N ρ ε [ (1/3)(σ/r_cut)⁹ - (σ/r_cut)³ ]

        Parameters
        ----------
        n_particles : int
        density : float
            Number density ρ = N/V.

        Returns
        -------
        float
            Energy correction in units of ε.
        """
        inv_rc = self.sigma / self.r_cut_abs
        inv_rc3 = inv_rc ** 3
        inv_rc9 = inv_rc3 ** 3
        return (
            (8.0 * np.pi / 3.0)
            * n_particles
            * density
            * self.epsilon
            * ((1.0 / 3.0) * inv_rc9 - inv_rc3)
        )

    def tail_correction_pressure(self, density: float) -> float:
        """Long-range pressure correction beyond r_cut.

        P_tail = (16π/3) ρ² ε [ (2/3)(σ/r_cut)⁹ - (σ/r_cut)³ ]

        Parameters
        ----------
        density : float

        Returns
        -------
        float
            Pressure correction (same units as ρ kT).
        """
        inv_rc = self.sigma / self.r_cut_abs
        inv_rc3 = inv_rc ** 3
        inv_rc9 = inv_rc3 ** 3
        return (
            (16.0 * np.pi / 3.0)
            * density ** 2
            * self.epsilon
            * ((2.0 / 3.0) * inv_rc9 - inv_rc3)
        )


# ── Binary hard-sphere mixture ───────────────────────────────────────


class HardSphereMixture:
    """Binary hard-sphere mixture with additive (Lorentz-Berthelot) mixing rule.

    Two species with diameters σ₁ and σ₂.  The cross-interaction diameter
    follows the additive (Lorentz) rule:

        σ₁₂ = (σ₁ + σ₂) / 2

    Species labels are integers: 0 → diameter σ₁, 1 → diameter σ₂.

    Parameters
    ----------
    sigma1 : float
        Diameter of species 0 (default 1.0).
    sigma2 : float
        Diameter of species 1 (default 1.2).
    x1 : float
        Mole fraction of species 0 (informational only; not used in
        overlap checks).

    Notes
    -----
    Uses pure NumPy (no JAX) so it can be used inside GCMC or Gibbs
    ensemble loops where particle counts change at runtime.
    """

    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 1.2,
        x1: float = 0.5,
    ) -> None:
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.x1 = float(x1)

        # Precompute squared cross-interaction diameters for the three pairs
        # (species_i, species_j) → σ_ij
        self._sigma = {
            (0, 0): sigma1,
            (1, 1): sigma2,
            (0, 1): (sigma1 + sigma2) / 2.0,
            (1, 0): (sigma1 + sigma2) / 2.0,
        }
        self._sigma2 = {k: v * v for k, v in self._sigma.items()}

    def sigma_ij(self, species_i: int, species_j: int) -> float:
        """Cross-interaction diameter for species pair (i, j).

        Parameters
        ----------
        species_i : int
            Species of particle i (0 or 1).
        species_j : int
            Species of particle j (0 or 1).

        Returns
        -------
        float
            Diameter σ_ij.
        """
        return self._sigma[(int(species_i), int(species_j))]

    def overlap(
        self,
        pos_i: np.ndarray,
        species_i: int,
        positions: np.ndarray,
        species: np.ndarray,
        i: int,
        box_length: float,
    ) -> bool:
        """Check whether particle *i* at trial position *pos_i* overlaps any other.

        Uses the species-dependent diameter σ_ij for each pair.
        Particle *i* itself is excluded from the check.

        Parameters
        ----------
        pos_i : np.ndarray, shape (3,)
            Trial position of the particle being moved (or inserted).
        species_i : int
            Species label of the particle (0 or 1).
        positions : np.ndarray, shape (N, 3)
            Current positions of all particles.
        species : np.ndarray, shape (N,) of int
            Species labels for all particles.
        i : int
            Index of the particle being tested (excluded from check).
            Pass ``i = -1`` (or any out-of-range index) for an insertion
            where the particle is not yet in *positions*.
        box_length : float
            Cubic box side length.

        Returns
        -------
        bool
            ``True`` if *pos_i* overlaps any particle j ≠ i.
        """
        pos_i = np.asarray(pos_i, dtype=float)
        positions = np.asarray(positions, dtype=float)
        species = np.asarray(species, dtype=int)
        species_i = int(species_i)

        N = len(positions)
        if N == 0:
            return False

        dr = positions - pos_i                               # (N, 3)
        dr -= box_length * np.round(dr / box_length)
        r2 = np.sum(dr ** 2, axis=1)                        # (N,)

        for j in range(N):
            if j == i:
                continue
            sij2 = self._sigma2[(species_i, int(species[j]))]
            if r2[j] < sij2:
                return True
        return False

    def pairwise_overlap(
        self,
        positions: np.ndarray,
        species: np.ndarray,
        box_length: float,
    ) -> bool:
        """O(N²) check: ``True`` if any pair of particles overlaps.

        Intended for validation and initialisation only.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
        species : np.ndarray, shape (N,) of int
        box_length : float

        Returns
        -------
        bool
        """
        positions = np.asarray(positions, dtype=float)
        species = np.asarray(species, dtype=int)
        N = len(positions)
        for i in range(N - 1):
            for j in range(i + 1, N):
                dr = positions[j] - positions[i]
                dr -= box_length * np.round(dr / box_length)
                r2 = float(np.dot(dr, dr))
                sij2 = self._sigma2[(int(species[i]), int(species[j]))]
                if r2 < sij2:
                    return True
        return False


# ── WCA potential (purely repulsive) ──────────────────────────────────


class WCA(LennardJones):
    """Weeks-Chandler-Andersen (purely repulsive) potential.

    r_cut = 2^(1/6) σ  — cuts at the potential minimum, so only the
    repulsive part remains.  No tail correction needed (u → 0 smoothly
    at the cutoff by construction of the shift).

    The WCA shift sets u(r_cut) = ε, giving:
        u_WCA(r) = 4ε[(σ/r)¹² - (σ/r)⁶] + ε   for r < 2^(1/6)σ
        u_WCA(r) = 0                              for r ≥ 2^(1/6)σ

    This is equivalent to the LennardJones class with r_cut = 2^(1/6).

    Parameters
    ----------
    epsilon : float
    sigma : float
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
    ) -> None:
        # WCA cutoff: 2^(1/6) sigma (in units of sigma: 2^(1/6))
        r_cut_wca = 2.0 ** (1.0 / 6.0)
        super().__init__(epsilon=epsilon, sigma=sigma, r_cut=r_cut_wca)

    def tail_correction_energy(
        self,
        n_particles: int,
        density: float,
    ) -> float:
        """WCA has no tail correction (potential is zero beyond r_cut)."""
        return 0.0

    def tail_correction_pressure(self, density: float) -> float:
        """WCA has no tail correction."""
        return 0.0
