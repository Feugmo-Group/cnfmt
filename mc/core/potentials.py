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
