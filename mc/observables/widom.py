"""
Widom Test-Particle Insertion
==============================

Estimates the excess chemical potential via:

    βμ_ex = -ln< exp(-βu_test) >

For hard spheres, exp(-βu) = 0 (overlap) or 1 (no overlap), so:

    βμ_ex = -ln(p_insert)

where p_insert is the fraction of trial insertions with no overlap.

Implementation
--------------
Random test positions are generated; for each one we check overlap
with every existing particle (minimum-image convention).  The JAX
vmap pattern enables batched overlap checking efficiently.

The :class:`WidomAccumulator` accumulates over multiple MC frames
and provides the mean and standard deviation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple

jax.config.update("jax_enable_x64", True)


# ── Batch overlap check (JIT + vmap) ─────────────────────────────────


@jax.jit
def _batch_overlap_check(
    positions: jnp.ndarray,
    test_positions: jnp.ndarray,
    box_length: float,
    sigma: float,
) -> jnp.ndarray:
    """Check overlap for many test positions simultaneously.

    Parameters
    ----------
    positions : (N, 3)  existing particles
    test_positions : (M, 3)  trial insertion positions
    box_length : float
    sigma : float

    Returns
    -------
    no_overlap : (M,) bool array
        True where the test position has no overlap with any particle.
    """
    sigma2 = sigma ** 2

    def single_test(test_pos):
        dr = positions - test_pos        # (N, 3)
        dr = dr - box_length * jnp.round(dr / box_length)
        r2 = jnp.sum(dr ** 2, axis=1)   # (N,)
        return ~jnp.any(r2 < sigma2)    # True = no overlap

    return jax.vmap(single_test)(test_positions)   # (M,)


# ── Public functions ──────────────────────────────────────────────────


def widom_insertion(
    positions,
    box_length: float,
    n_test: int = 1000,
    sigma: float = 1.0,
    key: Optional[jnp.ndarray] = None,
) -> float:
    """Compute Widom insertion probability for one configuration.

    Inserts *n_test* ghost particles at random positions and returns
    the fraction that do not overlap any real particle.

        p_insert = <exp(-β u_test)>  [hard spheres: 0 or 1]

    Parameters
    ----------
    positions : array-like, shape (N, 3)
    box_length : float
    n_test : int
    sigma : float
    key : jax PRNG key  (generated from numpy seed if None)

    Returns
    -------
    float
        Insertion probability p_insert ∈ [0, 1].
    """
    positions = jnp.asarray(positions)
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(0, 2**31))

    test_positions = jax.random.uniform(
        key, shape=(n_test, 3), minval=0.0, maxval=box_length
    )
    no_overlap = _batch_overlap_check(positions, test_positions, box_length, sigma)
    return float(jnp.mean(no_overlap))


def mu_excess(
    positions,
    box_length: float,
    n_test: int = 1000,
    sigma: float = 1.0,
    key: Optional[jnp.ndarray] = None,
) -> float:
    """Excess chemical potential via Widom insertion.

    βμ_ex = -ln(p_insert)

    Returns ``jnp.inf`` if p_insert = 0 (high-density limit where
    no insertions are accepted — Widom method breaks down).

    Parameters
    ----------
    positions : array-like, shape (N, 3)
    box_length : float
    n_test : int
    sigma : float
    key : jax PRNG key

    Returns
    -------
    float
        βμ_ex.
    """
    p = widom_insertion(positions, box_length, n_test, sigma, key)
    if p <= 0.0:
        return float("inf")
    return float(-np.log(p))


# ── Accumulator class ─────────────────────────────────────────────────


class WidomAccumulator:
    """Accumulate Widom insertion measurements over many MC frames.

    Each call to :meth:`update` performs *n_test* insertions for the
    current configuration and stores the resulting βμ_ex estimate.
    :meth:`get` returns the mean and standard deviation.

    Parameters
    ----------
    n_test : int
        Number of test insertions per configuration.
    sigma : float
    seed : int
        Base seed; each frame uses a different derived key.
    """

    def __init__(self, n_test: int = 1000, sigma: float = 1.0, seed: int = 0):
        self.n_test = n_test
        self.sigma = float(sigma)
        self._key = jax.random.PRNGKey(seed)
        self._samples: list[float] = []

    def update(self, positions, box_length: float) -> float:
        """Perform Widom insertion on the current configuration.

        Returns βμ_ex for this snapshot (also stored internally).
        """
        self._key, subkey = jax.random.split(self._key)
        mu_ex_val = mu_excess(
            positions, box_length, self.n_test, self.sigma, subkey
        )
        self._samples.append(mu_ex_val)
        return mu_ex_val

    def get(self) -> Tuple[float, float]:
        """Return (mean βμ_ex, std βμ_ex) over accumulated frames.

        Frames with infinite μ_ex (no successful insertions) are excluded
        from the mean; if all frames are infinite the result is (inf, nan).
        """
        data = np.array(self._samples, dtype=float)
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return float("inf"), float("nan")
        return float(np.mean(finite)), float(np.std(finite, ddof=1) if len(finite) > 1 else 0.0)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def n_frames(self) -> int:
        return len(self._samples)
