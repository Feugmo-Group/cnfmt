"""
MCState — Simulation State
===========================

Immutable JAX-compatible representation of an MC configuration.
All arrays are JAX arrays for GPU compatibility.

Key conventions
---------------
- Hard-sphere diameter σ = 1.0 throughout.
- Box is a cubic periodic cell of side L.
- Positions in [0, L)^3 (wrapped by PBC).
- FCC init: n_cells per dimension, 4 basis atoms per unit cell.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Enable double precision — required for physics accuracy
jax.config.update("jax_enable_x64", True)


@dataclass
class MCState:
    """Immutable simulation state.

    Parameters
    ----------
    positions : jnp.ndarray, shape (N, 3)
        Particle positions in [0, box_length)^3.
    box_length : float
        Side length of the cubic simulation box.
    n_particles : int
        Number of particles N.
    """
    positions: jnp.ndarray   # (N, 3)
    box_length: float
    n_particles: int

    # ── Thermodynamic properties ──────────────────────────────────

    def volume(self) -> float:
        """Box volume V = L^3."""
        return self.box_length ** 3

    def density(self) -> float:
        """Number density ρ = N / V."""
        return self.n_particles / self.volume()

    def packing_fraction(self, sigma: float = 1.0) -> float:
        """Packing fraction η = N * π * σ³ / (6V)."""
        return self.n_particles * np.pi * sigma**3 / (6.0 * self.volume())

    # ── Factory methods ───────────────────────────────────────────

    @staticmethod
    def from_fcc(N: int, box_length: float) -> "MCState":
        """Initialise an FCC lattice configuration.

        Constructs an FCC crystal with at least N sites and returns the first
        N positions.  If the requested N is not a perfect multiple of 4 (the
        FCC motif), a few extra sites are generated and then discarded.

        FCC basis vectors (relative to unit cell corner):
            (0, 0, 0)  (1/2, 1/2, 0)  (1/2, 0, 1/2)  (0, 1/2, 1/2)

        Parameters
        ----------
        N : int
            Number of particles.
        box_length : float
            Cubic box side length.

        Returns
        -------
        MCState
        """
        # Number of unit cells per dimension (ceil so we have enough sites)
        n_cells = int(np.ceil((N / 4) ** (1.0 / 3.0)))
        a = box_length / n_cells           # lattice spacing

        # FCC basis (fractional units of a)
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])

        positions = []
        for ix in range(n_cells):
            for iy in range(n_cells):
                for iz in range(n_cells):
                    corner = np.array([ix, iy, iz], dtype=float) * a
                    for b in basis:
                        positions.append(corner + b * a)

        positions = np.array(positions)   # (n_cells^3 * 4, 3)

        # Take first N and wrap into box (fractional coords can slightly exceed L
        # for last cell; PBC-wrap to be safe)
        positions = positions[:N] % box_length

        return MCState(
            positions=jnp.array(positions),
            box_length=float(box_length),
            n_particles=N,
        )

    @staticmethod
    def from_random(
        N: int,
        box_length: float,
        seed: int = 42,
        sigma: float = 1.0,
        max_attempts: int = 100_000,
    ) -> "MCState":
        """Random non-overlapping initialisation via sequential rejection.

        Sequentially inserts particles at random positions and rejects any
        placement that would overlap an already-placed particle.  Suitable
        for η ≲ 0.30; use :meth:`from_fcc` for denser systems.

        Parameters
        ----------
        N : int
        box_length : float
        seed : int
            NumPy random seed.
        sigma : float
            Hard-sphere diameter.
        max_attempts : int
            Maximum total insertion attempts before raising RuntimeError.

        Returns
        -------
        MCState
        """
        rng = np.random.default_rng(seed)
        sigma2 = sigma ** 2
        placed: list[np.ndarray] = []
        total_attempts = 0

        while len(placed) < N:
            if total_attempts > max_attempts:
                raise RuntimeError(
                    f"Could not place {N} particles after {max_attempts} attempts "
                    f"(placed {len(placed)}).  Try from_fcc for dense systems."
                )
            candidate = rng.uniform(0.0, box_length, size=(3,))
            total_attempts += 1

            overlap = False
            for p in placed:
                dr = candidate - p
                # Minimum image convention
                dr = dr - box_length * np.round(dr / box_length)
                if np.dot(dr, dr) < sigma2:
                    overlap = True
                    break

            if not overlap:
                placed.append(candidate)

        positions = np.array(placed)
        return MCState(
            positions=jnp.array(positions),
            box_length=float(box_length),
            n_particles=N,
        )
