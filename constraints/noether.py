"""
Noether Symmetry Constraints
=============================

Translational invariance constraint loss for density functionals.

Physics: For any proper density functional F_exc[rho], Noether's theorem
requires that F_exc is invariant under continuous symmetries of the
underlying Hamiltonian. For a bulk fluid with periodic boundaries,
translational invariance demands:

    F_exc[rho(. + a)] = F_exc[rho(.)]   for all translations a

This holds exactly for the true free energy functional. A learned functional
that violates this symmetry has spurious position-dependent artifacts,
which corrupt the one-body direct correlation function c_1 = -delta F/delta rho.

Implementation uses Fourier-space phase shifts for exact periodic translation:
    rho(r + delta) = IFFT[ FFT[rho] * exp(-i k . delta) ]

This is exact on the periodic grid (no interpolation error) and fully
differentiable through JAX autodiff.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from core.grid import Grid


def shift_density_fourier(rho: Array, grid: Grid, shift_vector: Array) -> Array:
    """
    Shift a density field by an arbitrary vector using Fourier phase shift.

    For a periodic density field on a grid with wavevectors k, the
    shifted field is computed exactly via:

        rho(r + delta) = IFFT[ FFT[rho] * exp(-i k . delta) ]

    This avoids interpolation artifacts and is exact for band-limited
    functions on the periodic grid.

    Parameters
    ----------
    rho : Array, shape (nx, ny, nz)
        Density field to shift.
    grid : Grid
        Computational grid providing wavevectors Kx, Ky, Kz.
    shift_vector : Array, shape (3,)
        Translation vector (delta_x, delta_y, delta_z) in real-space units.

    Returns
    -------
    rho_shifted : Array, shape (nx, ny, nz)
        Shifted density field, real-valued.
    """
    rho_k = jnp.fft.fftn(rho)

    # Phase shift: exp(-i k . delta)
    phase = -(grid.Kx * shift_vector[0]
              + grid.Ky * shift_vector[1]
              + grid.Kz * shift_vector[2])
    shift_factor = jnp.exp(1j * phase)

    rho_shifted_k = rho_k * shift_factor
    rho_shifted = jnp.fft.ifftn(rho_shifted_k)

    return rho_shifted.real


def translational_invariance_loss(functional, rho: Array, grid: Grid,
                                  n_shifts: int = 5, max_shift: float = 0.5,
                                  key: Array = None) -> float:
    """
    Measure violation of translational invariance for a density functional.

    Computes the mean squared relative deviation of F_exc under random
    translations of the density field:

        L_TI = (1/N) sum_i |F_exc[rho(. + a_i)] - F_exc[rho]|^2 / F_exc[rho]^2

    For an exact functional L_TI = 0 identically. A nonzero value indicates
    the learned functional has acquired spurious position dependence,
    typically from the neural network or learnable kernel breaking the
    translational symmetry that FFT convolutions naturally preserve.

    Parameters
    ----------
    functional : NonlocalLutskoFunctional (or any object with .excess_free_energy)
        Density functional to test. Must have an ``excess_free_energy(rho)``
        method returning a scalar.
    rho : Array, shape (nx, ny, nz)
        Reference density field.
    grid : Grid
        Computational grid (provides wavevectors for Fourier shift).
    n_shifts : int, optional
        Number of random translations to average over. Default 5.
    max_shift : float, optional
        Maximum shift magnitude in units of the hard-sphere diameter sigma.
        The shift vector is drawn uniformly from a ball of this radius.
        Default 0.5 (half a particle diameter).
    key : jax.random.PRNGKey
        JAX PRNG key for generating random shifts. Required.

    Returns
    -------
    loss : float
        Mean squared relative deviation (dimensionless, >= 0).
        Zero indicates perfect translational invariance.

    Notes
    -----
    The Fourier phase-shift approach ensures that the translation itself
    introduces zero numerical error on the periodic grid. Any nonzero loss
    is therefore attributable to the functional, not the shift operation.

    This loss is JAX-differentiable and can be added to the training
    objective to regularize the learned functional toward the exact symmetry.
    """
    if key is None:
        raise ValueError("A JAX PRNGKey must be provided via the `key` argument.")

    F_ref = functional.excess_free_energy(rho)
    F_ref_sq = F_ref ** 2
    # Guard against division by zero for trivial density
    F_ref_sq = jnp.where(F_ref_sq > 1e-30, F_ref_sq, 1.0)

    keys = jax.random.split(key, n_shifts)

    def _single_shift_loss(subkey):
        # Draw random direction on unit sphere
        direction = jax.random.normal(subkey, shape=(3,))
        direction = direction / (jnp.linalg.norm(direction) + 1e-14)

        # Draw random magnitude uniformly in [0, max_shift * sigma]
        # sigma = 1 in reduced units for hard spheres
        subkey2 = jax.random.fold_in(subkey, 1)
        magnitude = jax.random.uniform(subkey2, minval=0.0, maxval=max_shift)

        shift_vector = direction * magnitude

        rho_shifted = shift_density_fourier(rho, grid, shift_vector)
        F_shifted = functional.excess_free_energy(rho_shifted)

        return (F_shifted - F_ref) ** 2 / F_ref_sq

    losses = jax.vmap(_single_shift_loss)(keys)

    return jnp.mean(losses)
