"""
Ornstein-Zernike Equation Solver in Fourier Space
==================================================

Solves the Ornstein-Zernike (OZ) equation to obtain the total correlation
function h(r) and pair distribution function g(r) from the direct correlation
function c_2(r).

The OZ equation relates the total correlation function h(r) = g(r) - 1 to
the direct correlation function c_2(r) via the integral equation:

    h(r) = c_2(r) + rho_bulk * int c_2(|r - r'|) h(r') dr'

In Fourier space this convolution becomes algebraic:

    h_hat(k) = c_hat_2(k) / (1 - rho_bulk * c_hat_2(k))

This is one of the fundamental equations of liquid state theory (Ornstein &
Zernike, 1914). Combined with a closure relation (PY, HNC, etc.) it yields
the equilibrium pair structure of a fluid.

The static structure factor S(k), measurable by neutron or X-ray scattering,
is directly related:

    S(k) = 1 / (1 - rho_bulk * c_hat_2(k))

so that h_hat(k) = c_hat_2(k) * S(k).

Functions
---------
solve_oz_fourier
    Solve the OZ equation in Fourier space for h_hat(k).
compute_pair_correlation
    Full pipeline: c_2(k) -> h(k) -> h(r) -> g(r) on the 3D grid.
compute_structure_factor
    Static structure factor S(k) from c_2(k).
radial_average
    Spherical average of a 3D field f(r) -> f(|r|).
compute_g_radial
    Convenience: g(|r|) from c_2(k) via OZ + radial averaging.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

jax.config.update("jax_enable_x64", True)

# Small regularization to prevent division by zero at points where
# 1 - rho * c_hat_2(k) vanishes (spinodal instability).
_EPS = 1e-12


def solve_oz_fourier(c2_k: Array, rho_bulk: float) -> Array:
    """
    Solve the Ornstein-Zernike equation in Fourier space.

    Computes the Fourier transform of the total correlation function:

        h_hat(k) = c_hat_2(k) / (1 - rho_bulk * c_hat_2(k))

    At k = 0 this gives the compressibility relation:

        rho_bulk * h_hat(0) = rho_bulk * kT * chi_T - 1

    where chi_T is the isothermal compressibility. The denominator
    1 - rho * c_hat_2(k) vanishes at the spinodal, signalling a
    thermodynamic instability.

    Parameters
    ----------
    c2_k : Array
        Fourier transform of the direct correlation function c_2(r).
        Same shape as the grid wavevector arrays (3D complex array from
        ``jnp.fft.fftn``).
    rho_bulk : float
        Bulk number density of the fluid.

    Returns
    -------
    h_k : Array
        Fourier transform of the total correlation function h(r) = g(r) - 1.
        Same shape as ``c2_k``.

    Notes
    -----
    A small regularization ``_EPS`` is added to the denominator to avoid
    numerical divergence near the spinodal. For thermodynamically stable
    states the denominator is strictly positive for all k.
    """
    denominator = 1.0 - rho_bulk * c2_k
    # Regularize: keep the sign but clamp the magnitude away from zero
    denominator_safe = jnp.where(
        jnp.abs(denominator) < _EPS,
        jnp.sign(denominator) * _EPS,
        denominator,
    )
    h_k = c2_k / denominator_safe
    return h_k


def compute_structure_factor(c2_k: Array, rho_bulk: float) -> Array:
    """
    Compute the static structure factor S(k).

    The structure factor is the Fourier transform of the density-density
    correlation function, directly measurable by scattering experiments:

        S(k) = 1 + rho_bulk * h_hat(k)
             = 1 / (1 - rho_bulk * c_hat_2(k))

    For an ideal gas S(k) = 1 everywhere. Deviations encode the pair
    structure: the first peak of S(k) at k ~ 2*pi/sigma reflects the
    nearest-neighbour shell.

    At k -> 0, S(0) = rho_bulk * kT * chi_T gives the isothermal
    compressibility via the compressibility sum rule.

    Parameters
    ----------
    c2_k : Array
        Fourier transform of the direct correlation function c_2(r).
    rho_bulk : float
        Bulk number density.

    Returns
    -------
    S_k : Array
        Static structure factor. Real-valued for physical systems (the
        imaginary part is discarded).
    """
    denominator = 1.0 - rho_bulk * c2_k
    denominator_safe = jnp.where(
        jnp.abs(denominator) < _EPS,
        jnp.sign(denominator) * _EPS,
        denominator,
    )
    S_k = 1.0 / denominator_safe
    return jnp.real(S_k)


def compute_pair_correlation(c2_k: Array, rho_bulk: float, grid) -> Array:
    """
    Compute the pair distribution function g(r) on the 3D grid.

    Full pipeline:
        1. Solve OZ in Fourier space: c_hat_2(k) -> h_hat(k)
        2. Inverse FFT: h_hat(k) -> h(r)
        3. g(r) = 1 + h(r)

    The result is the full 3D g(r) field on the computational grid. For
    an isotropic bulk fluid this is spherically symmetric about the origin;
    use ``radial_average`` or ``compute_g_radial`` to obtain g(|r|).

    Parameters
    ----------
    c2_k : Array
        Fourier transform of the direct correlation function c_2(r).
        Must have shape ``grid.shape``.
    rho_bulk : float
        Bulk number density.
    grid : Grid
        Computational grid (provides the FFT conventions and normalization).

    Returns
    -------
    g_r : Array
        Pair distribution function g(r) on the 3D grid. Shape ``grid.shape``.
        Real-valued (imaginary part from numerical noise is discarded).
    """
    h_k = solve_oz_fourier(c2_k, rho_bulk)
    h_r = jnp.fft.ifftn(h_k)
    g_r = 1.0 + jnp.real(h_r)
    return g_r


def radial_average(field_3d: Array, grid, n_bins: int = 200):
    """
    Compute the spherical (radial) average of a 3D field.

    Maps f(r) -> f(|r|) by binning grid points according to their distance
    from the origin and averaging within each bin. Uses the minimum-image
    convention for a periodic box: distances are wrapped so that the maximum
    distance is L/2 (half the smallest box side).

    Parameters
    ----------
    field_3d : Array
        3D field to average, shape ``grid.shape``.
    grid : Grid
        Computational grid providing real-space coordinates.
    n_bins : int, optional
        Number of radial bins (default 200).

    Returns
    -------
    r_bins : Array, shape (n_bins,)
        Bin centres (radial distances).
    f_radial : Array, shape (n_bins,)
        Radially averaged field values. Bins with no grid points contain 0.
    """
    # Minimum-image distances from the origin for periodic box
    # Shift coordinates so origin is at corner (grid coords are [0, L))
    rx = grid.X
    ry = grid.Y
    rz = grid.Z

    # Minimum image: wrap to [-L/2, L/2)
    rx = rx - grid.Lx * jnp.round(rx / grid.Lx)
    ry = ry - grid.Ly * jnp.round(ry / grid.Ly)
    rz = rz - grid.Lz * jnp.round(rz / grid.Lz)

    r = jnp.sqrt(rx**2 + ry**2 + rz**2)

    # Maximum meaningful distance is half the smallest box side
    r_max = min(grid.Lx, grid.Ly, grid.Lz) / 2.0
    dr = r_max / n_bins

    # Bin edges and centres
    r_bins = (jnp.arange(n_bins) + 0.5) * dr  # bin centres

    # Assign each grid point to a bin index
    bin_idx = jnp.floor(r / dr).astype(jnp.int32)

    # Flatten for histogram-like accumulation
    flat_field = field_3d.flatten()
    flat_idx = bin_idx.flatten()

    # Accumulate values and counts per bin
    # Use segment_sum-like approach via jnp.zeros scatter
    sum_vals = jnp.zeros(n_bins, dtype=field_3d.dtype)
    counts = jnp.zeros(n_bins, dtype=jnp.float64)

    # Mask: only include points within r_max
    mask = flat_idx < n_bins

    sum_vals = sum_vals.at[flat_idx].add(jnp.where(mask, flat_field, 0.0))
    counts = counts.at[flat_idx].add(jnp.where(mask, 1.0, 0.0))

    # Average (avoid division by zero for empty bins)
    f_radial = jnp.where(counts > 0, sum_vals / counts, 0.0)

    return r_bins, f_radial


def compute_g_radial(
    c2_k: Array, rho_bulk: float, grid, n_bins: int = 200
):
    """
    Compute the radial pair distribution function g(|r|).

    Convenience function combining the full OZ pipeline with radial
    averaging:

        c_hat_2(k) -> OZ -> h_hat(k) -> IFFT -> h(r) -> g(r) -> g(|r|)

    Parameters
    ----------
    c2_k : Array
        Fourier transform of the direct correlation function c_2(r).
    rho_bulk : float
        Bulk number density.
    grid : Grid
        Computational grid.
    n_bins : int, optional
        Number of radial bins (default 200).

    Returns
    -------
    r_values : Array, shape (n_bins,)
        Radial distance bin centres.
    g_values : Array, shape (n_bins,)
        Radially averaged pair distribution function g(|r|).
    """
    g_r = compute_pair_correlation(c2_k, rho_bulk, grid)
    r_values, g_values = radial_average(g_r, grid, n_bins=n_bins)
    return r_values, g_values
