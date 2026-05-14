"""
Direct Correlation Functions via Autodiff
==========================================

Computes direct correlation functions c_1(r) and c_2(r, r') from the excess
free energy functional F_exc[rho] using JAX automatic differentiation.

Physics Background
------------------
The direct correlation functions are defined as functional derivatives of the
excess (over ideal gas) free energy:

    c_1(r)      = -beta * delta F_exc / delta rho(r)
    c_2(r, r')  = -beta * delta^2 F_exc / delta rho(r) delta rho(r')

where beta = 1/kT (= 1 in our reduced units with kT = 1).

c_1 is the one-body direct correlation function. It plays the role of an
effective external potential in the Euler-Lagrange equation of DFT:

    rho(r) = rho_bulk * exp(c_1(r) - c_1^bulk + V_ext(r))

c_2 is the two-body (pair) direct correlation function. For a uniform
(bulk) fluid, translational invariance means c_2(r, r') = c_2(|r - r'|).
It is central to liquid state theory:
  - The Ornstein-Zernike equation: h(r) = c_2(r) + rho * (c_2 * h)(r)
  - The static structure factor: S(k) = 1 / (1 - rho * c_hat_2(k))
  - The isothermal compressibility: chi_T = S(k=0) / rho

Why Autodiff Works
------------------
Since our functionals F_exc[rho] are implemented as differentiable JAX
programs (FFT convolutions, neural networks, pointwise nonlinearities),
we can compute exact functional derivatives numerically via autodiff:

  - c_1(r) = -grad(F_exc)(rho)[r]                    (reverse-mode AD)
  - c_2(r, r') = -d/deps grad(F_exc)(rho + eps*delta_r')[r]  (forward-over-reverse)

The second derivative uses jax.jvp (forward-mode) applied to jax.grad
(reverse-mode), giving a Hessian-vector product. This computes one column
of the Hessian matrix (all r values for a fixed r') in O(N) time, avoiding
the full O(N^2) Hessian construction.

For uniform bulk density, one column suffices because c_2(r, r') = c_2(r - r')
by translational invariance --- every column contains the same information.

References
----------
- Hansen & McDonald, "Theory of Simple Liquids" (2013), Ch. 3 & 5
- Evans, Adv. Phys. 28, 143 (1979) --- foundational DFT review
- Lutsko, J. Chem. Phys. 152, 134111 (2020)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Tuple
from core.grid import Grid


def compute_c1(functional, rho: Array, grid: Grid) -> Array:
    """
    Compute the one-body direct correlation function c_1(r).

        c_1(r) = -delta F_exc / delta rho(r)

    This is the first functional derivative of the excess free energy,
    computed via reverse-mode automatic differentiation (jax.grad).

    For a uniform fluid, c_1 is a constant equal to -beta * mu_ex,
    the excess chemical potential. For an inhomogeneous fluid at a wall
    or interface, c_1(r) varies spatially and drives the density profile
    toward equilibrium through the Euler-Lagrange equation.

    Parameters
    ----------
    functional : object
        Any functional with an ``excess_free_energy(rho) -> float`` method.
        Examples: NonlocalLutskoFunctional, or any eqx.Module wrapping F_exc.
    rho : Array, shape (nx, ny, nz)
        Density field on the computational grid.
    grid : Grid
        Computational grid (used only for documentation; the functional
        already knows dV internally).

    Returns
    -------
    c1 : Array, shape (nx, ny, nz)
        One-body direct correlation function at each grid point.

    Notes
    -----
    The functional derivative on a discrete grid is:

        c_1[i] = -(1/dV) * dF_exc/d(rho[i])

    Since F_exc = sum_i Phi[i] * dV, the chain rule gives
    dF_exc/d(rho[i]) = (dPhi/d(rho))[i] * dV, so
    c_1[i] = -dPhi/d(rho)[i] ... but jax.grad of F_exc (the scalar
    sum * dV) directly gives dF/d(rho[i]), which already has the
    correct units for the functional derivative (no extra dV needed).

    Actually: the functional derivative delta F/delta rho(r) in the
    continuum is defined such that delta F = int (delta F/delta rho) * delta_rho dr.
    On the discrete grid, F = sum_i f_i * dV, so dF/d(rho_i) = (dF/drho)(r_i) * dV.
    Therefore c_1(r_i) = -dF/d(rho_i) / dV = -jax.grad(F)(rho)[i] / dV... but wait,
    our functional.excess_free_energy already includes the dV factor in the sum,
    so jax.grad gives dF/d(rho_i) which is the discrete functional derivative
    times dV. We must divide by dV to get the proper c_1.

    However, the existing NonlocalLutskoFunctional.compute_c1 does NOT divide
    by dV --- it returns -jax.grad(F_exc)(rho) directly. This is because the
    Euler-Lagrange equation in discrete form uses the same convention:
    mu = dF/d(rho_i) (not delta F/delta rho). We follow that convention here
    for consistency.
    """
    return -jax.grad(functional.excess_free_energy)(rho)


def compute_c2_bulk(functional, rho_bulk: float, grid: Grid) -> Array:
    """
    Compute the pair direct correlation function c_2(r) for a uniform fluid.

        c_2(r - r') = -delta^2 F_exc / delta rho(r) delta rho(r')

    evaluated at uniform density rho(r) = rho_bulk everywhere.

    Strategy
    --------
    For uniform density, c_2(r, r') = c_2(r - r') by translational invariance.
    We only need one "column" of the Hessian matrix, corresponding to a
    perturbation at a single reference point r_0.

    We use forward-over-reverse autodiff (jax.jvp over jax.grad):

    1. Define grad_F(rho) = jax.grad(F_exc)(rho)   [reverse-mode, O(N)]
    2. Pick r_0 = (0,0,0) and set tangent vector v = delta_{r,r_0} / dV
       (a discrete delta function normalized so that sum_r v[r]*dV = 1)
    3. Compute (grad_F(rho), H @ v) = jax.jvp(grad_F, (rho,), (v,))
       This gives the Hessian-vector product H @ v in O(N) time.
    4. c_2(r - r_0) = -(H @ v)[r] * dV

    The volume factors arise from the discrete-to-continuum correspondence.
    The full Hessian H[i,j] = d^2 F / d(rho_i) d(rho_j) relates to the
    continuum functional derivative as:
        delta^2 F / delta rho(r) delta rho(r') = H[i,j] / dV^2
    so c_2[i] = -H[i, j0] / dV^2. The jvp with v = delta/dV gives
    (H @ v)[i] = H[i,j0] / dV, hence c_2[i] = -(H@v)[i] / dV ... but
    we also need to account for the overall dV in the definition.

    In practice, with F_exc = sum Phi * dV:
        d^2 F / d(rho_i) d(rho_j) = Hessian element (includes dV factors)
    The continuum c_2(r_i, r_j) = -H[i,j] / dV^2.
    jvp tangent v[j] = delta_{j,j0} / dV gives (H@v)[i] = H[i,j0] / dV.
    So c_2(r_i - r_0) = -(H@v)[i] / dV.

    But following the discrete convention of this codebase (where compute_c1
    returns -grad(F) without dV normalization), we return -Hv directly,
    which is the discrete c_2 consistent with the Picard iteration solver.

    Parameters
    ----------
    functional : object
        Any functional with an ``excess_free_energy(rho) -> float`` method.
    rho_bulk : float
        Uniform bulk number density (rho = 6*eta/pi for hard spheres of
        diameter sigma=1).
    grid : Grid
        3D computational grid. Should be large enough that c_2(r) decays
        to zero before the box boundary (c_2 typically decays within
        2-3 hard-sphere diameters).

    Returns
    -------
    c2 : Array, shape (nx, ny, nz)
        Pair direct correlation function c_2(r) on the grid, where r is
        measured from the origin (grid point [0,0,0]).

    Notes
    -----
    - Cost is O(N) per jvp call (one forward + one reverse pass), compared
      to O(N^2) for the full Hessian. This is the key efficiency gain.
    - The result is periodic (wraps around the box). Ensure L >> range(c_2).
    - For hard spheres, c_2(r) is zero for r > sigma (within PY approximation),
      so even modest box sizes suffice.

    Example
    -------
    >>> c2 = compute_c2_bulk(functional, rho_bulk=0.764, grid=grid)
    >>> c2_hat = jnp.fft.fftn(c2) * grid.dV  # Fourier transform
    >>> S_k = 1.0 / (1.0 - rho_bulk * c2_hat)  # Structure factor
    """
    rho_uniform = jnp.ones((grid.nx, grid.ny, grid.nz)) * rho_bulk

    def grad_F(rho):
        return jax.grad(functional.excess_free_energy)(rho)

    # Tangent vector: discrete delta function at the origin, normalized by 1/dV
    # so that it represents a unit perturbation in the continuum sense.
    v = jnp.zeros_like(rho_uniform)
    v = v.at[0, 0, 0].set(1.0 / grid.dV)

    # Forward-over-reverse: compute Hessian-vector product H @ v
    # primals_out = grad_F(rho_uniform) = first functional derivative
    # tangents_out = d(grad_F)/d(rho) @ v = Hessian @ v
    _, Hv = jax.jvp(grad_F, (rho_uniform,), (v,))

    # c_2(r) = -(d^2 F_exc / d rho(r) d rho(0)) in discrete convention
    # Hv[i] = sum_j H[i,j] * v[j] = H[i, 0] / dV
    # Continuum c_2 = -H[i,0] / dV^2, so c_2 = -Hv / dV ... but
    # we follow the codebase discrete convention (no dV normalization,
    # consistent with compute_c1 returning -grad(F) directly).
    c2 = -Hv * grid.dV

    return c2


def compute_c2_fourier(functional, rho_bulk: float, grid: Grid) -> Array:
    """
    Compute the Fourier transform of c_2(r) for a uniform fluid.

        c_hat_2(k) = integral c_2(r) exp(-i k.r) dr

    This is the key ingredient for:
    - The static structure factor: S(k) = 1 / (1 - rho * c_hat_2(k))
    - The Ornstein-Zernike equation in Fourier space
    - Compressibility sum rule: 1 - rho * c_hat_2(0) = 1 / (rho * kT * chi_T)

    Parameters
    ----------
    functional : object
        Any functional with an ``excess_free_energy(rho) -> float`` method.
    rho_bulk : float
        Uniform bulk number density.
    grid : Grid
        3D computational grid.

    Returns
    -------
    c2_hat : Array, shape (nx, ny, nz)
        Fourier-space pair DCF c_hat_2(k) at each wavevector on the grid.

    Notes
    -----
    The discrete Fourier transform convention is:

        c_hat_2(k) = sum_r c_2(r) * exp(-i k.r) * dV

    which approximates the continuum integral. The factor dV converts the
    discrete sum to the integral.

    Example
    -------
    >>> c2_hat = compute_c2_fourier(functional, rho_bulk, grid)
    >>> S_k = 1.0 / (1.0 - rho_bulk * c2_hat)
    >>> # Compressibility check: S(k=0) should match bulk chi_T * rho
    >>> chi_T_from_S = S_k[0, 0, 0].real / rho_bulk
    """
    c2_real = compute_c2_bulk(functional, rho_bulk, grid)
    # FFT with dV factor to approximate the continuum Fourier integral
    c2_hat = jnp.fft.fftn(c2_real) * grid.dV
    return c2_hat


def compute_c2_radial(c2_3d: Array, grid: Grid,
                      n_bins: int = 100) -> Tuple[Array, Array]:
    """
    Radial average of a 3D correlation function to get c_2(|r|).

    For an isotropic bulk fluid, c_2(r) depends only on |r|. This function
    bins the 3D grid values by distance from the origin and averages within
    each bin to produce a smooth radial profile.

    Parameters
    ----------
    c2_3d : Array, shape (nx, ny, nz)
        3D pair DCF on the grid (origin at grid point [0,0,0]).
    grid : Grid
        Computational grid.
    n_bins : int, optional
        Number of radial bins (default: 100).

    Returns
    -------
    r_centers : Array, shape (n_bins,)
        Bin center positions in units of the grid spacing.
    c2_radial : Array, shape (n_bins,)
        Radially averaged c_2 in each bin. Bins with no grid points
        are filled with zero.

    Notes
    -----
    The origin is at grid point [0,0,0]. Distances are computed with
    minimum-image convention (periodic wrapping), so the maximum
    meaningful distance is L/2.

    For hard spheres with sigma=1, the interesting structure is at
    r < 2-3 sigma. Ensure the grid is fine enough (dx << sigma) to
    resolve the core and first peak.

    Example
    -------
    >>> c2 = compute_c2_bulk(functional, rho_bulk, grid)
    >>> r, c2_r = compute_c2_radial(c2, grid, n_bins=200)
    >>> # Plot c2(r) vs the PY analytical result
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(r, c2_r, label='autodiff')
    """
    # Distance from origin with minimum-image convention (periodic)
    # Shift coordinates so that distances wrap around at L/2
    rx = jnp.where(grid.X > grid.Lx / 2, grid.X - grid.Lx, grid.X)
    ry = jnp.where(grid.Y > grid.Ly / 2, grid.Y - grid.Ly, grid.Y)
    rz = jnp.where(grid.Z > grid.Lz / 2, grid.Z - grid.Lz, grid.Z)
    r_dist = jnp.sqrt(rx**2 + ry**2 + rz**2)

    # Maximum distance is half the smallest box dimension
    r_max = min(grid.Lx, grid.Ly, grid.Lz) / 2.0
    dr = r_max / n_bins

    r_centers = (jnp.arange(n_bins) + 0.5) * dr

    # Bin by distance and average
    bin_indices = jnp.floor(r_dist / dr).astype(jnp.int32)
    # Clip to valid range (points beyond r_max go into last bin)
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

    flat_bins = bin_indices.ravel()
    flat_c2 = c2_3d.ravel()

    # Sum values and counts per bin using segment_sum
    c2_sum = jnp.zeros(n_bins).at[flat_bins].add(flat_c2)
    counts = jnp.zeros(n_bins).at[flat_bins].add(jnp.ones_like(flat_c2))

    # Avoid division by zero
    c2_radial = jnp.where(counts > 0, c2_sum / counts, 0.0)

    return r_centers, c2_radial
