"""
Thermodynamic Consistency Constraint Losses
============================================

Physics-based loss functions enforcing thermodynamic consistency of the
neural density functional.  These go beyond sum rules (which check one
identity at a time) by cross-checking *different thermodynamic routes*
to the same quantity --- virial vs compressibility pressure, functional
c_2 vs Percus-Yevick reference, and structure factor comparisons.

A truly self-consistent functional must give the same answer regardless
of the thermodynamic route used to compute a given observable.  Classical
closures (PY, HNC) notoriously violate this --- the PY virial and
compressibility pressures differ.  A neural functional trained with these
losses can do better.

Pressure Consistency (virial = compressibility)
-----------------------------------------------
The virial pressure comes from the equation of state:

    beta P_vir = rho + rho^2 d(f_exc)/d(rho)

where f_exc = F_exc / V is the excess free energy density.

The compressibility pressure is obtained by integrating the compressibility:

    1 / (rho kT chi_T) = 1 - rho * c_hat_2(k=0)

For a self-consistent functional, P_vir = P_comp at all densities.

Ornstein-Zernike Consistency
----------------------------
The c_2(r) from autodiff, when fed through the OZ equation, must produce
a physically sensible g(r):
  - g(r) = 0 for r < sigma  (hard-core exclusion)
  - g(r) -> 1 for r -> infinity  (decorrelation)

c_2 Reference (Percus-Yevick)
-----------------------------
Soft constraint comparing c_2(r) from the functional to the analytical
PY solution.  The PY is approximate (not exact), so this is a regulariser
rather than a strict constraint.

Structure Factor
----------------
Compares S(k) from the functional to PY S(k).  More robust than
real-space c_2 comparison because S(k) is an experimentally measurable
quantity and small errors in c_2 are amplified differently in k-space.

References
----------
Hansen JP, McDonald IR, Theory of Simple Liquids, 4th ed. (2013), Ch. 3.
Lutsko JF, J. Chem. Phys. 152, 134111 (2020).
Evans R, Adv. Phys. 28, 143 (1979).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Optional

from core.grid import Grid
from correlations.direct import compute_c2_bulk, compute_c2_fourier, compute_c2_radial
from correlations.ornstein_zernike import (
    compute_g_radial,
    compute_structure_factor,
    radial_average,
)
from validation.percus_yevick import (
    py_direct_correlation,
    py_direct_correlation_fourier,
    py_structure_factor,
)

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════
# 1. PRESSURE CONSISTENCY (virial = compressibility)
# ═══════════════════════════════════════════════════════════════════

def pressure_consistency_loss(
    functional,
    eta_values: Array,
    grid: Grid,
    sigma: float = 1.0,
) -> Array:
    """
    Penalise inconsistency between virial and compressibility pressures.

    A thermodynamically self-consistent functional yields the same pressure
    from the virial route (d F_exc / d rho at uniform density) and the
    compressibility route (k=0 limit of c_hat_2).

    Virial route
    ~~~~~~~~~~~~~
    beta P_vir = rho + rho^2 * d(f_exc)/d(rho)

    where f_exc(rho) = F_exc[rho_uniform] / V.  The derivative is computed
    by autodiff of F_exc with respect to a global density scaling parameter
    alpha at alpha = 1:

        d(F_exc)/d(alpha)|_{alpha=1} = sum_r (dF/d rho_r) * rho_r
                                      = V * rho * d(f_exc)/d(rho)

    So  rho^2 * d(f_exc)/d(rho) = rho / V * dF/d(alpha).

    Compressibility route
    ~~~~~~~~~~~~~~~~~~~~~~
    S(k=0) = 1 / (1 - rho * c_hat_2(k=0))

    The compressibility pressure is obtained via the thermodynamic identity
    beta P_comp = rho * S(k=0)^{-1} integrated, but a simpler check is
    to compare the *inverse compressibility* from each route:

        (d(beta P)/d(rho))_vir  vs  1 - rho * c_hat_2(0)

    We use the direct comparison of beta P from each route for clarity.

    Parameters
    ----------
    functional : object
        Excess free energy functional with ``excess_free_energy(rho)`` method.
    eta_values : Array, shape (N,)
        Packing fractions at which to evaluate consistency.
    grid : Grid
        3D computational grid.
    sigma : float
        Hard-sphere diameter (default 1.0).

    Returns
    -------
    loss : Array (scalar)
        Mean squared relative pressure inconsistency over ``eta_values``.
    """
    loss = jnp.float64(0.0)
    n_eta = len(eta_values)

    for eta in eta_values:
        rho_bulk = 6.0 * eta / (jnp.pi * sigma**3)
        rho_uniform = jnp.ones((grid.nx, grid.ny, grid.nz)) * rho_bulk

        # ── Virial pressure ────────────────────────────────────────
        # F_exc(alpha * rho) as a function of scalar alpha, evaluated
        # at alpha = 1.  dF/d(alpha) = sum_r dF/d(rho_r) * rho_r.
        def F_of_alpha(alpha):
            return functional.excess_free_energy(alpha * rho_uniform)

        dF_dalpha = jax.grad(F_of_alpha)(jnp.float64(1.0))

        # f_exc = F_exc / V,  d(f_exc)/d(rho) = dF_dalpha / (V * rho)
        # beta P_vir = rho + rho^2 * d(f_exc)/d(rho)
        #            = rho + rho * dF_dalpha / V
        V = grid.volume
        P_virial = rho_bulk + rho_bulk * dF_dalpha / V

        # ── Compressibility pressure ───────────────────────────────
        # From c_hat_2(k=0): inverse compressibility factor
        #   1 / (rho * chi_T) = 1 - rho * c_hat_2(0)
        # beta P_comp is obtained from integrating d(beta P)/d(rho) = 1/(rho * chi_T)
        # but a direct comparison of the *local* d(beta P)/d(rho) is cleaner.
        #
        # Instead, we compare via the compressibility factor Z = beta P / rho:
        #   Z_vir = P_virial / rho
        #   Z_comp from OZ:  d(Z*rho)/d(rho) = 1 - rho * c_hat_2(0)
        #     => Z + rho * dZ/drho = 1 - rho * c_hat_2(0)
        #
        # The simplest route: compare beta P_virial to beta P from
        # the compressibility sum rule. For a uniform fluid at density rho,
        # the compressibility gives d(beta P)/d(rho) = 1/(rho chi_T),
        # and we can compute beta P_comp = integral_0^rho [1/(rho' chi_T(rho'))] drho'.
        #
        # But this integral is expensive.  Instead, compare the *derivative*:
        #   (d/d rho)(beta P_vir) vs 1/(rho chi_T) from c_hat_2(k=0)
        #
        # d(beta P_vir)/d(rho) via autodiff:

        def P_vir_of_rho(rho_val):
            rho_field = jnp.ones((grid.nx, grid.ny, grid.nz)) * rho_val

            def F_alpha(alpha):
                return functional.excess_free_energy(alpha * rho_field)

            dF_da = jax.grad(F_alpha)(jnp.float64(1.0))
            return rho_val + rho_val * dF_da / V

        dP_drho_vir = jax.grad(P_vir_of_rho)(rho_bulk)

        # From compressibility: d(beta P)/d(rho) = 1 - rho * c_hat_2(0)
        c2_k = compute_c2_fourier(functional, rho_bulk, grid)
        c2_k0 = jnp.real(c2_k[0, 0, 0])
        dP_drho_comp = 1.0 - rho_bulk * c2_k0

        # Squared relative error, normalised by compressibility route
        loss += (dP_drho_vir - dP_drho_comp)**2 / (dP_drho_comp**2 + 1e-10)

    return loss / n_eta


# ═══════════════════════════════════════════════════════════════════
# 2. ORNSTEIN-ZERNIKE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════

def oz_consistency_loss(
    functional,
    rho_bulk: float,
    grid: Grid,
    sigma: float = 1.0,
    w_core: float = 10.0,
    w_tail: float = 1.0,
    n_bins: int = 200,
) -> Array:
    """
    Penalise unphysical g(r) obtained from the functional's c_2 via OZ.

    The direct correlation function c_2(r) computed from the functional by
    autodiff is fed through the Ornstein-Zernike equation to produce g(r).
    For hard spheres, two exact constraints must hold:

    1. **Core exclusion**: g(r < sigma) = 0.
       Particles cannot overlap; any nonzero g inside the core signals
       an inconsistent c_2.

    2. **Decorrelation**: g(r >> sigma) -> 1.
       At large separations the pair distribution must approach unity.
       Persistent oscillations or drift indicate problems.

    Parameters
    ----------
    functional : object
        Excess free energy functional with ``excess_free_energy(rho)`` method.
    rho_bulk : float
        Uniform bulk number density.
    grid : Grid
        3D computational grid.  Must be large enough that g(r) decays
        within L/2 (typically L > 10*sigma).
    sigma : float
        Hard-sphere diameter (default 1.0).
    w_core : float
        Weight for core violation loss (default 10.0).
        Core violations are very serious --- g(r<sigma)=0 is exact.
    w_tail : float
        Weight for decorrelation loss (default 1.0).
    n_bins : int
        Number of radial bins for g(|r|) (default 200).

    Returns
    -------
    loss : Array (scalar)
        Weighted sum of core and tail violations.
    """
    # Compute c_2(k) from the functional via autodiff
    c2_k = compute_c2_fourier(functional, rho_bulk, grid)

    # Solve OZ and radially average to get g(|r|)
    r_values, g_values = compute_g_radial(c2_k, rho_bulk, grid, n_bins=n_bins)

    # ── Core exclusion: g(r < sigma) should be 0 ──────────────────
    core_mask = r_values < sigma
    g_core = jnp.where(core_mask, g_values, 0.0)
    loss_core = jnp.mean(g_core**2)

    # ── Decorrelation: g(r > 5*sigma) should be 1 ────────────────
    tail_mask = r_values > 5.0 * sigma
    g_tail_deviation = jnp.where(tail_mask, g_values - 1.0, 0.0)
    # Count tail points to avoid dividing by zero if box is too small
    n_tail = jnp.maximum(jnp.sum(tail_mask.astype(jnp.float64)), 1.0)
    loss_tail = jnp.sum(g_tail_deviation**2) / n_tail

    return w_core * loss_core + w_tail * loss_tail


# ═══════════════════════════════════════════════════════════════════
# 3. c_2 REFERENCE (PERCUS-YEVICK)
# ═══════════════════════════════════════════════════════════════════

def c2_reference_loss(
    functional,
    rho_bulk: float,
    grid: Grid,
    eta: float,
    sigma: float = 1.0,
    n_bins: int = 200,
) -> Array:
    """
    Soft constraint: compare c_2(r) from the functional to analytical PY.

    The Percus-Yevick solution is the best-known analytical approximation
    for the hard-sphere direct correlation function.  It is not exact, so
    this loss is a *soft* regulariser that biases the functional toward
    the PY structure rather than an absolute constraint.

    The comparison is restricted to r < 2*sigma where c_2 has significant
    structure.  Beyond 2*sigma, PY gives c_2 = 0 (exact for r > sigma
    in PY), so errors there are less informative.

    Parameters
    ----------
    functional : object
        Excess free energy functional with ``excess_free_energy(rho)`` method.
    rho_bulk : float
        Uniform bulk number density.
    grid : Grid
        3D computational grid.
    eta : float
        Packing fraction (needed for PY analytical formula).
    sigma : float
        Hard-sphere diameter (default 1.0).
    n_bins : int
        Number of radial bins (default 200).

    Returns
    -------
    loss : Array (scalar)
        Mean squared difference (c2_func - c2_PY)^2 for r < 2*sigma.

    Notes
    -----
    Since PY itself is approximate, the weight on this loss should be
    moderate (e.g. 0.1-1.0) to allow the functional freedom to improve
    upon PY where it can.
    """
    # Compute c_2(r) from functional via autodiff and radial averaging
    c2_3d = compute_c2_bulk(functional, rho_bulk, grid)
    r_values, c2_func = compute_c2_radial(c2_3d, grid, n_bins=n_bins)

    # PY analytical c_2(r) at the same radial positions
    c2_py = py_direct_correlation(r_values, eta, sigma)

    # Compare within r < 2*sigma (where c_2 has structure)
    comparison_mask = r_values < 2.0 * sigma
    diff = jnp.where(comparison_mask, c2_func - c2_py, 0.0)
    n_compare = jnp.maximum(jnp.sum(comparison_mask.astype(jnp.float64)), 1.0)

    loss = jnp.sum(diff**2) / n_compare
    return loss


# ═══════════════════════════════════════════════════════════════════
# 4. STRUCTURE FACTOR
# ═══════════════════════════════════════════════════════════════════

def structure_factor_loss(
    functional,
    rho_bulk: float,
    grid: Grid,
    eta: float,
    sigma: float = 1.0,
    n_bins: int = 200,
    k_max: Optional[float] = None,
) -> Array:
    """
    Compare S(k) from the functional to the analytical PY structure factor.

    The static structure factor S(k) = 1 / (1 - rho * c_hat_2(k)) is an
    experimentally measurable quantity (neutron / X-ray scattering).
    Matching the PY S(k) ensures the functional reproduces correct pair
    structure in reciprocal space.

    Comparison is done on a radially averaged S(|k|) grid.  The loss uses
    relative errors to handle the wide dynamic range of S(k) (it peaks
    strongly near k ~ 2*pi/sigma for dense fluids).

    Parameters
    ----------
    functional : object
        Excess free energy functional with ``excess_free_energy(rho)`` method.
    rho_bulk : float
        Uniform bulk number density.
    grid : Grid
        3D computational grid.
    eta : float
        Packing fraction (for PY reference).
    sigma : float
        Hard-sphere diameter (default 1.0).
    n_bins : int
        Number of radial k-bins (default 200).
    k_max : float, optional
        Maximum wavenumber for comparison.  Default: use all k within
        the grid's Nyquist range.

    Returns
    -------
    loss : Array (scalar)
        Mean squared relative error (S_func - S_PY)^2 / S_PY^2 over
        the usable k range.
    """
    # S(k) from functional via autodiff + OZ
    c2_k = compute_c2_fourier(functional, rho_bulk, grid)
    S_k_3d = compute_structure_factor(c2_k, rho_bulk)

    # Radially average S(|k|) using the Fourier-space grid
    # We create a k-magnitude field and bin S(k) by |k|
    k_mag = grid.k_abs

    # Determine k range
    if k_max is None:
        k_max_val = jnp.pi / min(grid.dx, grid.dy, grid.dz)  # Nyquist
    else:
        k_max_val = k_max

    dk = k_max_val / n_bins
    k_centers = (jnp.arange(n_bins) + 0.5) * dk

    # Bin S(k) by |k|
    bin_indices = jnp.floor(k_mag / dk).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

    flat_bins = bin_indices.ravel()
    flat_S = S_k_3d.ravel()

    S_sum = jnp.zeros(n_bins, dtype=jnp.float64).at[flat_bins].add(flat_S)
    counts = jnp.zeros(n_bins, dtype=jnp.float64).at[flat_bins].add(
        jnp.ones_like(flat_S)
    )

    S_func_radial = jnp.where(counts > 0, S_sum / counts, 1.0)

    # PY reference S(k) at the same k values
    S_py_radial = py_structure_factor(k_centers, eta, sigma)

    # Exclude k ~ 0 bin (k_centers[0] ~ dk/2 is near zero; S(0) is just
    # the compressibility which is checked elsewhere)
    # Also exclude bins with very few counts (noisy)
    valid_mask = (k_centers > 0.5) & (counts > 0)

    # Squared relative error
    rel_err = jnp.where(
        valid_mask,
        (S_func_radial - S_py_radial)**2 / (S_py_radial**2 + 1e-10),
        0.0,
    )
    n_valid = jnp.maximum(jnp.sum(valid_mask.astype(jnp.float64)), 1.0)

    loss = jnp.sum(rel_err) / n_valid
    return loss
