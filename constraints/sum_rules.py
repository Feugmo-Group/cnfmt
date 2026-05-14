"""
Sum Rule Constraint Losses
==========================

Physics-based loss functions enforcing exact sum rules for hard-sphere DFT.

These constraints are model-independent exact relations that any correct
density functional must satisfy. They can be used as regularisation losses
during neural functional training.

Contact Sum Rule (Henderson, Blum, Lebowitz 1979)
--------------------------------------------------
The density at contact with a hard wall equals the bulk pressure:

    rho(sigma/2) = beta * P = rho_bulk * Z_CS(eta)

This is exact for any hard-wall external potential and follows from the
first member of the YBG hierarchy (wall virial route).

Compressibility Sum Rule (Hansen & McDonald)
--------------------------------------------
Relates the k=0 limit of the two-body direct correlation function to
the isothermal compressibility:

    1 - rho * c_hat_2(k=0) = 1 / chi_T

where c_hat_2(k) is the Fourier transform of c_2(r).  Any self-consistent
functional must satisfy this identity linking its second functional
derivative to the bulk equation of state.

Gibbs Adsorption (Gibbs 1878)
-----------------------------
Connects the surface excess to the derivative of surface tension with
respect to chemical potential:

    Gamma = -d(gamma) / d(mu)

where Gamma = integral [rho(z) - rho_bulk] dz is the surface excess
adsorption.  This thermodynamic identity is useful for checking
consistency across a family of density profiles at neighbouring state
points.

References
----------
Henderson D, Blum L, Lebowitz JL, J. Electroanal. Chem. 102, 315 (1979).
Hansen JP, McDonald IR, Theory of Simple Liquids, 4th ed. (Academic Press, 2013).
Gibbs JW, Trans. Connecticut Acad. Arts Sci. 3, 108 (1878).
"""

import jax.numpy as jnp
from jaxtyping import Array

from core.thermodynamics import BulkThermodynamics


# ═══════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════

def compute_contact_density(rho_profile: Array, z_grid: Array,
                            sigma: float = 1.0) -> Array:
    """
    Interpolate the density profile at the wall contact distance z = sigma/2.

    Uses linear interpolation between the two grid points straddling
    z = sigma/2, which is JAX-differentiable.

    Parameters
    ----------
    rho_profile : Array, shape (N,)
        1D density profile rho(z).
    z_grid : Array, shape (N,)
        Spatial coordinates corresponding to ``rho_profile``.
    sigma : float
        Hard-sphere diameter (default 1.0).

    Returns
    -------
    rho_contact : Array (scalar)
        Density at the contact distance z = sigma/2.
    """
    z_contact = sigma / 2.0

    # Distance of each grid point from the contact position
    dz = z_grid - z_contact

    # Find the last grid point at or before z_contact.
    # Use a soft argmax via large negative values for points past contact.
    mask_before = jnp.where(dz <= 0.0, dz, -jnp.inf)
    idx_lo = jnp.argmax(mask_before)  # closest point <= z_contact

    # Guard: clamp so idx_hi stays in bounds
    idx_hi = jnp.minimum(idx_lo + 1, len(z_grid) - 1)

    z_lo = z_grid[idx_lo]
    z_hi = z_grid[idx_hi]
    rho_lo = rho_profile[idx_lo]
    rho_hi = rho_profile[idx_hi]

    # Linear interpolation weight (safe division)
    dz_span = jnp.maximum(z_hi - z_lo, 1e-14)
    t = (z_contact - z_lo) / dz_span

    rho_contact = rho_lo + t * (rho_hi - rho_lo)
    return rho_contact


# ═══════════════════════════════════════════════════════════════════
# 1. CONTACT SUM RULE
# ═══════════════════════════════════════════════════════════════════

def contact_sum_rule_loss(rho_profile: Array, z_grid: Array,
                          eta: Array, sigma: float = 1.0) -> Array:
    """
    Contact-theorem loss: rho(sigma/2) should equal rho_bulk * Z_CS(eta).

    The exact wall contact density for a hard-sphere fluid at a planar
    hard wall is given by the contact theorem:

        rho(sigma/2) = beta P = rho_bulk * Z_CS(eta)

    Parameters
    ----------
    rho_profile : Array, shape (N,)
        Solved 1D density profile rho(z).
    z_grid : Array, shape (N,)
        Spatial grid coordinates.
    eta : Array (scalar)
        Bulk packing fraction.
    sigma : float
        Hard-sphere diameter (default 1.0).

    Returns
    -------
    loss : Array (scalar)
        Squared relative error: ((rho_contact - rho_exact) / rho_exact)^2.
    """
    # Bulk number density from packing fraction: eta = (pi/6) * rho * sigma^3
    rho_bulk = 6.0 * eta / (jnp.pi * sigma**3)

    # Exact contact density from Carnahan-Starling
    rho_exact = rho_bulk * BulkThermodynamics.Z_CS(eta)

    # Measured contact density from the profile
    rho_contact = compute_contact_density(rho_profile, z_grid, sigma)

    # Squared relative error
    loss = ((rho_contact - rho_exact) / (rho_exact + 1e-10))**2
    return loss


# ═══════════════════════════════════════════════════════════════════
# 2. COMPRESSIBILITY SUM RULE
# ═══════════════════════════════════════════════════════════════════

def compressibility_sum_rule_loss(c2_k: Array, rho_bulk: Array,
                                  chi_target: Array) -> Array:
    """
    Compressibility sum rule loss: 1 - rho * c_hat_2(k=0) = 1 / chi_T.

    For a correct functional the k -> 0 limit of the Fourier-transformed
    two-body direct correlation function is related to the isothermal
    compressibility by

        1 - rho_bulk * c_hat_2(k=0)  =  1 / chi_T

    where chi_T = (1/rho)(d rho / d mu) is the reduced isothermal
    compressibility (dimensionless, in units of beta / rho).

    Parameters
    ----------
    c2_k : Array, shape (N,)
        Fourier transform of the two-body direct correlation function
        c_2(|r - r'|).  The k=0 component is ``c2_k[0]``.
    rho_bulk : Array (scalar)
        Bulk number density.
    chi_target : Array (scalar)
        Target isothermal compressibility, e.g. from
        ``BulkThermodynamics.chi_T_CS(eta)``.

    Returns
    -------
    loss : Array (scalar)
        Squared relative error of the sum-rule identity.
    """
    # LHS of the sum rule: evaluated at k=0
    lhs = 1.0 - rho_bulk * c2_k[0]

    # RHS: inverse compressibility
    rhs = 1.0 / (chi_target + 1e-14)

    # Squared relative error (normalise by rhs to make scale-free)
    loss = ((lhs - rhs) / (jnp.abs(rhs) + 1e-10))**2
    return loss


# ═══════════════════════════════════════════════════════════════════
# 3. GIBBS ADSORPTION
# ═══════════════════════════════════════════════════════════════════

def gibbs_adsorption_loss(surface_excess: Array, surface_tensions: Array,
                          mu_values: Array) -> Array:
    """
    Gibbs adsorption loss: Gamma = -d(gamma)/d(mu).

    The Gibbs adsorption equation for a one-component fluid at a planar
    wall reads

        Gamma = -d(gamma) / d(mu)

    where

    * Gamma = integral [rho(z) - rho_bulk] dz   (surface excess adsorption)
    * gamma  is the wall-fluid surface tension
    * mu     is the chemical potential

    Given pre-computed arrays of Gamma, gamma, and mu at a sequence of
    nearby packing fractions, this loss penalises violations of the
    identity.

    The numerical derivative d(gamma)/d(mu) is computed via central
    finite differences on interior points (forward/backward at the
    boundaries).

    Parameters
    ----------
    surface_excess : Array, shape (M,)
        Surface excess adsorption Gamma at each state point.
    surface_tensions : Array, shape (M,)
        Surface tension gamma at each state point.
    mu_values : Array, shape (M,)
        Chemical potential mu at each state point.

    Returns
    -------
    loss : Array (scalar)
        Mean squared absolute violation |d(gamma)/d(mu) + Gamma|^2
        averaged over interior points.

    Notes
    -----
    At least three state points (M >= 3) are needed so that central
    differences can be evaluated on at least one interior point.
    """
    M = surface_tensions.shape[0]

    # ── numerical d(gamma)/d(mu) via central differences ──
    # Interior points: central difference
    dmu_fwd = mu_values[2:] - mu_values[:-2]  # (M-2,)
    dgamma_fwd = surface_tensions[2:] - surface_tensions[:-2]
    dgamma_dmu_interior = dgamma_fwd / (dmu_fwd + 1e-14)  # (M-2,)

    # Corresponding Gamma values at interior points
    gamma_interior = surface_excess[1:-1]  # (M-2,)

    # Gibbs relation violation: d(gamma)/d(mu) + Gamma should be zero
    violation = dgamma_dmu_interior + gamma_interior

    # Mean squared violation
    loss = jnp.mean(violation**2)
    return loss
