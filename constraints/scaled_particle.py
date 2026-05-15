"""
Scaled-Particle Theory Limit Conditions
========================================

Physics constraint losses enforcing exact limiting behaviors of the
hard-sphere excess free energy functional.

Scaled-particle theory (SPT) provides exact results in several limits:

Low density (η → 0):
    - Excess free energy vanishes: F_exc → 0
    - Compressibility factor approaches ideal gas: Z → 1
    - Excess chemical potential vanishes: βμ_ex → 0

Close packing (η → η_cp):
    - Pressure diverges: P → ∞
    - Z must increase monotonically

Exact SPT relations:
    - dβμ_ex/dη |_{η=0} = 8  (for hard spheres with σ = 1)
    - Second virial coefficient B₂ = (2/3)πσ³

Positivity:
    - F_exc[ρ] ≥ 0 for any physical density field

All losses are JAX-differentiable and follow the conventions in
``training.losses``.

Reference
---------
Reiss, Frisch, Lebowitz, J. Chem. Phys. 31, 369 (1959).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Optional

from core.thermodynamics import BulkThermodynamics


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════

# Sphere volume for σ = 1: v = (4/3)π(σ/2)³ = π/6
_V_SPHERE = jnp.pi / 6.0

# Exact second virial coefficient for hard spheres (σ = 1):
# B₂ = (2/3)πσ³ = 2π/3
_B2_EXACT = 2.0 * jnp.pi / 3.0

# Exact slope dβμ_ex/dη at η = 0 for hard spheres (σ = 1):
# From SPT: dβμ_ex/dη|₀ = 8
_DMU_DETA_EXACT = 8.0


# ══════════════════════════════════════════════════════════════════════
# Low-density limit
# ══════════════════════════════════════════════════════════════════════

def low_density_limit_loss(
    network,
    eta_values: Optional[Array] = None,
) -> float:
    """
    Enforce vanishing excess free energy and ideal-gas EOS at η → 0.

    At low density the hard-sphere fluid must reduce to an ideal gas:

        F_exc / (N η) → 0   as η → 0
        Z(η) → 1            as η → 0

    The loss is evaluated at several small η values and returns the
    sum of squared, scale-free deviations.

    Parameters
    ----------
    network : ConditionalNetwork
        Neural network predicting (A, B) from η.
    eta_values : array-like, optional
        Packing fractions to test.  Default ``[0.001, 0.005, 0.01]``.

    Returns
    -------
    loss : float
        Combined low-density constraint loss (JAX scalar).
    """
    if eta_values is None:
        eta_values = jnp.array([0.001, 0.005, 0.01])
    else:
        eta_values = jnp.asarray(eta_values)

    loss = 0.0

    for i in range(len(eta_values)):
        eta = eta_values[i]
        A, B = network.from_eta(eta)

        # ── Excess free energy per particle, normalized by η ──
        # For uniform fluid: βF_exc/N = βμ_ex - Z + 1  (thermodynamic identity)
        # At low η this should vanish.  We use μ_ex as proxy since
        # F_exc/N ~ μ_ex at leading order.
        mu_ex = BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B)
        # μ_ex → 0 as η → 0 (excess chemical potential vanishes at zero density).
        # NOTE: μ_ex/η → 8 always (exact second virial coefficient — a physics
        # constant, not something to minimize). Using (μ_ex/η)² gives a constant
        # ~64 loss with near-zero gradient, masking all other training signal.
        loss += mu_ex ** 2

        # ── Compressibility factor → 1 ──
        Z = BulkThermodynamics.Z_lutsko(eta, A, B)
        loss += (Z - 1.0) ** 2

    return loss


# ══════════════════════════════════════════════════════════════════════
# Close-packing limit
# ══════════════════════════════════════════════════════════════════════

def close_packing_limit_loss(
    network,
    eta_max: float = 0.52,
) -> float:
    """
    Enforce monotonically increasing Z(η) approaching close packing.

    Near the fluid close-packing limit (η ≈ 0.494 freezing, practical
    upper bound ~0.52) the pressure must diverge.  A necessary condition
    is that Z(η) is strictly monotonically increasing.

    The loss penalizes any *decrease* in Z between successive η values:

        L = Σ_i relu(Z(η_i) − Z(η_{i+1}))²

    It also penalizes non-divergent chemical potential by requiring
    μ_ex to increase monotonically in the same range.

    Parameters
    ----------
    network : ConditionalNetwork
        Neural network predicting (A, B) from η.
    eta_max : float
        Upper bound of the test range.  Default 0.52.

    Returns
    -------
    loss : float
        Monotonicity violation loss (JAX scalar).
    """
    eta_values = jnp.arange(0.40, eta_max + 0.01, 0.02)

    # Pre-compute Z and μ_ex at each η
    Z_vals = []
    mu_vals = []
    for i in range(len(eta_values)):
        eta = eta_values[i]
        A, B = network.from_eta(eta)
        Z_vals.append(BulkThermodynamics.Z_lutsko(eta, A, B))
        mu_vals.append(BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B))

    loss = 0.0

    # Penalize non-monotonic Z
    for i in range(len(Z_vals) - 1):
        loss += jax.nn.relu(Z_vals[i] - Z_vals[i + 1]) ** 2

    # Penalize non-monotonic μ_ex
    for i in range(len(mu_vals) - 1):
        loss += jax.nn.relu(mu_vals[i] - mu_vals[i + 1]) ** 2

    return loss


# ══════════════════════════════════════════════════════════════════════
# Exact SPT relations
# ══════════════════════════════════════════════════════════════════════

def spt_exact_relations_loss(
    network,
    eta_values: Optional[Array] = None,
) -> float:
    """
    Enforce exact scaled-particle theory relations at low density.

    Three exact conditions for hard spheres (σ = 1):

    1. βμ_ex(η=0) = 0
       The excess chemical potential must vanish at zero density.

    2. dβμ_ex/dη |_{η→0} = 8
       The initial slope is fixed by the second virial coefficient.
       From the virial expansion: βμ_ex = B₂ρ + ... = (2π/3)(6η/π) + ...
       so dβμ_ex/dη |₀ = 4·B₂/v_sphere = 8.

    3. B₂ = (2/3)πσ³
       The second virial coefficient must be exact.  For the Lutsko
       functional this is guaranteed when Z(η→0) has the correct
       linear term: Z ≈ 1 + 4η + ...  (since B₂ = 4v_sphere).
       We check via the numerical Z slope at small η.

    Parameters
    ----------
    network : ConditionalNetwork
        Neural network predicting (A, B) from η.
    eta_values : array-like, optional
        Packing fractions for finite-difference derivatives.
        Default ``[0.001, 0.002]`` (small η for numerical derivative).

    Returns
    -------
    loss : float
        Squared deviations from exact SPT values (JAX scalar).
    """
    if eta_values is None:
        eta_values = jnp.array([0.001, 0.002])
    else:
        eta_values = jnp.asarray(eta_values)

    # Use the two smallest η for finite-difference derivatives
    eta_lo = eta_values[0]
    eta_hi = eta_values[jnp.minimum(1, len(eta_values) - 1)]

    A_lo, B_lo = network.from_eta(eta_lo)
    A_hi, B_hi = network.from_eta(eta_hi)

    # ── Condition 1: βμ_ex(η→0) = 0 ──
    mu_lo = BulkThermodynamics.mu_ex_bulk_lutsko(eta_lo, A_lo, B_lo)
    loss_mu_zero = mu_lo ** 2

    # ── Condition 2: dβμ_ex/dη |_{η→0} = 8 ──
    mu_hi = BulkThermodynamics.mu_ex_bulk_lutsko(eta_hi, A_hi, B_hi)
    dmu_deta = (mu_hi - mu_lo) / (eta_hi - eta_lo + 1e-12)
    loss_dmu = (dmu_deta - _DMU_DETA_EXACT) ** 2

    # ── Condition 3: correct B₂ via Z slope ──
    # Z ≈ 1 + B₂/v_sphere · η = 1 + 4η  for hard spheres
    # So dZ/dη|₀ = 4  (since B₂ = 4 v_sphere)
    Z_lo = BulkThermodynamics.Z_lutsko(eta_lo, A_lo, B_lo)
    Z_hi = BulkThermodynamics.Z_lutsko(eta_hi, A_hi, B_hi)
    dZ_deta = (Z_hi - Z_lo) / (eta_hi - eta_lo + 1e-12)
    # Exact: dZ/dη|₀ = 4
    loss_B2 = (dZ_deta - 4.0) ** 2

    return loss_mu_zero + loss_dmu + loss_B2


# ══════════════════════════════════════════════════════════════════════
# Positivity of excess free energy
# ══════════════════════════════════════════════════════════════════════

def positivity_loss(
    functional,
    rho: Array,
    grid,
) -> float:
    """
    Enforce non-negative excess free energy: F_exc[ρ] ≥ 0.

    The hard-sphere excess free energy is strictly non-negative for
    any physical density field (ρ ≥ 0).  This follows from the
    convexity of the free energy and F_exc[0] = 0.

    The loss penalizes negative values via a one-sided quadratic:

        L = relu(−F_exc)²

    Parameters
    ----------
    functional : NonlocalLutskoFunctional or similar
        Functional with an ``excess_free_energy(rho)`` method.
    rho : Array
        Density field, shape ``(nx, ny, nz)``.
    grid : Grid
        Computational grid (used only for context; the functional
        already contains its own grid reference).

    Returns
    -------
    loss : float
        Positivity violation loss (JAX scalar, zero when F_exc ≥ 0).
    """
    F_exc = functional.excess_free_energy(rho)
    return jax.nn.relu(-F_exc) ** 2
