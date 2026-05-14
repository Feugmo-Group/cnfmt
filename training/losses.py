"""
Loss Functions
==============

Loss functions for training the conditional neural functional.

Bulk Loss
---------
Measures deviation from Carnahan-Starling thermodynamics:
- EOS: (Z_LK - Z_CS)² / Z_CS²
- Chemical potential: (μ_LK - μ_CS)² / μ_CS²  
- Compressibility: (log χ_LK - log χ_RF)²

DFT Loss
--------
Measures sum rule violations:
- δμ²: Chemical potential deviation squared
- δχ²: Compressibility deviation squared
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Dict, Tuple, Any

from core.thermodynamics import BulkThermodynamics
from neural.network import ConditionalNetwork
from .config import TrainingConfig


def compute_bulk_loss_single(network: ConditionalNetwork, eta: float,
                             config: TrainingConfig) -> Tuple[float, Dict]:
    """
    Compute loss for single packing fraction.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Neural network predicting (A, B)
    eta : float
        Packing fraction
    config : TrainingConfig
        Training configuration
    
    Returns
    -------
    loss : float
        Total loss for this eta
    info : dict
        Individual loss components and parameters
    """
    A, B = network.from_eta(eta)
    
    # ──────────────────────────────────────────────────────────
    # Equation of state loss
    # ──────────────────────────────────────────────────────────
    Z_lut = BulkThermodynamics.Z_lutsko(eta, A, B)
    Z_cs = BulkThermodynamics.Z_CS(eta)
    loss_Z = ((Z_lut - Z_cs) / (Z_cs + 1e-6))**2
    
    # ──────────────────────────────────────────────────────────
    # Chemical potential loss
    # ──────────────────────────────────────────────────────────
    mu_lut = BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B)
    mu_cs = BulkThermodynamics.mu_ex_CS(eta)
    loss_mu = ((mu_lut - mu_cs) / (jnp.abs(mu_cs) + 0.1))**2
    
    # ──────────────────────────────────────────────────────────
    # Compressibility loss (log scale for stability)
    # ──────────────────────────────────────────────────────────
    chi_lut = BulkThermodynamics.chi_T_bulk_lutsko(eta, A, B)
    chi_RF = BulkThermodynamics.chi_T_RF(eta)
    log_chi_lut = jnp.log(jnp.maximum(chi_lut, 1e-10))
    log_chi_RF = jnp.log(jnp.maximum(chi_RF, 1e-10))
    loss_chi = (log_chi_lut - log_chi_RF)**2
    
    # Weighted sum
    loss = (config.weight_Z * loss_Z + 
            config.weight_mu * loss_mu + 
            config.weight_chi * loss_chi)
    
    return loss, {
        'loss_Z': loss_Z, 'loss_mu': loss_mu, 'loss_chi': loss_chi,
        'A': A, 'B': B, 'Z_lut': Z_lut, 'Z_cs': Z_cs
    }


def compute_bulk_loss(network: ConditionalNetwork, eta_values: Array,
                      config: TrainingConfig) -> float:
    """
    Compute total bulk loss with regularization.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Neural network
    eta_values : Array
        Training packing fractions
    config : TrainingConfig
        Training configuration
    
    Returns
    -------
    total_loss : float
        Combined thermodynamic + regularization loss
    """
    total_loss = 0.0
    n_eta = len(eta_values)
    
    # ──────────────────────────────────────────────────────────
    # Thermodynamic losses
    # ──────────────────────────────────────────────────────────
    for eta in eta_values:
        loss, _ = compute_bulk_loss_single(network, eta, config)
        total_loss += loss
    
    total_loss /= n_eta
    
    # ──────────────────────────────────────────────────────────
    # Smoothness regularization
    # ──────────────────────────────────────────────────────────
    eta_sorted = jnp.sort(jnp.array(eta_values))
    smooth_loss = 0.0
    
    for i in range(len(eta_sorted) - 1):
        A1, B1 = network.from_eta(eta_sorted[i])
        A2, B2 = network.from_eta(eta_sorted[i + 1])
        deta = eta_sorted[i + 1] - eta_sorted[i]
        
        dA_deta = (A2 - A1) / (deta + 1e-6)
        dB_deta = (B2 - B1) / (deta + 1e-6)
        smooth_loss += dA_deta**2 + dB_deta**2
    
    smooth_loss /= max(len(eta_sorted) - 1, 1)
    
    # ──────────────────────────────────────────────────────────
    # Soft constraint penalty: -4 ≤ C ≤ 1
    # ──────────────────────────────────────────────────────────
    constraint_loss = 0.0
    for eta in eta_values:
        C = network.constraint_value(eta)
        constraint_loss += jax.nn.relu(-C - 4.0)**2  # Penalize C < -4
        constraint_loss += jax.nn.relu(C - 1.0)**2   # Penalize C > 1
    constraint_loss /= n_eta
    
    # Total loss
    total_loss += config.weight_smooth * smooth_loss
    total_loss += config.weight_constraint * constraint_loss
    
    return total_loss


def compute_dft_loss(delta_mu: float, delta_chi: float) -> float:
    """
    Compute DFT loss from sum rule deviations (Gül et al. Eq. 28).
    
    Parameters
    ----------
    delta_mu : float
        Relative chemical potential deviation
    delta_chi : float
        Relative compressibility deviation
    
    Returns
    -------
    loss : float
        Combined squared deviation
    """
    return delta_mu**2 + delta_chi**2


def compute_contact_loss_single(network: ConditionalNetwork, eta: float,
                                config: TrainingConfig) -> Tuple[float, Dict]:
    """
    Compute contact density loss for single packing fraction.
    
    This is the test-particle approach used by Gül et al. to obtain
    their optimized parameters A=1.3, B=-1.0.
    
    The contact density at a hard wall is computed from the DFT pressure
    via the contact theorem: ρ(R⁺) = βP = ρ_bulk × Z
    
    Parameters
    ----------
    network : ConditionalNetwork
        Neural network predicting (A, B)
    eta : float
        Packing fraction
    config : TrainingConfig
        Training configuration
    
    Returns
    -------
    loss : float
        Contact density loss
    info : dict
        Contact density values
    """
    A, B = network.from_eta(eta)
    
    # Bulk density
    rho_bulk = eta / ((4.0/3.0) * jnp.pi * 0.5**3)  # σ = 1
    
    # Contact density from Lutsko functional (via contact theorem)
    Z_lut = BulkThermodynamics.Z_lutsko(eta, A, B)
    contact_dft = rho_bulk * Z_lut
    
    # Exact contact density (from Carnahan-Starling)
    Z_cs = BulkThermodynamics.Z_CS(eta)
    contact_exact = rho_bulk * Z_cs
    
    # Relative error
    loss = ((contact_dft - contact_exact) / contact_exact)**2
    
    return loss, {
        'contact_dft': contact_dft,
        'contact_exact': contact_exact,
        'A': A, 'B': B
    }


def compute_contact_loss(network: ConditionalNetwork, eta_values: Array,
                         config: TrainingConfig) -> float:
    """
    Compute total contact density loss.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Neural network
    eta_values : Array
        Training packing fractions
    config : TrainingConfig
        Training configuration
    
    Returns
    -------
    total_loss : float
        Sum of contact density losses
    """
    total_loss = 0.0
    n_eta = len(eta_values)
    
    for eta in eta_values:
        loss, _ = compute_contact_loss_single(network, eta, config)
        total_loss += loss
    
    return total_loss / n_eta


def compute_combined_loss(network: ConditionalNetwork, eta_values: Array,
                          config: TrainingConfig,
                          w_bulk: float = 1.0,
                          w_contact: float = 1.0) -> float:
    """
    Compute combined loss: bulk EOS + contact density.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Neural network
    eta_values : Array
        Training packing fractions
    config : TrainingConfig
        Training configuration
    w_bulk : float
        Weight for bulk EOS loss
    w_contact : float
        Weight for contact density loss
    
    Returns
    -------
    total_loss : float
        Weighted sum of losses
    """
    bulk_loss = compute_bulk_loss(network, eta_values, config)
    contact_loss = compute_contact_loss(network, eta_values, config)
    
    return w_bulk * bulk_loss + w_contact * contact_loss


# ═══════════════════════════════════════════════════════════════════
# Nonlocal Functional Losses
# ═══════════════════════════════════════════════════════════════════

from core.grid import Grid
from constraints.sum_rules import contact_sum_rule_loss
from constraints.noether import translational_invariance_loss
from constraints.scaled_particle import (
    low_density_limit_loss,
    close_packing_limit_loss,
    spt_exact_relations_loss,
    positivity_loss,
)


def compute_nonlocal_bulk_loss(functional, eta_values: Array,
                               grid: Grid, config: TrainingConfig) -> float:
    """
    Compute bulk thermodynamic loss for a NonlocalLutskoFunctional.

    Like compute_bulk_loss but uses functional.bulk_parameters(eta) to
    obtain (A, B) from the nonlocal functional's neural network applied
    to a uniform density field, rather than from a standalone network.

    For each eta the loss penalises deviations from Carnahan-Starling in:
    - Equation of state Z
    - Excess chemical potential mu_ex
    - Isothermal compressibility chi_T (log scale)

    A smoothness regularisation on A(eta) and B(eta) is included.

    Parameters
    ----------
    functional : NonlocalLutskoFunctional
        Nonlocal functional whose network predicts (A, B)
    eta_values : Array
        Training packing fractions
    grid : Grid
        Computational grid (needed by functional.bulk_parameters)
    config : TrainingConfig
        Training configuration with loss weights

    Returns
    -------
    total_loss : float
        Combined thermodynamic + smoothness loss
    """
    n_eta = len(eta_values)
    total_loss = 0.0

    # Collect A, B at each eta for smoothness regularisation
    A_vals = []
    B_vals = []

    for eta in eta_values:
        A, B = functional.bulk_parameters(eta)
        A_vals.append(A)
        B_vals.append(B)

        # ── Equation of state loss ──
        Z_lut = BulkThermodynamics.Z_lutsko(eta, A, B)
        Z_cs = BulkThermodynamics.Z_CS(eta)
        loss_Z = ((Z_lut - Z_cs) / (Z_cs + 1e-6))**2

        # ── Chemical potential loss ──
        mu_lut = BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B)
        mu_cs = BulkThermodynamics.mu_ex_CS(eta)
        loss_mu = ((mu_lut - mu_cs) / (jnp.abs(mu_cs) + 0.1))**2

        # ── Compressibility loss (log scale) ──
        chi_lut = BulkThermodynamics.chi_T_bulk_lutsko(eta, A, B)
        chi_RF = BulkThermodynamics.chi_T_RF(eta)
        log_chi_lut = jnp.log(jnp.maximum(chi_lut, 1e-10))
        log_chi_RF = jnp.log(jnp.maximum(chi_RF, 1e-10))
        loss_chi = (log_chi_lut - log_chi_RF)**2

        total_loss += (config.weight_Z * loss_Z +
                       config.weight_mu * loss_mu +
                       config.weight_chi * loss_chi)

    total_loss /= n_eta

    # ── Smoothness regularisation on A(eta), B(eta) curve ──
    eta_sorted = jnp.sort(jnp.array(eta_values))
    smooth_loss = 0.0

    # Re-evaluate at sorted etas for consistent finite differences
    A_sorted = []
    B_sorted = []
    for eta in eta_sorted:
        A, B = functional.bulk_parameters(float(eta))
        A_sorted.append(A)
        B_sorted.append(B)

    for i in range(len(eta_sorted) - 1):
        deta = float(eta_sorted[i + 1] - eta_sorted[i])
        dA_deta = (A_sorted[i + 1] - A_sorted[i]) / (deta + 1e-6)
        dB_deta = (B_sorted[i + 1] - B_sorted[i]) / (deta + 1e-6)
        smooth_loss += dA_deta**2 + dB_deta**2

    smooth_loss /= max(len(eta_sorted) - 1, 1)
    total_loss += config.weight_smooth * smooth_loss

    return total_loss


def compute_nonlocal_constraint_loss(
    functional, rho_profile: Array, z_grid: Array,
    eta: Array, grid: Grid, config: TrainingConfig,
    key: Array = None,
) -> Tuple[float, Dict]:
    """
    Combine all physics constraint losses for the nonlocal functional.

    Aggregates individual constraint losses with configurable weights:
    - Contact sum rule (wall contact density = bulk pressure)
    - SPT low density limit (ideal gas at eta -> 0)
    - SPT close packing limit (monotonic Z at high eta)
    - SPT exact relations (second virial, mu slope)
    - Positivity of excess free energy
    - Noether translational invariance

    Parameters
    ----------
    functional : NonlocalLutskoFunctional
        Nonlocal density functional to evaluate constraints on
    rho_profile : Array, shape (N,)
        1D density profile rho(z) (solved at the given eta)
    z_grid : Array, shape (N,)
        Spatial coordinates for the 1D profile
    eta : Array (scalar)
        Bulk packing fraction for this profile
    grid : Grid
        3D computational grid (needed for translational invariance)
    config : TrainingConfig
        Training configuration with constraint weights
    key : jax.random.PRNGKey, optional
        PRNG key for Noether translational invariance test.
        If None, the Noether loss is skipped.

    Returns
    -------
    total_loss : float
        Weighted sum of all constraint losses
    info : dict
        Individual loss components keyed by name
    """
    info = {}

    # ── Contact sum rule ──
    loss_contact = contact_sum_rule_loss(rho_profile, z_grid, eta)
    info['loss_contact'] = loss_contact

    # ── SPT low density limit ──
    # Use the nonlocal functional's bulk_parameters as a network-like
    # object: we create a thin wrapper that exposes from_eta
    class _BulkParamWrapper:
        """Adapter so SPT losses can call .from_eta(eta)."""
        def __init__(self, func):
            self._func = func
        def from_eta(self, eta_val):
            return self._func.bulk_parameters(eta_val)
    wrapper = _BulkParamWrapper(functional)

    loss_low_density = low_density_limit_loss(wrapper)
    info['loss_low_density'] = loss_low_density

    # ── SPT close packing limit ──
    loss_close_packing = close_packing_limit_loss(wrapper)
    info['loss_close_packing'] = loss_close_packing

    # ── SPT exact relations ──
    loss_spt_exact = spt_exact_relations_loss(wrapper)
    info['loss_spt_exact'] = loss_spt_exact

    # ── Positivity ──
    # Evaluate on a uniform density field at this eta
    rho_bulk = 6.0 * eta / jnp.pi
    rho_uniform = jnp.ones((grid.nx, grid.ny, grid.nz)) * rho_bulk
    loss_pos = positivity_loss(functional, rho_uniform, grid)
    info['loss_positivity'] = loss_pos

    # ── Noether translational invariance ──
    if key is not None:
        loss_noether = translational_invariance_loss(
            functional, rho_uniform, grid, n_shifts=3, key=key
        )
    else:
        loss_noether = 0.0
    info['loss_noether'] = loss_noether

    # ── Weighted combination ──
    total_loss = (
        config.weight_contact * loss_contact
        + config.weight_spt * (loss_low_density + loss_close_packing
                               + loss_spt_exact)
        + config.weight_positivity * loss_pos
        + config.weight_noether * loss_noether
    )

    info['total_constraint_loss'] = total_loss
    return total_loss, info


def compute_nonlocal_combined_loss(
    functional, eta_values: Array, grid: Grid, config: TrainingConfig,
    rho_profiles: Array = None, z_grids: Array = None,
    key: Array = None,
) -> float:
    """
    Master loss for training the nonlocal functional.

    Combines bulk thermodynamic matching and physics constraints into
    a single scalar objective suitable for gradient-based optimisation.

    Parameters
    ----------
    functional : NonlocalLutskoFunctional
        Nonlocal density functional to train
    eta_values : Array
        Training packing fractions
    grid : Grid
        Computational grid
    config : TrainingConfig
        Training configuration with all loss weights
    rho_profiles : list of Array or None, optional
        Solved 1D density profiles for each eta in eta_values.
        If None, only the bulk loss is computed (no constraint losses).
    z_grids : list of Array or None, optional
        Spatial grids for each profile.  Must have same length as
        rho_profiles.
    key : jax.random.PRNGKey, optional
        PRNG key for stochastic constraint evaluations (Noether).
        If None, Noether loss is skipped.

    Returns
    -------
    total_loss : float
        Combined objective for the optimizer
    """
    # ── Bulk thermodynamic loss ──
    bulk_loss = compute_nonlocal_bulk_loss(functional, eta_values, grid, config)

    # ── Constraint losses (only if profiles are provided) ──
    constraint_loss = 0.0
    if rho_profiles is not None and z_grids is not None:
        n_profiles = len(rho_profiles)
        for i in range(n_profiles):
            # Split key for each profile so Noether shifts are independent
            if key is not None:
                subkey = jax.random.fold_in(key, i)
            else:
                subkey = None

            eta_i = eta_values[i] if i < len(eta_values) else eta_values[-1]
            c_loss, _ = compute_nonlocal_constraint_loss(
                functional, rho_profiles[i], z_grids[i],
                eta_i, grid, config, key=subkey,
            )
            constraint_loss += c_loss

        constraint_loss /= max(n_profiles, 1)

    total_loss = bulk_loss + constraint_loss
    return total_loss
