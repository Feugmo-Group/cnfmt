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
