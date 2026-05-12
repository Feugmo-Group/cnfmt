"""
Feature Extraction
==================

Extracts local structural features from density fields for
conditional parameter prediction.

Features
--------
1. η(r) / 0.5: Normalized local packing fraction
2. |∇ρ| / ρ: Relative gradient magnitude (interface indicator)
3. ∇²ρ / ρ: Relative Laplacian (curvature)
4. n₂ / (4πR²ρ): Normalized surface density
5. Local order parameter: variance of n₂
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from core.grid import Grid
from core.densities import WeightedDensityCalculator


class FeatureExtractor(eqx.Module):
    """
    Extracts local structural features from density field.
    
    Parameters
    ----------
    calculator : WeightedDensityCalculator
        For computing weighted densities
    grid : Grid
        Computational grid
    
    Example
    -------
    >>> extractor = FeatureExtractor(calculator, grid)
    >>> features = extractor(rho)  # Shape (nx, ny, nz, 5)
    """
    
    calculator: WeightedDensityCalculator
    grid: Grid
    
    def __init__(self, calculator: WeightedDensityCalculator, grid: Grid):
        self.calculator = calculator
        self.grid = grid
    
    def __call__(self, rho: Array, n_features: int = 5) -> Array:
        """
        Extract features from density field.
        
        Parameters
        ----------
        rho : Array
            Density field, shape (nx, ny, nz)
        n_features : int
            Number of features to extract (max 5)
        
        Returns
        -------
        features : Array
            Feature array, shape (nx, ny, nz, n_features)
        """
        eps = 1e-10
        rho_safe = jnp.maximum(rho, eps)
        
        # ──────────────────────────────────────────────────────────
        # Weighted densities
        # ──────────────────────────────────────────────────────────
        measures = self.calculator(rho)
        
        # ──────────────────────────────────────────────────────────
        # Spectral gradient: ∇ρ
        # ──────────────────────────────────────────────────────────
        rho_hat = jnp.fft.fftn(rho)
        grad_rho_x = jnp.real(jnp.fft.ifftn(1j * self.grid.Kx * rho_hat))
        grad_rho_y = jnp.real(jnp.fft.ifftn(1j * self.grid.Ky * rho_hat))
        grad_rho_z = jnp.real(jnp.fft.ifftn(1j * self.grid.Kz * rho_hat))
        grad_magnitude = jnp.sqrt(grad_rho_x**2 + grad_rho_y**2 + grad_rho_z**2)
        
        # ──────────────────────────────────────────────────────────
        # Spectral Laplacian: ∇²ρ
        # ──────────────────────────────────────────────────────────
        laplacian_rho = jnp.real(jnp.fft.ifftn(-self.grid.k_sq * rho_hat))
        
        # ──────────────────────────────────────────────────────────
        # Normalized features
        # ──────────────────────────────────────────────────────────
        
        # 1. Normalized packing fraction
        eta_normalized = measures.eta / 0.5
        
        # 2. Relative gradient magnitude (interface indicator)
        grad_normalized = grad_magnitude / (rho_safe + eps)
        
        # 3. Relative Laplacian (curvature)
        laplacian_normalized = laplacian_rho / (rho_safe + eps)
        
        # 4. Normalized surface density
        R = self.calculator.kernels.R
        n2_normalized = measures.n2 / (4 * jnp.pi * R**2 * rho_safe + eps)
        
        # 5. Local order parameter (variance of n2)
        n2_mean = jnp.mean(measures.n2)
        order_param = (measures.n2 - n2_mean)**2 / (n2_mean**2 + eps)
        
        # Stack features
        all_features = jnp.stack([
            eta_normalized,
            grad_normalized,
            laplacian_normalized,
            n2_normalized,
            order_param
        ], axis=-1)
        
        return all_features[..., :n_features]
