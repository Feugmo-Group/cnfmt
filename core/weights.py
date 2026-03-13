"""
FMT Weight Kernels
==================

Fourier-space weight functions for Fundamental Measure Theory.

For hard spheres of radius R, the weight functions are:
- w₃(r): Heaviside step function (ball indicator)
- w₂(r): Surface delta function
- w₁(r) = w₂(r) / (4πR)
- w₀(r) = w₂(r) / (4πR²)
- wᵥ₂(r): Vector weight (gradient of w₃)
- wᵥ₁(r) = wᵥ₂(r) / (4πR)

In Fourier space:
    ŵ₃(k) = (4π/k³)[sin(kR) - kR·cos(kR)]
    ŵ₂(k) = (4πR/k)·sin(kR)

Reference
---------
Y. Rosenfeld, Phys. Rev. Lett. 63, 980 (1989)
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from cnfmt.core.grid import Grid


class FMTKernels(eqx.Module):
    """
    Fourier-space FMT weight kernels for hard-sphere functional.
    
    Parameters
    ----------
    grid : Grid
        Computational grid
    R : float
        Hard sphere radius (typically σ/2 where σ is diameter)
    
    Attributes
    ----------
    w3_hat : Array
        Fourier transform of volume weight (ball indicator)
    w2_hat : Array
        Fourier transform of surface weight
    w1_hat, w0_hat : Array
        Scaled surface weights
    wv2_hat, wv1_hat : Array
        Vector weights (3, nx, ny, nz)
    T_hat : Array
        Tensor weight (3, 3, nx, ny, nz)
    """
    
    # Scalar weights
    w3_hat: Array
    w2_hat: Array
    w1_hat: Array
    w0_hat: Array
    
    # Vector weights (3, ...)
    wv2_hat: Array
    wv1_hat: Array
    
    # Tensor weight (3, 3, ...)
    T_hat: Array
    
    # Sphere radius
    R: float = eqx.field(static=True)
    
    def __init__(self, grid: Grid, R: float):
        """
        Initialize FMT weight kernels.
        
        Parameters
        ----------
        grid : Grid
            Computational grid with wavevector arrays
        R : float
            Hard sphere radius
        """
        self.R = R
        k = grid.k_abs
        eps = 1e-12  # Regularization for k→0 limit
        
        # ──────────────────────────────────────────────────────────
        # Scalar weights
        # ──────────────────────────────────────────────────────────
        
        # w₃: Ball indicator (Heaviside)
        # ŵ₃(k) = (4π/k³)[sin(kR) - kR·cos(kR)]
        # k→0 limit: (4/3)πR³
        self.w3_hat = jnp.where(
            k < eps,
            (4.0 / 3.0) * jnp.pi * R**3,
            4.0 * jnp.pi * (jnp.sin(k * R) - k * R * jnp.cos(k * R)) / (k**3 + eps)
        )
        
        # w₂: Surface delta
        # ŵ₂(k) = (4πR/k)·sin(kR)
        # k→0 limit: 4πR²
        self.w2_hat = jnp.where(
            k < eps,
            4.0 * jnp.pi * R**2,
            4.0 * jnp.pi * R * jnp.sin(k * R) / (k + eps)
        )
        
        # Scaled weights
        self.w1_hat = self.w2_hat / (4.0 * jnp.pi * R)
        self.w0_hat = self.w2_hat / (4.0 * jnp.pi * R**2)
        
        # ──────────────────────────────────────────────────────────
        # Vector weights
        # ──────────────────────────────────────────────────────────
        
        # Unit wavevector components
        k_unit_x = jnp.where(k < eps, 0.0, grid.Kx / k)
        k_unit_y = jnp.where(k < eps, 0.0, grid.Ky / k)
        k_unit_z = jnp.where(k < eps, 0.0, grid.Kz / k)
        
        # wᵥ₂ = -∇w₃ → ŵᵥ₂ = -ik·ŵ₃ (but we use convention with w₂)
        self.wv2_hat = jnp.stack([
            1j * k_unit_x * self.w2_hat,
            1j * k_unit_y * self.w2_hat,
            1j * k_unit_z * self.w2_hat
        ], axis=0)
        
        self.wv1_hat = self.wv2_hat / (4.0 * jnp.pi * R)
        
        # ──────────────────────────────────────────────────────────
        # Tensor weight
        # ──────────────────────────────────────────────────────────
        
        # T = (k⊗k)/|k|² · ŵ₂
        k_stack = jnp.stack([grid.Kx, grid.Ky, grid.Kz], axis=0)
        k2_safe = jnp.where(k < eps, 1.0, k**2)
        self.T_hat = jnp.einsum('i...,j...->ij...', k_stack, k_stack) / k2_safe * self.w2_hat
    
    def __repr__(self) -> str:
        return f"FMTKernels(R={self.R})"
