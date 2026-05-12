"""
Weighted Density Calculations
=============================

Computes the six weighted densities of FMT via FFT convolution:

    nα(r) = ∫ ρ(r') wα(r - r') dr'

In Fourier space:
    n̂α(k) = ρ̂(k) · ŵα(k)

Weighted Densities
------------------
- n₃ (η): Local packing fraction
- n₂: Surface-weighted density  
- n₁: Line-weighted density
- n₀: Point-weighted density
- nᵥ₂: Vector density
- nᵥ₁: Vector density (scaled)
- T: Tensor density

Derived Quantities
------------------
- nᵥ₁·nᵥ₂: Scalar product
- |nᵥ₂|²: Vector squared
- Tr(T²), Tr(T³): Tensor traces
- nᵥ₂·T·nᵥ₂: Tensor contraction
"""

import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple
from jaxtyping import Array
from .weights import FMTKernels


class WeightedDensities(NamedTuple):
    """
    Container for all weighted densities.
    
    Attributes
    ----------
    eta : Array
        n₃ - local packing fraction
    n0, n1, n2 : Array
        Scalar weighted densities
    nv1, nv2 : Array
        Vector weighted densities, shape (3, nx, ny, nz)
    T : Array
        Tensor weighted density, shape (3, 3, nx, ny, nz)
    nv1_dot_nv2 : Array
        nᵥ₁ · nᵥ₂
    nv2_sq : Array
        |nᵥ₂|²
    T2 : Array
        Tr(T²)
    T3 : Array
        Tr(T³)
    nvTnv : Array
        nᵥ₂ · T · nᵥ₂
    """
    eta: Array          # n₃ - packing fraction
    n0: Array           # n₀
    n1: Array           # n₁
    n2: Array           # n₂
    nv1: Array          # nᵥ₁ (3, ...)
    nv2: Array          # nᵥ₂ (3, ...)
    T: Array            # Tensor (3, 3, ...)
    nv1_dot_nv2: Array  # nᵥ₁ · nᵥ₂
    nv2_sq: Array       # |nᵥ₂|²
    T2: Array           # Tr(T²)
    T3: Array           # Tr(T³)
    nvTnv: Array        # nᵥ₂ · T · nᵥ₂


class WeightedDensityCalculator(eqx.Module):
    """
    Computes weighted densities via FFT convolution.
    
    Parameters
    ----------
    kernels : FMTKernels
        Precomputed FMT weight kernels in Fourier space
    
    Example
    -------
    >>> grid = Grid((32, 32, 32), 12.0)
    >>> kernels = FMTKernels(grid, R=0.5)
    >>> calculator = WeightedDensityCalculator(kernels)
    >>> rho = jnp.ones(grid.shape) * 0.5  # Uniform density
    >>> measures = calculator(rho)
    >>> print(f"Mean packing fraction: {jnp.mean(measures.eta):.4f}")
    """
    
    kernels: FMTKernels
    
    def __init__(self, kernels: FMTKernels):
        """Initialize calculator with precomputed kernels."""
        self.kernels = kernels
    
    def __call__(self, rho: Array) -> WeightedDensities:
        """
        Compute all weighted densities from density field.
        
        Parameters
        ----------
        rho : Array
            Density field ρ(r), shape (nx, ny, nz)
        
        Returns
        -------
        WeightedDensities
            All weighted densities and derived quantities
        """
        # FFT of density
        rho_hat = jnp.fft.fftn(rho)
        
        # ──────────────────────────────────────────────────────────
        # Scalar weighted densities
        # ──────────────────────────────────────────────────────────
        eta = jnp.real(jnp.fft.ifftn(rho_hat * self.kernels.w3_hat))
        n0 = jnp.real(jnp.fft.ifftn(rho_hat * self.kernels.w0_hat))
        n1 = jnp.real(jnp.fft.ifftn(rho_hat * self.kernels.w1_hat))
        n2 = jnp.real(jnp.fft.ifftn(rho_hat * self.kernels.w2_hat))
        
        # ──────────────────────────────────────────────────────────
        # Vector weighted densities
        # ──────────────────────────────────────────────────────────
        nv1 = jnp.stack([
            jnp.real(jnp.fft.ifftn(rho_hat * self.kernels.wv1_hat[i]))
            for i in range(3)
        ], axis=0)
        
        nv2 = jnp.stack([
            jnp.real(jnp.fft.ifftn(rho_hat * self.kernels.wv2_hat[i]))
            for i in range(3)
        ], axis=0)
        
        # ──────────────────────────────────────────────────────────
        # Tensor weighted density
        # ──────────────────────────────────────────────────────────
        T = jnp.stack([
            jnp.stack([
                jnp.real(jnp.fft.ifftn(rho_hat * self.kernels.T_hat[i, j]))
                for j in range(3)
            ], axis=0)
            for i in range(3)
        ], axis=0)
        
        # ──────────────────────────────────────────────────────────
        # Derived quantities
        # ──────────────────────────────────────────────────────────
        
        # Vector products
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=0)
        nv2_sq = jnp.sum(nv2 * nv2, axis=0)
        
        # Tensor traces: Tr(T²) and Tr(T³)
        TT = jnp.einsum('ik...,kj...->ij...', T, T)
        T2 = jnp.einsum('ii...->...', TT)
        T3 = jnp.einsum('ij...,jk...,ki...->...', T, T, T)
        
        # Tensor contraction: nᵥ₂ · T · nᵥ₂
        nvTnv = jnp.einsum('i...,ij...,j...->...', nv2, T, nv2)
        
        return WeightedDensities(
            eta=eta, n0=n0, n1=n1, n2=n2,
            nv1=nv1, nv2=nv2, T=T,
            nv1_dot_nv2=nv1_dot_nv2, nv2_sq=nv2_sq,
            T2=T2, T3=T3, nvTnv=nvTnv
        )
