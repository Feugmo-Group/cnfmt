"""
Full 3D FMT Implementation with Tensor Terms
=============================================

This module implements the complete 3D Fundamental Measure Theory (FMT)
with proper tensor weighted densities for accurate hard wall profiles.

Key Features:
1. Full tensor weighted densities (scalar, vector, tensor)
2. White Bear II formulation with φ₂(η), φ₃(η) corrections
3. esFMT (Lutsko) framework with (A, B) parameters
4. 3D FFT-based convolutions for weighted densities
5. Proper c⁽¹⁾ calculation via chain rule through weighted densities
6. Picard iteration solver for equilibrium profiles

Theory:
-------
The FMT excess free energy density Φ has three contributions:

  Φ = Φ₁ + Φ₂ + Φ₃

where:
  Φ₁ = -n₀ ln(1-η)
  Φ₂ = (n₁n₂ - nᵥ₁·nᵥ₂)/(1-η)        [or with φ₂(η) for WBII]
  Φ₃ = function of (η, n₂, nᵥ₂, T)    [various formulations]

Weighted Densities:
  η(r)  = n₃(r) = ∫ρ(r')w₃(r-r')dr'     (scalar, packing fraction)
  n₂(r) = ∫ρ(r')w₂(r-r')dr'             (scalar, surface)
  n₁(r) = ∫ρ(r')w₁(r-r')dr'             (scalar)
  n₀(r) = ∫ρ(r')w₀(r-r')dr'             (scalar)
  nᵥ₂(r) = ∫ρ(r')wᵥ₂(r-r')dr'          (vector)
  nᵥ₁(r) = ∫ρ(r')wᵥ₁(r-r')dr'          (vector)
  T(r) = ∫ρ(r')wₜ(r-r')dr'              (tensor)

Weight Functions for spheres of diameter σ (R = σ/2):
  w₃(r) = Θ(R - |r|)                     (volume)
  w₂(r) = δ(|r| - R)                     (surface)
  w₁(r) = w₂/(4πR)                       
  w₀(r) = w₂/(4πR²)
  wᵥ₂(r) = (r/|r|)δ(|r| - R)            (vector)
  wᵥ₁(r) = wᵥ₂/(4πR)
  wₜ(r) = (rr/|r|² - I/3)δ(|r| - R)     (tensor, traceless)

Author: Computational Materials Science
Date: 2024
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from typing import Dict, Tuple, Optional, NamedTuple
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONSTANTS AND PARAMETERS
# ============================================================================

PI = jnp.pi
SIGMA = 1.0  # Hard sphere diameter
R = SIGMA / 2.0  # Hard sphere radius


# ============================================================================
# WHITE BEAR II CORRECTION FUNCTIONS
# ============================================================================

def phi2_WBII(eta: jnp.ndarray) -> jnp.ndarray:
    """
    White Bear II φ₂(η) correction function.
    
    φ₂(η) = 1 - [2η - 3η² + 2η³ + 2(1-η)²ln(1-η)] / (3η²)
    
    This modifies the Φ₂ contribution for improved accuracy.
    
    Limit: φ₂(0) → 1
    """
    eta_safe = jnp.clip(eta, 1e-14, 1.0 - 1e-10)
    log_term = jnp.log(1.0 - eta_safe)
    
    numer = 2*eta_safe - 3*eta_safe**2 + 2*eta_safe**3 + 2*(1-eta_safe)**2 * log_term
    
    # Taylor expansion for small η: φ₂ ≈ 1 - η/3 + O(η²)
    phi2_small = 1.0 - eta_safe/3.0 - eta_safe**2/6.0
    phi2_full = 1.0 - numer / (3*eta_safe**2)
    
    return jnp.where(eta_safe < 1e-6, phi2_small, phi2_full)


def phi3_WBII(eta: jnp.ndarray) -> jnp.ndarray:
    """
    White Bear II φ₃(η) correction function.
    
    φ₃(η) = 1 - [2η - η² + 2(1-η)ln(1-η)] / (3η²)
    
    This gives Carnahan-Starling EOS for uniform fluids.
    
    Limit: φ₃(0) → 1
    """
    eta_safe = jnp.clip(eta, 1e-14, 1.0 - 1e-10)
    log_term = jnp.log(1.0 - eta_safe)
    
    numer = 2*eta_safe - eta_safe**2 + 2*(1-eta_safe) * log_term
    
    # Taylor expansion for small η
    phi3_small = 1.0 - eta_safe/3.0 - eta_safe**2/9.0
    phi3_full = 1.0 - numer / (3*eta_safe**2)
    
    return jnp.where(eta_safe < 1e-6, phi3_small, phi3_full)


def dphi2_WBII(eta: jnp.ndarray) -> jnp.ndarray:
    """dφ₂/dη via autodiff."""
    return grad(lambda x: phi2_WBII(x).sum())(eta)


def dphi3_WBII(eta: jnp.ndarray) -> jnp.ndarray:
    """dφ₃/dη via autodiff."""
    return grad(lambda x: phi3_WBII(x).sum())(eta)


# ============================================================================
# WEIGHTED DENSITY CLASSES
# ============================================================================

class WeightedDensities(NamedTuple):
    """Container for all weighted densities."""
    n0: jnp.ndarray  # Scalar
    n1: jnp.ndarray  # Scalar
    n2: jnp.ndarray  # Scalar  
    n3: jnp.ndarray  # Scalar (η, packing fraction)
    nv1: jnp.ndarray  # Vector (3 components)
    nv2: jnp.ndarray  # Vector (3 components)
    T: jnp.ndarray   # Tensor (3x3, traceless symmetric)


class WeightedDensityDerivatives(NamedTuple):
    """Container for ∂Φ/∂nα derivatives."""
    dn0: jnp.ndarray
    dn1: jnp.ndarray
    dn2: jnp.ndarray
    dn3: jnp.ndarray
    dnv1: jnp.ndarray
    dnv2: jnp.ndarray
    dT: jnp.ndarray


# ============================================================================
# 3D GRID AND FFT WEIGHT FUNCTIONS
# ============================================================================

class Grid3D:
    """3D grid for FMT calculations."""
    
    def __init__(self, nx: int, ny: int, nz: int, 
                 Lx: float, Ly: float, Lz: float):
        """
        Initialize 3D grid.
        
        Parameters:
        -----------
        nx, ny, nz : int
            Number of grid points in each direction
        Lx, Ly, Lz : float
            Box dimensions
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dz = Lz / nz
        self.dV = self.dx * self.dy * self.dz
        
        # Real-space coordinates
        self.x = jnp.linspace(self.dx/2, Lx - self.dx/2, nx)
        self.y = jnp.linspace(self.dy/2, Ly - self.dy/2, ny)
        self.z = jnp.linspace(self.dz/2, Lz - self.dz/2, nz)
        
        # 3D meshgrid
        self.X, self.Y, self.Z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # k-space frequencies
        self.kx = 2*PI * jnp.fft.fftfreq(nx, self.dx)
        self.ky = 2*PI * jnp.fft.fftfreq(ny, self.dy)
        self.kz = 2*PI * jnp.fft.fftfreq(nz, self.dz)
        
        # 3D k-space meshgrid
        self.KX, self.KY, self.KZ = jnp.meshgrid(self.kx, self.ky, self.kz, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.K = jnp.sqrt(self.K2)


class FMTWeights3D:
    """
    3D weight functions for FMT in Fourier space.
    
    Uses analytical Fourier transforms of weight functions.
    """
    
    def __init__(self, grid: Grid3D, R: float = 0.5):
        """
        Initialize weight functions.
        
        Parameters:
        -----------
        grid : Grid3D
            The computational grid
        R : float
            Hard sphere radius (σ/2)
        """
        self.grid = grid
        self.R = R
        self._compute_weights()
    
    def _compute_weights(self):
        """Compute weight functions in Fourier space."""
        R = self.R
        k = self.grid.K
        kx, ky, kz = self.grid.KX, self.grid.KY, self.grid.KZ
        
        # Avoid division by zero
        k_safe = jnp.where(k < 1e-10, 1e-10, k)
        kR = k_safe * R
        
        # Scalar weight functions (Fourier transforms)
        # ŵ₃(k) = (4π/k³)[sin(kR) - kR·cos(kR)]
        self.w3_hat = jnp.where(
            k < 1e-10,
            (4/3) * PI * R**3,  # k→0 limit
            (4*PI / k_safe**3) * (jnp.sin(kR) - kR * jnp.cos(kR))
        )
        
        # ŵ₂(k) = 4πR²·sin(kR)/(kR)
        self.w2_hat = jnp.where(
            k < 1e-10,
            4*PI * R**2,  # k→0 limit
            4*PI * R**2 * jnp.sin(kR) / kR
        )
        
        # ŵ₁ = ŵ₂/(4πR)
        self.w1_hat = self.w2_hat / (4*PI * R)
        
        # ŵ₀ = ŵ₂/(4πR²)
        self.w0_hat = self.w2_hat / (4*PI * R**2)
        
        # Vector weight functions: ŵᵥ₂(k) = -i·k̂·ŵ₂(k) 
        # In components: ŵᵥ₂,ᵢ(k) = -i·kᵢ/|k|·ŵ₂(k)
        # This gives wᵥ₂ = (r/|r|)δ(|r|-R) in real space
        
        # j₁(kR) = sin(kR)/(kR)² - cos(kR)/(kR)
        j1_kR = jnp.where(
            k < 1e-10,
            1/3,  # j₁(0) = 1/3
            jnp.sin(kR)/kR**2 - jnp.cos(kR)/kR
        )
        
        # ŵᵥ₂(k) = -4πiR·j₁(kR)·k̂
        prefactor_v2 = jnp.where(
            k < 1e-10,
            0.0,
            -4*PI*1j*R * j1_kR
        )
        
        self.wv2_hat_x = prefactor_v2 * kx / k_safe
        self.wv2_hat_y = prefactor_v2 * ky / k_safe
        self.wv2_hat_z = prefactor_v2 * kz / k_safe
        
        # ŵᵥ₁ = ŵᵥ₂/(4πR)
        self.wv1_hat_x = self.wv2_hat_x / (4*PI * R)
        self.wv1_hat_y = self.wv2_hat_y / (4*PI * R)
        self.wv1_hat_z = self.wv2_hat_z / (4*PI * R)
        
        # Tensor weight function (traceless part)
        # wₜ,ᵢⱼ(r) = (rᵢrⱼ/r² - δᵢⱼ/3)δ(|r|-R)
        # In Fourier space, this involves spherical harmonics
        # ŵₜ,ᵢⱼ(k) ∝ (kᵢkⱼ/k² - δᵢⱼ/3)·f(kR)
        
        # The tensor FT is more complex - using numerical approach
        # For the traceless symmetric tensor:
        # f(kR) = 4πR²[3j₂(kR)/kR] where j₂ is spherical Bessel
        
        # j₂(x) = (3/x² - 1)sin(x)/x - 3cos(x)/x²
        j2_kR = jnp.where(
            k < 1e-10,
            2/15,  # j₂(0) limit
            (3/kR**2 - 1)*jnp.sin(kR)/kR - 3*jnp.cos(kR)/kR**2
        )
        
        # Prefactor for tensor weights
        tensor_prefactor = jnp.where(
            k < 1e-10,
            0.0,
            4*PI * R**2 * 3 * j2_kR
        )
        
        # Components: wT_ij_hat = prefactor × (ki*kj/k² - δij/3)
        kk_norm = k_safe**2
        
        # Diagonal components
        self.wT_hat_xx = tensor_prefactor * (kx*kx/kk_norm - 1/3)
        self.wT_hat_yy = tensor_prefactor * (ky*ky/kk_norm - 1/3)
        self.wT_hat_zz = tensor_prefactor * (kz*kz/kk_norm - 1/3)
        
        # Off-diagonal components (symmetric)
        self.wT_hat_xy = tensor_prefactor * (kx*ky/kk_norm)
        self.wT_hat_xz = tensor_prefactor * (kx*kz/kk_norm)
        self.wT_hat_yz = tensor_prefactor * (ky*kz/kk_norm)
    
    def compute_weighted_densities(self, rho: jnp.ndarray) -> WeightedDensities:
        """
        Compute all weighted densities from density field.
        
        Parameters:
        -----------
        rho : jnp.ndarray
            3D density field, shape (nx, ny, nz)
        
        Returns:
        --------
        WeightedDensities : NamedTuple with all weighted densities
        """
        # FFT of density
        rho_hat = jnp.fft.fftn(rho) * self.grid.dV
        
        # Scalar densities via convolution
        n3 = jnp.real(jnp.fft.ifftn(rho_hat * self.w3_hat))
        n2 = jnp.real(jnp.fft.ifftn(rho_hat * self.w2_hat))
        n1 = jnp.real(jnp.fft.ifftn(rho_hat * self.w1_hat))
        n0 = jnp.real(jnp.fft.ifftn(rho_hat * self.w0_hat))
        
        # Clip n3 to valid range
        n3 = jnp.clip(n3, 0.0, 0.9999)
        
        # Vector densities
        nv2_x = jnp.real(jnp.fft.ifftn(rho_hat * self.wv2_hat_x))
        nv2_y = jnp.real(jnp.fft.ifftn(rho_hat * self.wv2_hat_y))
        nv2_z = jnp.real(jnp.fft.ifftn(rho_hat * self.wv2_hat_z))
        nv2 = jnp.stack([nv2_x, nv2_y, nv2_z], axis=-1)
        
        nv1_x = jnp.real(jnp.fft.ifftn(rho_hat * self.wv1_hat_x))
        nv1_y = jnp.real(jnp.fft.ifftn(rho_hat * self.wv1_hat_y))
        nv1_z = jnp.real(jnp.fft.ifftn(rho_hat * self.wv1_hat_z))
        nv1 = jnp.stack([nv1_x, nv1_y, nv1_z], axis=-1)
        
        # Tensor densities (3x3 symmetric traceless)
        T_xx = jnp.real(jnp.fft.ifftn(rho_hat * self.wT_hat_xx))
        T_yy = jnp.real(jnp.fft.ifftn(rho_hat * self.wT_hat_yy))
        T_zz = jnp.real(jnp.fft.ifftn(rho_hat * self.wT_hat_zz))
        T_xy = jnp.real(jnp.fft.ifftn(rho_hat * self.wT_hat_xy))
        T_xz = jnp.real(jnp.fft.ifftn(rho_hat * self.wT_hat_xz))
        T_yz = jnp.real(jnp.fft.ifftn(rho_hat * self.wT_hat_yz))
        
        # Assemble tensor (shape: nx, ny, nz, 3, 3)
        T = jnp.zeros((*rho.shape, 3, 3))
        T = T.at[..., 0, 0].set(T_xx)
        T = T.at[..., 1, 1].set(T_yy)
        T = T.at[..., 2, 2].set(T_zz)
        T = T.at[..., 0, 1].set(T_xy)
        T = T.at[..., 1, 0].set(T_xy)
        T = T.at[..., 0, 2].set(T_xz)
        T = T.at[..., 2, 0].set(T_xz)
        T = T.at[..., 1, 2].set(T_yz)
        T = T.at[..., 2, 1].set(T_yz)
        
        return WeightedDensities(n0=n0, n1=n1, n2=n2, n3=n3, 
                                  nv1=nv1, nv2=nv2, T=T)


# ============================================================================
# FMT FREE ENERGY FUNCTIONALS
# ============================================================================

class FMTFunctional:
    """
    Base class for FMT functionals.
    
    Computes free energy density Φ and its derivatives ∂Φ/∂nα.
    """
    
    def __init__(self, name: str = "Base FMT"):
        self.name = name
    
    def Phi(self, wd: WeightedDensities) -> jnp.ndarray:
        """Compute free energy density Φ(r)."""
        raise NotImplementedError
    
    def dPhi(self, wd: WeightedDensities) -> WeightedDensityDerivatives:
        """Compute ∂Φ/∂nα for all weighted densities."""
        raise NotImplementedError


class RosenfeldFMT(FMTFunctional):
    """
    Rosenfeld FMT (Original 1989 formulation).
    
    Φ₁ = -n₀ ln(1-η)
    Φ₂ = (n₁n₂ - nᵥ₁·nᵥ₂)/(1-η)
    Φ₃ = (n₂³ - 3n₂·nᵥ₂²)/(24π(1-η)²)
    
    No tensor terms.
    """
    
    def __init__(self):
        super().__init__("Rosenfeld FMT")
    
    def Phi(self, wd: WeightedDensities) -> jnp.ndarray:
        """Compute Rosenfeld free energy density."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2 = wd.nv1, wd.nv2
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        # Vector dot products
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        
        # Free energy contributions
        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = (n1*n2 - nv1_dot_nv2) / one_m_eta
        Phi3 = (n2**3 - 3*n2*nv2_sq) / (24*PI * one_m_eta**2)
        
        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    def dPhi(self, wd: WeightedDensities) -> WeightedDensityDerivatives:
        """Compute ∂Φ/∂nα for Rosenfeld."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2, T = wd.nv1, wd.nv2, wd.T
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        
        # ∂Φ/∂n₀ = -ln(1-η)
        dn0 = -jnp.log(one_m_eta)
        
        # ∂Φ/∂n₁ = n₂/(1-η)
        dn1 = n2 / one_m_eta
        
        # ∂Φ/∂n₂ = n₁/(1-η) + (3n₂² - 3nᵥ₂²)/(24π(1-η)²)
        dn2 = n1/one_m_eta + (3*n2**2 - 3*nv2_sq)/(24*PI*one_m_eta**2)
        
        # ∂Φ/∂n₃ (η) = n₀/(1-η) + (n₁n₂ - nᵥ₁·nᵥ₂)/(1-η)² 
        #             + 2(n₂³ - 3n₂nᵥ₂²)/(24π(1-η)³)
        dn3 = (n0/one_m_eta + 
               (n1*n2 - nv1_dot_nv2)/one_m_eta**2 + 
               2*(n2**3 - 3*n2*nv2_sq)/(24*PI*one_m_eta**3))
        
        # ∂Φ/∂nᵥ₁ = -nᵥ₂/(1-η)
        dnv1 = -nv2 / one_m_eta[..., None]
        
        # ∂Φ/∂nᵥ₂ = -nᵥ₁/(1-η) - 6n₂nᵥ₂/(24π(1-η)²)
        dnv2 = -nv1/one_m_eta[..., None] - n2[..., None]*nv2/(4*PI*one_m_eta**2)[..., None]
        
        # ∂Φ/∂T = 0 for Rosenfeld
        dT = jnp.zeros_like(T)
        
        return WeightedDensityDerivatives(dn0, dn1, dn2, dn3, dnv1, dnv2, dT)


class WhiteBearIIFMT(FMTFunctional):
    """
    White Bear Mark II FMT.
    
    Uses φ₂(η) and φ₃(η) correction functions for improved accuracy.
    
    Φ₁ = -n₀ ln(1-η)
    Φ₂ = φ₂(η)(n₁n₂ - nᵥ₁·nᵥ₂)/(1-η)
    Φ₃ = φ₃(η)(n₂³ - 3n₂·nᵥ₂²)/(24π(1-η)²)
    
    where:
    φ₂(η) = 1 - [2η - 3η² + 2η³ + 2(1-η)²ln(1-η)]/(3η²)
    φ₃(η) = 1 - [2η - η² + 2(1-η)ln(1-η)]/(3η²)
    
    This gives the Carnahan-Starling EOS for uniform fluids.
    """
    
    def __init__(self):
        super().__init__("White Bear II FMT")
    
    def Phi(self, wd: WeightedDensities) -> jnp.ndarray:
        """Compute WBII free energy density."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2 = wd.nv1, wd.nv2
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        # Correction functions
        phi2 = phi2_WBII(eta)
        phi3 = phi3_WBII(eta)
        
        # Vector dot products
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        
        # Free energy contributions
        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = phi2 * (n1*n2 - nv1_dot_nv2) / one_m_eta
        Phi3 = phi3 * (n2**3 - 3*n2*nv2_sq) / (24*PI * one_m_eta**2)
        
        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    def dPhi(self, wd: WeightedDensities) -> WeightedDensityDerivatives:
        """Compute ∂Φ/∂nα for WBII (using autodiff for φ derivatives)."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2, T = wd.nv1, wd.nv2, wd.T
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        phi2 = phi2_WBII(eta)
        phi3 = phi3_WBII(eta)
        
        # Numerical derivatives of phi functions
        eps = 1e-6
        dphi2 = (phi2_WBII(eta + eps) - phi2_WBII(eta - eps)) / (2*eps)
        dphi3 = (phi3_WBII(eta + eps) - phi3_WBII(eta - eps)) / (2*eps)
        
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        
        # Base terms
        term2_base = (n1*n2 - nv1_dot_nv2)
        term3_base = (n2**3 - 3*n2*nv2_sq) / (24*PI)
        
        # ∂Φ/∂n₀ = -ln(1-η)
        dn0 = -jnp.log(one_m_eta)
        
        # ∂Φ/∂n₁ = φ₂·n₂/(1-η)
        dn1 = phi2 * n2 / one_m_eta
        
        # ∂Φ/∂n₂ = φ₂·n₁/(1-η) + φ₃·(3n₂² - 3nᵥ₂²)/(24π(1-η)²)
        dn2 = phi2*n1/one_m_eta + phi3*(3*n2**2 - 3*nv2_sq)/(24*PI*one_m_eta**2)
        
        # ∂Φ/∂n₃ (complex due to φ₂', φ₃')
        dn3 = (n0/one_m_eta + 
               dphi2*term2_base/one_m_eta + phi2*term2_base/one_m_eta**2 +
               dphi3*term3_base/one_m_eta**2 + 2*phi3*term3_base/one_m_eta**3)
        
        # ∂Φ/∂nᵥ₁ = -φ₂·nᵥ₂/(1-η)
        dnv1 = -phi2[..., None] * nv2 / one_m_eta[..., None]
        
        # ∂Φ/∂nᵥ₂ = -φ₂·nᵥ₁/(1-η) - φ₃·6n₂nᵥ₂/(24π(1-η)²)
        dnv2 = (-phi2[..., None]*nv1/one_m_eta[..., None] - 
                phi3[..., None]*n2[..., None]*nv2/(4*PI*one_m_eta**2)[..., None])
        
        # ∂Φ/∂T = 0 for WBII (Rosenfeld-based, no tensor)
        dT = jnp.zeros_like(T)
        
        return WeightedDensityDerivatives(dn0, dn1, dn2, dn3, dnv1, dnv2, dT)


class esFMT_Tensor(FMTFunctional):
    """
    Extended Scalar FMT (Lutsko) with full tensor terms.
    
    Φ₃ = [(A/24π)(n₂³ - 3n₂nᵥ₂² + 3nᵥ₂·T·nᵥ₂ - Tr(T³)) + 
          (B/24π)(n₂³ - 3n₂·Tr(T²) + 2·Tr(T³))] / (1-η)²
    
    Special cases:
    - A=3/2, B=0     → Rosenfeld (scalar only)
    - A=3/2, B=-3/2  → Tarazona (full tensor)
    - A=1,   B=-1    → White Bear basis
    
    Parameters (A, B) control:
    - Equation of state (bulk thermodynamics)
    - Surface tension and contact theorem
    """
    
    def __init__(self, A: float = 1.0, B: float = -1.0):
        """
        Initialize esFMT with parameters.
        
        Parameters:
        -----------
        A : float
            Coefficient for first tensor term (default 1.0 = WB)
        B : float
            Coefficient for second tensor term (default -1.0 = WB)
        """
        super().__init__(f"esFMT (A={A:.2f}, B={B:.2f})")
        self.A = A
        self.B = B
    
    def Phi(self, wd: WeightedDensities) -> jnp.ndarray:
        """Compute esFMT free energy density with tensor terms."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2, T = wd.nv1, wd.nv2, wd.T
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        # Vector quantities
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        
        # Tensor quantities
        # T² = T·T
        T_sq = jnp.einsum('...ik,...kj->...ij', T, T)
        # Tr(T²)
        T2_trace = jnp.trace(T_sq, axis1=-2, axis2=-1)
        # T³ = T·T·T
        T_cube = jnp.einsum('...ik,...kj->...ij', T_sq, T)
        # Tr(T³)
        T3_trace = jnp.trace(T_cube, axis1=-2, axis2=-1)
        # nᵥ₂·T·nᵥ₂
        vTv = jnp.einsum('...i,...ij,...j->...', nv2, T, nv2)
        
        # Free energy contributions
        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = (n1*n2 - nv1_dot_nv2) / one_m_eta
        
        # Φ₃ with tensor terms
        # A term: (n₂³ - 3n₂nᵥ₂² + 3vTv - Tr(T³))
        term_A = n2**3 - 3*n2*nv2_sq + 3*vTv - T3_trace
        
        # B term: (n₂³ - 3n₂·Tr(T²) + 2·Tr(T³))
        term_B = n2**3 - 3*n2*T2_trace + 2*T3_trace
        
        Phi3 = (self.A * term_A + self.B * term_B) / (24*PI * one_m_eta**2)
        
        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    def dPhi(self, wd: WeightedDensities) -> WeightedDensityDerivatives:
        """Compute ∂Φ/∂nα for esFMT with tensor terms."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2, T = wd.nv1, wd.nv2, wd.T
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        A, B = self.A, self.B
        
        # Pre-compute tensor quantities
        T_sq = jnp.einsum('...ik,...kj->...ij', T, T)
        T2_trace = jnp.trace(T_sq, axis1=-2, axis2=-1)
        T_cube = jnp.einsum('...ik,...kj->...ij', T_sq, T)
        T3_trace = jnp.trace(T_cube, axis1=-2, axis2=-1)
        
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        vTv = jnp.einsum('...i,...ij,...j->...', nv2, T, nv2)
        
        # A and B terms for Φ₃
        term_A = n2**3 - 3*n2*nv2_sq + 3*vTv - T3_trace
        term_B = n2**3 - 3*n2*T2_trace + 2*T3_trace
        
        # ∂Φ/∂n₀ = -ln(1-η)
        dn0 = -jnp.log(one_m_eta)
        
        # ∂Φ/∂n₁ = n₂/(1-η)
        dn1 = n2 / one_m_eta
        
        # ∂Φ/∂n₂ = n₁/(1-η) + [A(3n₂² - 3nᵥ₂²) + B(3n₂² - 3Tr(T²))]/(24π(1-η)²)
        dn2 = (n1/one_m_eta + 
               (A*(3*n2**2 - 3*nv2_sq) + B*(3*n2**2 - 3*T2_trace))/(24*PI*one_m_eta**2))
        
        # ∂Φ/∂n₃ (η)
        dn3 = (n0/one_m_eta + 
               (n1*n2 - nv1_dot_nv2)/one_m_eta**2 +
               2*(A*term_A + B*term_B)/(24*PI*one_m_eta**3))
        
        # ∂Φ/∂nᵥ₁ = -nᵥ₂/(1-η)
        dnv1 = -nv2 / one_m_eta[..., None]
        
        # ∂Φ/∂nᵥ₂ = -nᵥ₁/(1-η) + A·[-6n₂nᵥ₂ + 6T·nᵥ₂]/(24π(1-η)²)
        # Compute T·nᵥ₂
        T_nv2 = jnp.einsum('...ij,...j->...i', T, nv2)
        
        dnv2 = (-nv1/one_m_eta[..., None] + 
                A*(-6*n2[..., None]*nv2 + 6*T_nv2)/(24*PI*one_m_eta**2)[..., None])
        
        # ∂Φ/∂T = [A·(3nᵥ₂⊗nᵥ₂ - 3T²) + B·(-6n₂T + 6T²)]/(24π(1-η)²)
        # nᵥ₂⊗nᵥ₂ outer product
        nv2_outer = jnp.einsum('...i,...j->...ij', nv2, nv2)
        
        dT = ((A*(3*nv2_outer - 3*T_sq) + B*(-6*n2[..., None, None]*T + 6*T_sq)) /
              (24*PI*one_m_eta**2)[..., None, None])
        
        return WeightedDensityDerivatives(dn0, dn1, dn2, dn3, dnv1, dnv2, dT)


class WBII_Tensor(FMTFunctional):
    """
    White Bear II with full tensor terms.
    
    Combines:
    - WBII φ₂(η), φ₃(η) correction functions 
    - Tensor terms from esFMT (Tarazona-like)
    
    This should give the best accuracy for hard wall profiles.
    
    Φ₁ = -n₀ ln(1-η)
    Φ₂ = φ₂(η)(n₁n₂ - nᵥ₁·nᵥ₂)/(1-η)
    Φ₃ = φ₃(η) × tensor_term / (24π(1-η)²)
    
    where tensor_term uses Tarazona formulation with (1-ξ²)³ factor.
    """
    
    def __init__(self):
        super().__init__("White Bear II + Tensor")
    
    def Phi(self, wd: WeightedDensities) -> jnp.ndarray:
        """Compute WBII+Tensor free energy density."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2, T = wd.nv1, wd.nv2, wd.T
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        phi2 = phi2_WBII(eta)
        phi3 = phi3_WBII(eta)
        
        # Vector quantities
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        
        # Tensor quantities
        T_sq = jnp.einsum('...ik,...kj->...ij', T, T)
        T2_trace = jnp.trace(T_sq, axis1=-2, axis2=-1)
        T_cube = jnp.einsum('...ik,...kj->...ij', T_sq, T)
        T3_trace = jnp.trace(T_cube, axis1=-2, axis2=-1)
        
        # ξ² = |nᵥ₂|²/n₂² (anisotropy parameter)
        n2_safe = jnp.maximum(jnp.abs(n2), 1e-20)
        xi2 = nv2_sq / n2_safe**2
        xi2 = jnp.clip(xi2, 0, 0.9999)
        
        # Free energy contributions
        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = phi2 * (n1*n2 - nv1_dot_nv2) / one_m_eta
        
        # Φ₃ with Tarazona-style tensor term and (1-ξ²) factor
        # Using combination of vector and tensor terms
        Phi3_num = n2**3 * (1 - xi2)**2 - 3*n2*T2_trace + 2*T3_trace
        Phi3 = phi3 * Phi3_num / (24*PI * one_m_eta**2)
        
        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    def dPhi(self, wd: WeightedDensities) -> WeightedDensityDerivatives:
        """Compute derivatives for WBII+Tensor (using numerical approach)."""
        n0, n1, n2, n3 = wd.n0, wd.n1, wd.n2, wd.n3
        nv1, nv2, T = wd.nv1, wd.nv2, wd.T
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        phi2 = phi2_WBII(eta)
        phi3 = phi3_WBII(eta)
        
        # Numerical derivatives
        eps = 1e-6
        dphi2 = (phi2_WBII(eta + eps) - phi2_WBII(eta - eps)) / (2*eps)
        dphi3 = (phi3_WBII(eta + eps) - phi3_WBII(eta - eps)) / (2*eps)
        
        # Pre-compute quantities
        nv1_dot_nv2 = jnp.sum(nv1 * nv2, axis=-1)
        nv2_sq = jnp.sum(nv2 * nv2, axis=-1)
        
        T_sq = jnp.einsum('...ik,...kj->...ij', T, T)
        T2_trace = jnp.trace(T_sq, axis1=-2, axis2=-1)
        T_cube = jnp.einsum('...ik,...kj->...ij', T_sq, T)
        T3_trace = jnp.trace(T_cube, axis1=-2, axis2=-1)
        
        n2_safe = jnp.maximum(jnp.abs(n2), 1e-20)
        xi2 = jnp.clip(nv2_sq / n2_safe**2, 0, 0.9999)
        
        # Base terms
        term2_base = n1*n2 - nv1_dot_nv2
        Phi3_num = n2**3 * (1 - xi2)**2 - 3*n2*T2_trace + 2*T3_trace
        
        # ∂Φ/∂n₀
        dn0 = -jnp.log(one_m_eta)
        
        # ∂Φ/∂n₁
        dn1 = phi2 * n2 / one_m_eta
        
        # ∂Φ/∂n₂ (complex due to ξ² = nᵥ₂²/n₂²)
        # d(n₂³(1-ξ²)²)/dn₂ = 3n₂²(1-ξ²)² + n₂³·2(1-ξ²)·2ξ²/n₂
        #                   = 3n₂²(1-ξ²)² + 4n₂²(1-ξ²)ξ² = 3n₂²(1-ξ²)(1+ξ²/3)
        dPhi3_num_dn2 = 3*n2**2*(1-xi2)*(1+xi2/3) - 3*T2_trace
        dn2 = phi2*n1/one_m_eta + phi3*dPhi3_num_dn2/(24*PI*one_m_eta**2)
        
        # ∂Φ/∂n₃
        dn3 = (n0/one_m_eta +
               dphi2*term2_base/one_m_eta + phi2*term2_base/one_m_eta**2 +
               dphi3*Phi3_num/(24*PI*one_m_eta**2) + 2*phi3*Phi3_num/(24*PI*one_m_eta**3))
        
        # ∂Φ/∂nᵥ₁
        dnv1 = -phi2[..., None] * nv2 / one_m_eta[..., None]
        
        # ∂Φ/∂nᵥ₂
        # d(n₂³(1-ξ²)²)/dnᵥ₂ = n₂³·2(1-ξ²)·(-2nᵥ₂/n₂²) = -4n₂(1-ξ²)nᵥ₂
        dPhi3_num_dnv2 = -4*n2[..., None]*(1-xi2)[..., None]*nv2
        dnv2 = (-phi2[..., None]*nv1/one_m_eta[..., None] + 
                phi3[..., None]*dPhi3_num_dnv2/(24*PI*one_m_eta**2)[..., None])
        
        # ∂Φ/∂T
        # d(-3n₂T²+2T³)/dT = -6n₂T + 6T²
        dT = phi3[..., None, None] * (-6*n2[..., None, None]*T + 6*T_sq) / (24*PI*one_m_eta**2)[..., None, None]
        
        return WeightedDensityDerivatives(dn0, dn1, dn2, dn3, dnv1, dnv2, dT)


# ============================================================================
# DFT SOLVER
# ============================================================================

class DFTSolver3D:
    """
    3D DFT solver for hard sphere systems.
    
    Solves the Euler-Lagrange equation:
        ln ρ(r) + βμ_ex(r) + βV_ext(r) = βμ
    
    where μ_ex(r) = -c⁽¹⁾(r) is the local excess chemical potential.
    """
    
    def __init__(self, grid: Grid3D, weights: FMTWeights3D, 
                 functional: FMTFunctional):
        """
        Initialize DFT solver.
        
        Parameters:
        -----------
        grid : Grid3D
            The computational grid
        weights : FMTWeights3D
            Weight functions
        functional : FMTFunctional
            The FMT functional to use
        """
        self.grid = grid
        self.weights = weights
        self.functional = functional
    
    def compute_c1(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        Compute one-body direct correlation function c⁽¹⁾(r).
        
        c⁽¹⁾(r) = -δF_ex/δρ(r) = -Σ_α (∂Φ/∂n_α ★ w_α)
        
        This uses the chain rule through weighted densities.
        """
        # Compute weighted densities
        wd = self.weights.compute_weighted_densities(rho)
        
        # Get functional derivatives
        dPhi = self.functional.dPhi(wd)
        
        # Compute c1 via convolution
        # c1 = -Σ_α ∫(∂Φ/∂n_α)(r') w_α(r-r') dr'
        
        dV = self.grid.dV
        
        # Scalar contributions
        c1 = jnp.zeros_like(rho)
        
        # n₀ contribution
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dn0)*dV * self.weights.w0_hat))
        
        # n₁ contribution
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dn1)*dV * self.weights.w1_hat))
        
        # n₂ contribution
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dn2)*dV * self.weights.w2_hat))
        
        # n₃ (η) contribution
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dn3)*dV * self.weights.w3_hat))
        
        # Vector contributions (nᵥ₁)
        # Need to flip sign of vector weights for correlation
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dnv1[..., 0])*dV * jnp.conj(self.weights.wv1_hat_x)))
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dnv1[..., 1])*dV * jnp.conj(self.weights.wv1_hat_y)))
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dnv1[..., 2])*dV * jnp.conj(self.weights.wv1_hat_z)))
        
        # Vector contributions (nᵥ₂)
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dnv2[..., 0])*dV * jnp.conj(self.weights.wv2_hat_x)))
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dnv2[..., 1])*dV * jnp.conj(self.weights.wv2_hat_y)))
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dnv2[..., 2])*dV * jnp.conj(self.weights.wv2_hat_z)))
        
        # Tensor contributions
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dT[..., 0, 0])*dV * self.weights.wT_hat_xx))
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dT[..., 1, 1])*dV * self.weights.wT_hat_yy))
        c1 -= jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dT[..., 2, 2])*dV * self.weights.wT_hat_zz))
        c1 -= 2*jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dT[..., 0, 1])*dV * self.weights.wT_hat_xy))
        c1 -= 2*jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dT[..., 0, 2])*dV * self.weights.wT_hat_xz))
        c1 -= 2*jnp.real(jnp.fft.ifftn(jnp.fft.fftn(dPhi.dT[..., 1, 2])*dV * self.weights.wT_hat_yz))
        
        return c1
    
    def solve_wall(self, eta_bulk: float, z_wall: float = 0.0,
                   max_iter: int = 5000, tol: float = 1e-8,
                   alpha_init: float = 0.01, verbose: bool = True):
        """
        Solve for density profile at a hard wall.
        
        Parameters:
        -----------
        eta_bulk : float
            Bulk packing fraction
        z_wall : float
            Position of the hard wall (default 0)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        alpha_init : float
            Initial mixing parameter
        verbose : bool
            Print progress
        
        Returns:
        --------
        dict with solution data
        """
        grid = self.grid
        R = self.weights.R
        
        # Bulk density
        rho_bulk = 6 * eta_bulk / PI
        
        # External potential (hard wall)
        V_ext = jnp.where(grid.Z < z_wall + R, jnp.inf, 0.0)
        wall_mask = grid.Z >= z_wall + R
        
        # Initial guess: bulk density outside wall
        rho = jnp.where(wall_mask, rho_bulk, 0.0)
        
        # Reference c1 in bulk
        rho_uniform = jnp.ones_like(rho) * rho_bulk
        c1_bulk_field = self.compute_c1(rho_uniform)
        # Take average in bulk region (far from wall)
        bulk_region = grid.Z > grid.Lz / 2
        c1_bulk = jnp.mean(jnp.where(bulk_region, c1_bulk_field, jnp.nan))
        
        if verbose:
            print(f"\n[DFT Solver: {self.functional.name}]")
            print(f"  η_bulk = {eta_bulk:.3f}")
            print(f"  ρ_bulk = {rho_bulk:.4f}")
            print(f"  c1_bulk = {float(c1_bulk):.4f}")
        
        errors = []
        alpha = alpha_init
        
        for iteration in range(max_iter):
            rho_old = rho.copy()
            
            # Compute c1
            c1 = self.compute_c1(rho)
            
            # Update density
            # ρ(r) = ρ_bulk × exp(c1(r) - c1_bulk) × mask
            rho_new = rho_bulk * jnp.exp(c1 - c1_bulk)
            rho_new = jnp.where(wall_mask, rho_new, 0.0)
            rho_new = jnp.clip(rho_new, 0.0, rho_bulk * 30)
            
            # Adaptive mixing
            if iteration < 100:
                alpha = 0.005
            elif iteration < 300:
                alpha = 0.01
            elif iteration < 1000:
                alpha = 0.02
            else:
                alpha = 0.05
            
            rho = alpha * rho_new + (1 - alpha) * rho_old
            
            # Check convergence
            error = jnp.max(jnp.abs(rho - rho_old)) / (rho_bulk + 1e-10)
            errors.append(float(error))
            
            if verbose and iteration % 500 == 0:
                # Get z-averaged profile
                rho_z = jnp.mean(rho, axis=(0, 1))
                contact_idx = jnp.argmin(jnp.abs(grid.z - R))
                contact = float(rho_z[contact_idx]) / rho_bulk
                print(f"  Iter {iteration}: err={error:.2e}, contact={contact:.3f}, α={alpha:.3f}")
            
            if error < tol:
                break
        
        # Final profile (z-averaged)
        rho_z = jnp.mean(rho, axis=(0, 1))
        
        # Contact density
        contact_idx = jnp.argmin(jnp.abs(grid.z - R))
        contact = float(rho_z[contact_idx]) / rho_bulk
        
        # CS prediction
        Z_CS = (1 + eta_bulk + eta_bulk**2 - eta_bulk**3) / (1 - eta_bulk)**3
        
        if verbose:
            print(f"  Converged: {error < tol}")
            print(f"  Contact density: {contact:.4f} (CS: {Z_CS:.4f})")
        
        return {
            'z': grid.z,
            'rho_z': rho_z,
            'rho_norm': rho_z / rho_bulk,
            'rho_3d': rho,
            'eta_bulk': eta_bulk,
            'contact': contact,
            'contact_CS': Z_CS,
            'converged': error < tol,
            'iterations': iteration + 1,
            'errors': np.array(errors),
            'functional': self.functional.name,
        }


# ============================================================================
# MONTE CARLO DATA
# ============================================================================

def get_mc_wall_profile(eta: float) -> np.ndarray:
    """Get MC wall profile data for given η."""
    profiles = {
        0.367: np.array([
            [0.510, 3.7543085], [0.530, 3.2698767], [0.550, 2.8546749],
            [0.570, 2.4986631], [0.590, 2.1929623], [0.610, 1.9302458],
            [0.630, 1.7044568], [0.650, 1.5098530], [0.670, 1.3422220],
            [0.690, 1.1976265], [0.710, 1.0726264], [0.730, 0.9646101],
            [0.750, 0.8711540], [0.770, 0.7901845], [0.790, 0.7200606],
            [0.810, 0.6592646], [0.830, 0.6065577], [0.850, 0.5609323],
            [0.870, 0.5215091], [0.890, 0.4874595], [0.910, 0.4582073],
            [0.930, 0.4331748], [0.950, 0.4119227], [0.970, 0.3940790],
            [0.990, 0.3793644], [1.010, 0.3675033], [1.030, 0.3583127],
            [1.050, 0.3516432], [1.070, 0.3474103], [1.090, 0.3455326],
            [1.110, 0.3460356], [1.130, 0.3489446], [1.150, 0.3543193],
            [1.170, 0.3622934], [1.190, 0.3729875], [1.210, 0.3866957],
            [1.230, 0.4036505], [1.250, 0.4241440], [1.270, 0.4485735],
            [1.290, 0.4773881], [1.310, 0.5108371], [1.330, 0.5495006],
        ]),
        0.393: np.array([
            [0.510, 4.6143129], [0.530, 3.9234880], [0.550, 3.3460584],
            [0.570, 2.8629162], [0.590, 2.4580002], [0.610, 2.1180064],
            [0.630, 1.8322161], [0.650, 1.5915280], [0.670, 1.3885278],
            [0.690, 1.2169994], [0.710, 1.0718423], [0.730, 0.9487341],
            [0.750, 0.8442476], [0.770, 0.7554538], [0.790, 0.6798796],
            [0.810, 0.6155189], [0.830, 0.5607089], [0.850, 0.5140389],
            [0.870, 0.4743454], [0.890, 0.4407258], [0.910, 0.4123178],
            [0.930, 0.3885200], [0.950, 0.3688011], [0.970, 0.3527117],
            [0.990, 0.3398599], [1.010, 0.3300786], [1.030, 0.3231189],
            [1.050, 0.3188159], [1.070, 0.3171822], [1.090, 0.3182020],
            [1.110, 0.3218368], [1.130, 0.3283230], [1.150, 0.3377242],
            [1.170, 0.3503607], [1.190, 0.3665090], [1.210, 0.3865268],
            [1.230, 0.4110076], [1.250, 0.4403478], [1.270, 0.4752818],
        ]),
        0.449: np.array([
            [0.510, 7.1434255], [0.530, 5.6966596], [0.550, 4.5630358],
            [0.570, 3.6720352], [0.590, 2.9702559], [0.610, 2.4154181],
            [0.630, 1.9761309], [0.650, 1.6268439], [0.670, 1.3483351],
            [0.690, 1.1256610], [0.710, 0.9469973], [0.730, 0.8031472],
            [0.750, 0.6869939], [0.770, 0.5928965], [0.790, 0.5164968],
            [0.810, 0.4542666], [0.830, 0.4035183], [0.850, 0.3621957],
            [0.870, 0.3284787], [0.890, 0.3011020], [0.910, 0.2790407],
            [0.930, 0.2615448], [0.950, 0.2478760], [0.970, 0.2376521],
            [0.990, 0.2305936], [1.010, 0.2263595], [1.030, 0.2249940],
            [1.050, 0.2263999], [1.070, 0.2307339], [1.090, 0.2381395],
            [1.110, 0.2489811], [1.130, 0.2637171], [1.150, 0.2830402],
            [1.170, 0.3076345], [1.190, 0.3387804], [1.210, 0.3776682],
        ]),
    }
    
    available = list(profiles.keys())
    closest = min(available, key=lambda x: abs(x - eta))
    
    data = profiles[closest]
    # Convert to normalized form: z/σ, ρ/ρ_bulk
    rho_bulk = 6 * closest / PI
    return np.column_stack([data[:, 0], data[:, 1] / rho_bulk])


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def run_3d_wall_validation():
    """Run full 3D FMT validation against MC data."""
    
    print("="*70)
    print("3D FMT WALL PROFILE VALIDATION WITH TENSOR TERMS")
    print("="*70)
    
    # Create grid (smaller for speed, can increase for accuracy)
    nx, ny, nz = 4, 4, 128  # Only need resolution in z
    Lx, Ly, Lz = 2.0, 2.0, 6.0
    
    print(f"\nGrid: {nx}×{ny}×{nz}, Box: {Lx}×{Ly}×{Lz}")
    
    grid = Grid3D(nx, ny, nz, Lx, Ly, Lz)
    weights = FMTWeights3D(grid, R=0.5)
    
    # Functionals to test
    functionals = [
        RosenfeldFMT(),
        WhiteBearIIFMT(),
        esFMT_Tensor(A=1.0, B=-1.0),  # WB parameters
        esFMT_Tensor(A=1.5, B=-1.5),  # Tarazona parameters
        WBII_Tensor(),
    ]
    
    # Test at different eta values
    eta_values = [0.367, 0.393, 0.449]
    
    all_results = {}
    
    for eta in eta_values:
        print(f"\n{'='*70}")
        print(f"η = {eta}")
        print('='*70)
        
        mc_data = get_mc_wall_profile(eta)
        results = {'MC': mc_data}
        
        for func in functionals:
            try:
                solver = DFTSolver3D(grid, weights, func)
                result = solver.solve_wall(eta, max_iter=3000, tol=1e-7, verbose=True)
                results[func.name] = result
            except Exception as e:
                print(f"  {func.name}: FAILED - {e}")
        
        all_results[eta] = results
    
    # Create plot
    fig, axes = plt.subplots(1, len(eta_values), figsize=(5*len(eta_values), 4))
    if len(eta_values) == 1:
        axes = [axes]
    
    colors = {
        'Rosenfeld FMT': 'C0',
        'White Bear II FMT': 'C1',
        'esFMT (A=1.00, B=-1.00)': 'C2',
        'esFMT (A=1.50, B=-1.50)': 'C3',
        'White Bear II + Tensor': 'C4',
    }
    
    for idx, eta in enumerate(eta_values):
        ax = axes[idx]
        results = all_results[eta]
        
        # MC data
        mc = results['MC']
        ax.plot(mc[:, 0], mc[:, 1], 'ko', ms=4, label='MC')
        
        # DFT results
        for name, res in results.items():
            if name == 'MC':
                continue
            ax.plot(res['z'], res['rho_norm'], '-', 
                   color=colors.get(name, 'gray'), lw=1.5, label=name[:15])
        
        ax.set_xlabel('z/σ')
        ax.set_ylabel('ρ(z)/ρ_bulk')
        ax.set_title(f'η = {eta}')
        ax.legend(fontsize=7)
        ax.set_xlim([0.4, 2.0])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/fmt_3d_validation.png', dpi=150)
    print(f"\nSaved: outputs/fmt_3d_validation.png")
    plt.close()
    
    # Summary table
    print("\n" + "="*70)
    print("CONTACT DENSITY SUMMARY")
    print("="*70)
    
    print(f"\n{'η':>6} {'CS':>8} {'MC':>8}", end='')
    for func in functionals[:3]:
        print(f" {func.name[:10]:>10}", end='')
    print()
    print("-"*60)
    
    for eta in eta_values:
        results = all_results[eta]
        mc = results['MC']
        Z_CS = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
        mc_contact = mc[0, 1]
        
        print(f"{eta:6.3f} {Z_CS:8.3f} {mc_contact:8.3f}", end='')
        for func in functionals[:3]:
            if func.name in results:
                print(f" {results[func.name]['contact']:10.3f}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        print()
    
    return all_results


if __name__ == "__main__":
    results = run_3d_wall_validation()
