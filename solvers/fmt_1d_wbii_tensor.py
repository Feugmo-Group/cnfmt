"""
1D FMT with White Bear II and Tensor Terms
==========================================

This is a clean 1D implementation for planar hard wall geometry.
Uses direct real-space convolution for weighted densities.

Implements:
1. Rosenfeld FMT (original)
2. White Bear II with φ₂, φ₃ corrections
3. esFMT with tensor terms (A, B parametrization)
4. Modified RSLT ((1-ξ²)³ factor)

Author: Computational Materials Science
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from typing import Dict, Tuple, NamedTuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PI = jnp.pi


# ============================================================================
# WHITE BEAR II CORRECTION FUNCTIONS
# ============================================================================

def phi2_WBII(eta):
    """
    White Bear II φ₂(η) correction.
    
    φ₂(η) = 1 - [2η - 3η² + 2η³ + 2(1-η)²ln(1-η)] / (3η²)
    """
    eta = jnp.clip(eta, 1e-14, 0.9999)
    log_term = jnp.log(1.0 - eta)
    numer = 2*eta - 3*eta**2 + 2*eta**3 + 2*(1-eta)**2 * log_term
    return jnp.where(eta > 1e-6, 1.0 - numer/(3*eta**2), 1.0 - eta/3)


def phi3_WBII(eta):
    """
    White Bear II φ₃(η) correction.
    
    φ₃(η) = 1 - [2η - η² + 2(1-η)ln(1-η)] / (3η²)
    
    This gives Carnahan-Starling EOS.
    """
    eta = jnp.clip(eta, 1e-14, 0.9999)
    log_term = jnp.log(1.0 - eta)
    numer = 2*eta - eta**2 + 2*(1-eta) * log_term
    return jnp.where(eta > 1e-6, 1.0 - numer/(3*eta**2), 1.0 - eta/3)


def dphi2_deta(eta):
    """dφ₂/dη via finite difference."""
    eps = 1e-6
    return (phi2_WBII(eta + eps) - phi2_WBII(eta - eps)) / (2*eps)


def dphi3_deta(eta):
    """dφ₃/dη via finite difference."""
    eps = 1e-6
    return (phi3_WBII(eta + eps) - phi3_WBII(eta - eps)) / (2*eps)


# ============================================================================
# 1D WEIGHT FUNCTIONS (Real-space convolution)
# ============================================================================

class Weights1D:
    """1D weight functions for planar FMT."""
    
    def __init__(self, dz: float, R: float = 0.5):
        """
        Parameters:
        -----------
        dz : float
            Grid spacing
        R : float
            Hard sphere radius (σ/2 = 0.5)
        """
        self.dz = dz
        self.R = R
        self._setup_weights()
    
    def _setup_weights(self):
        """Create weight function arrays."""
        R = self.R
        dz = self.dz
        
        # Grid for weights: [-R, R]
        n_half = max(int(R / dz) + 2, 4)
        n_w = 2 * n_half + 1
        self.z_w = np.linspace(-n_half * dz, n_half * dz, n_w)
        
        inside = np.abs(self.z_w) < R
        
        # Scalar weights
        # w₃(z) = π(R² - z²) for |z| < R (volume slice)
        self.w3 = np.where(inside, PI * (R**2 - self.z_w**2), 0.0)
        
        # w₂(z) = 2πR for |z| < R (surface projection)
        self.w2 = np.where(inside, 2 * PI * R, 0.0)
        
        # w₁ = w₂/(4πR) = 1/2
        self.w1 = np.where(inside, 0.5, 0.0)
        
        # w₀ = w₂/(4πR²) = 1/(2R)
        self.w0 = np.where(inside, 1.0 / (2*R), 0.0)
        
        # Vector weights (z-component only in 1D)
        # wᵥ₂,z(z) = 2πz for |z| < R
        self.wv2_z = np.where(inside, 2 * PI * self.z_w, 0.0)
        
        # wᵥ₁,z = wᵥ₂,z/(4πR)
        self.wv1_z = np.where(inside, self.z_w / (2*R), 0.0)
        
        # Tensor weight (zz component for planar)
        # wT,zz(z) = (2πR/3)(3z²/R² - 1) for |z| < R
        # This is the traceless tensor component in planar geometry
        self.wT_zz = np.where(inside, (2*PI*R/3) * (3*self.z_w**2/R**2 - 1), 0.0)
        
        # Convert to JAX arrays
        self.w3 = jnp.array(self.w3)
        self.w2 = jnp.array(self.w2)
        self.w1 = jnp.array(self.w1)
        self.w0 = jnp.array(self.w0)
        self.wv2_z = jnp.array(self.wv2_z)
        self.wv1_z = jnp.array(self.wv1_z)
        self.wT_zz = jnp.array(self.wT_zz)
    
    def convolve(self, rho, weight):
        """Convolve density with weight function."""
        conv = jnp.convolve(rho, weight, mode='full') * self.dz
        start = len(weight) // 2
        return conv[start:start + len(rho)]
    
    def compute_weighted_densities(self, rho):
        """Compute all weighted densities."""
        n3 = jnp.clip(self.convolve(rho, self.w3), 0, 0.9999)
        n2 = self.convolve(rho, self.w2)
        n1 = self.convolve(rho, self.w1)
        n0 = self.convolve(rho, self.w0)
        
        # Vector (z-component)
        nv2_z = self.convolve(rho, self.wv2_z)
        nv1_z = self.convolve(rho, self.wv1_z)
        
        # Tensor (zz component)
        T_zz = self.convolve(rho, self.wT_zz)
        
        return {
            'n0': n0, 'n1': n1, 'n2': n2, 'n3': n3,
            'nv1_z': nv1_z, 'nv2_z': nv2_z,
            'T_zz': T_zz
        }


# ============================================================================
# FMT FUNCTIONALS
# ============================================================================

class RosenfeldFMT:
    """
    Rosenfeld FMT (1989).
    
    Φ₁ = -n₀ ln(1-η)
    Φ₂ = (n₁n₂ - nᵥ₁·nᵥ₂)/(1-η)
    Φ₃ = (n₂³ - 3n₂nᵥ₂²)/(24π(1-η)²)
    """
    name = "Rosenfeld"
    
    @staticmethod
    def Phi(n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        # In 1D planar: nᵥ₁·nᵥ₂ = nv1_z × nv2_z
        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        
        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = (n1*n2 - nv1_dot_nv2) / one_m_eta
        Phi3 = (n2**3 - 3*n2*nv2_sq) / (24*PI * one_m_eta**2)
        
        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    @staticmethod
    def dPhi(n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        
        dn0 = -jnp.log(one_m_eta)
        dn1 = n2 / one_m_eta
        dn2 = n1/one_m_eta + (3*n2**2 - 3*nv2_sq)/(24*PI*one_m_eta**2)
        dn3 = (n0/one_m_eta + 
               (n1*n2 - nv1_dot_nv2)/one_m_eta**2 +
               2*(n2**3 - 3*n2*nv2_sq)/(24*PI*one_m_eta**3))
        
        dnv1_z = -nv2_z / one_m_eta
        dnv2_z = -nv1_z/one_m_eta - n2*nv2_z/(4*PI*one_m_eta**2)
        
        return {
            'dn0': dn0, 'dn1': dn1, 'dn2': dn2, 'dn3': dn3,
            'dnv1_z': dnv1_z, 'dnv2_z': dnv2_z, 'dT_zz': jnp.zeros_like(n3)
        }


class WhiteBearIIFMT:
    """
    White Bear II FMT.
    
    Uses φ₂(η), φ₃(η) corrections:
    
    Φ₁ = -n₀ ln(1-η)
    Φ₂ = φ₂(η)(n₁n₂ - nᵥ₁·nᵥ₂)/(1-η)
    Φ₃ = φ₃(η)(n₂³ - 3n₂nᵥ₂²)/(24π(1-η)²)
    
    Gives Carnahan-Starling EOS.
    """
    name = "White Bear II"
    
    @staticmethod
    def Phi(n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        phi2 = phi2_WBII(eta)
        phi3 = phi3_WBII(eta)
        
        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        
        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = phi2 * (n1*n2 - nv1_dot_nv2) / one_m_eta
        Phi3 = phi3 * (n2**3 - 3*n2*nv2_sq) / (24*PI * one_m_eta**2)
        
        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    @staticmethod
    def dPhi(n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        phi2 = phi2_WBII(eta)
        phi3 = phi3_WBII(eta)
        dp2 = dphi2_deta(eta)
        dp3 = dphi3_deta(eta)
        
        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        
        term2 = n1*n2 - nv1_dot_nv2
        term3 = (n2**3 - 3*n2*nv2_sq) / (24*PI)
        
        dn0 = -jnp.log(one_m_eta)
        dn1 = phi2 * n2 / one_m_eta
        dn2 = phi2*n1/one_m_eta + phi3*(3*n2**2 - 3*nv2_sq)/(24*PI*one_m_eta**2)
        
        # dn3 with phi corrections
        dn3 = (n0/one_m_eta +
               dp2*term2/one_m_eta + phi2*term2/one_m_eta**2 +
               dp3*term3/one_m_eta**2 + 2*phi3*term3/one_m_eta**3)
        
        dnv1_z = -phi2 * nv2_z / one_m_eta
        dnv2_z = -phi2*nv1_z/one_m_eta - phi3*n2*nv2_z/(4*PI*one_m_eta**2)
        
        return {
            'dn0': dn0, 'dn1': dn1, 'dn2': dn2, 'dn3': dn3,
            'dnv1_z': dnv1_z, 'dnv2_z': dnv2_z, 'dT_zz': jnp.zeros_like(n3)
        }


class ModifiedRSLT:
    """
    Modified RSLT with (1-ξ²)³ factor.
    
    Φ₃ = φ₂(η) × n₂³(1-ξ²)³ / (24π(1-η)²)
    
    where ξ² = nᵥ₂²/n₂² is anisotropy parameter.
    
    Positive definite and matches WBII in bulk.
    """
    name = "Modified RSLT"
    
    @staticmethod
    def Phi(n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']

        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta

        phi2 = phi2_WBII(eta)

        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2

        # ξ² = nᵥ₂²/n₂² — floor at 1e-10 to avoid underflow in n2_safe**2
        n2_safe = jnp.maximum(jnp.abs(n2), 1e-10)
        xi2 = nv2_sq / n2_safe**2
        xi2 = jnp.clip(xi2, 0, 0.9999)

        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = (n1*n2 - nv1_dot_nv2) / one_m_eta
        Phi3 = phi2 * n2**3 * (1-xi2)**3 / (24*PI * one_m_eta**2)

        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    @staticmethod
    def dPhi(n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        phi2 = phi2_WBII(eta)
        dp2 = dphi2_deta(eta)
        
        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        
        n2_safe = jnp.maximum(jnp.abs(n2), 1e-10)
        xi2 = jnp.clip(nv2_sq / n2_safe**2, 0, 0.9999)
        
        term2 = n1*n2 - nv1_dot_nv2
        Phi3_num = n2**3 * (1-xi2)**3
        
        dn0 = -jnp.log(one_m_eta)
        dn1 = n2 / one_m_eta
        
        # d(n₂³(1-ξ²)³)/dn₂ = 3n₂²(1-ξ²)²(1+ξ²)
        dPhi3_num_dn2 = 3 * n2**2 * (1-xi2)**2 * (1+xi2)
        dn2 = n1/one_m_eta + phi2*dPhi3_num_dn2/(24*PI*one_m_eta**2)
        
        dn3 = (n0/one_m_eta + term2/one_m_eta**2 +
               dp2*Phi3_num/(24*PI*one_m_eta**2) +
               2*phi2*Phi3_num/(24*PI*one_m_eta**3))
        
        dnv1_z = -nv2_z / one_m_eta
        
        # d(n₂³(1-ξ²)³)/dnᵥ₂ = -6n₂(1-ξ²)²nᵥ₂
        dPhi3_num_dnv2 = -6 * n2 * (1-xi2)**2 * nv2_z
        dnv2_z = -nv1_z/one_m_eta + phi2*dPhi3_num_dnv2/(24*PI*one_m_eta**2)
        
        return {
            'dn0': dn0, 'dn1': dn1, 'dn2': dn2, 'dn3': dn3,
            'dnv1_z': dnv1_z, 'dnv2_z': dnv2_z, 'dT_zz': jnp.zeros_like(n3)
        }


class esFMT_Tensor:
    """
    esFMT with tensor terms and (A, B) parameters.
    
    In 1D planar geometry:
    Φ₃ = [(A/24π)(n₂³ - 3n₂nᵥ₂² + 3nᵥ₂²T_zz - T_zz³/n₂²) + 
          (B/24π)(n₂³ - 3n₂T_zz² + 2T_zz³/n₂²)] / (1-η)²
    
    Special cases:
    - A=1.5, B=0    → Rosenfeld-like
    - A=1.5, B=-1.5 → Tarazona
    - A=1, B=-1     → White Bear basis
    """
    
    def __init__(self, A: float = 1.0, B: float = -1.0):
        self.A = A
        self.B = B
        self.name = f"esFMT(A={A:.1f},B={B:.1f})"
    
    def Phi(self, n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']
        T_zz = n.get('T_zz', jnp.zeros_like(n0))
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        
        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        
        # Tensor terms (simplified for 1D planar)
        # In planar geometry, T is diagonal with T_zz and T_xx=T_yy = -T_zz/2
        # Tr(T²) = T_zz² + 2(T_zz/2)² = 3T_zz²/2
        # Tr(T³) = T_zz³ + 2(-T_zz/2)³ = T_zz³ - T_zz³/4 = 3T_zz³/4
        # vTv = nv2_z² × T_zz
        
        T2 = 1.5 * T_zz**2
        T3 = 0.75 * T_zz**3
        vTv = nv2_sq * T_zz
        
        Phi1 = -n0 * jnp.log(one_m_eta)
        Phi2 = (n1*n2 - nv1_dot_nv2) / one_m_eta
        
        # A term: n₂³ - 3n₂nᵥ₂² + 3vTv - T³
        term_A = n2**3 - 3*n2*nv2_sq + 3*vTv - T3
        
        # B term: n₂³ - 3n₂T² + 2T³
        term_B = n2**3 - 3*n2*T2 + 2*T3
        
        Phi3 = (self.A*term_A + self.B*term_B) / (24*PI * one_m_eta**2)
        
        return jnp.where(n3 > 1e-12, Phi1 + Phi2 + Phi3, 0.0)
    
    def dPhi(self, n):
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z, nv2_z = n['nv1_z'], n['nv2_z']
        T_zz = n.get('T_zz', jnp.zeros_like(n0))
        
        eta = jnp.clip(n3, 1e-14, 0.9999)
        one_m_eta = 1.0 - eta
        A, B = self.A, self.B
        
        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        
        T2 = 1.5 * T_zz**2
        T3 = 0.75 * T_zz**3
        vTv = nv2_sq * T_zz
        
        term_A = n2**3 - 3*n2*nv2_sq + 3*vTv - T3
        term_B = n2**3 - 3*n2*T2 + 2*T3
        
        dn0 = -jnp.log(one_m_eta)
        dn1 = n2 / one_m_eta
        
        # dn2: derivative of terms w.r.t n2
        dn2 = (n1/one_m_eta + 
               (A*(3*n2**2 - 3*nv2_sq) + B*(3*n2**2 - 3*T2))/(24*PI*one_m_eta**2))
        
        # dn3
        dn3 = (n0/one_m_eta + 
               (n1*n2 - nv1_dot_nv2)/one_m_eta**2 +
               2*(A*term_A + B*term_B)/(24*PI*one_m_eta**3))
        
        dnv1_z = -nv2_z / one_m_eta
        
        # dnv2: A term contributes -6n2*nv2 + 6*T_zz*nv2
        dnv2_z = (-nv1_z/one_m_eta + 
                  A*(-6*n2*nv2_z + 6*T_zz*nv2_z)/(24*PI*one_m_eta**2))
        
        # dT_zz: A(-3/4*3T_zz² + 3nv2²) + B(-3*3T_zz + 2*3/4*3T_zz²)
        # = A*(3nv2² - 9/4 T_zz²) + B*(-9T_zz + 9/2 T_zz²)
        dT_zz = (A*(3*nv2_sq - 2.25*T_zz**2) + 
                 B*(-9*n2*T_zz + 4.5*T_zz**2))/(24*PI*one_m_eta**2)
        
        return {
            'dn0': dn0, 'dn1': dn1, 'dn2': dn2, 'dn3': dn3,
            'dnv1_z': dnv1_z, 'dnv2_z': dnv2_z, 'dT_zz': dT_zz
        }


# ============================================================================
# DFT WALL SOLVER
# ============================================================================

class WallSolver:
    """Solve density profiles at hard walls."""
    
    def __init__(self, nz: int = 2048, Lz: float = 6.0, R: float = 0.5):
        self.nz = nz
        self.Lz = Lz
        self.R = R
        self.dz = Lz / nz
        self.z = jnp.linspace(self.dz/2, Lz - self.dz/2, nz)
        self.weights = Weights1D(self.dz, R)
    
    def compute_c1(self, rho, functional):
        """
        Compute c⁽¹⁾(z) via chain rule through weighted densities.
        
        c⁽¹⁾(z) = -Σ_α (∂Φ/∂n_α ★ w_α)
        """
        n = self.weights.compute_weighted_densities(rho)
        dPhi = functional.dPhi(n)
        
        c1 = jnp.zeros_like(rho)
        
        # Scalar contributions
        c1 -= self.weights.convolve(dPhi['dn0'], self.weights.w0)
        c1 -= self.weights.convolve(dPhi['dn1'], self.weights.w1)
        c1 -= self.weights.convolve(dPhi['dn2'], self.weights.w2)
        c1 -= self.weights.convolve(dPhi['dn3'], self.weights.w3)
        
        # Vector contributions (note: flip sign of weight for correlation)
        c1 -= self.weights.convolve(dPhi['dnv1_z'], self.weights.wv1_z[::-1])
        c1 -= self.weights.convolve(dPhi['dnv2_z'], self.weights.wv2_z[::-1])
        
        # Tensor contribution
        c1 -= self.weights.convolve(dPhi['dT_zz'], self.weights.wT_zz[::-1])
        
        return c1
    
    def solve(self, eta_bulk: float, functional, max_iter: int = 8000,
              tol: float = 1e-9, verbose: bool = True):
        """Solve for wall profile via Picard iteration."""
        z = self.z
        R = self.R
        
        rho_bulk = 6 * eta_bulk / PI
        Z_CS = (1 + eta_bulk + eta_bulk**2 - eta_bulk**3) / (1 - eta_bulk)**3
        
        # Initial guess
        wall_mask = z >= R
        rho = jnp.where(wall_mask, rho_bulk, 0.0)
        
        # Bulk c1 reference from uniform system
        # This is the correct normalization: c1_bulk = -μ_ex
        rho_uniform = jnp.ones_like(rho) * rho_bulk
        c1_bulk_field = self.compute_c1(rho_uniform, functional)
        # Take value at center (far from periodic images)
        c1_bulk = c1_bulk_field[len(z)//2]
        
        if verbose:
            print(f"\n[{functional.name}] η={eta_bulk:.3f}, ρ_bulk={rho_bulk:.4f}")
            print(f"  c1_bulk = {float(c1_bulk):.4f}")
        
        errors = []
        for iteration in range(max_iter):
            rho_old = rho.copy()
            
            c1 = self.compute_c1(rho, functional)
            
            # Update: ρ = ρ_bulk × exp(c1 - c1_bulk)
            rho_new = rho_bulk * jnp.exp(c1 - c1_bulk)
            rho_new = jnp.where(wall_mask, rho_new, 0.0)
            rho_new = jnp.clip(rho_new, 0.0, rho_bulk * 30)
            
            # Adaptive mixing
            if iteration < 100:
                alpha = 0.002
            elif iteration < 500:
                alpha = 0.005
            elif iteration < 2000:
                alpha = 0.01
            else:
                alpha = 0.02
            
            rho = alpha * rho_new + (1 - alpha) * rho_old
            
            error = jnp.max(jnp.abs(rho - rho_old)) / (rho_bulk + 1e-10)
            errors.append(float(error))
            
            if verbose and iteration % 1000 == 0:
                # Contact is first point where z >= R
                contact_idx = int(jnp.argmax(z >= R))
                contact = float(rho[contact_idx]) / rho_bulk
                print(f"  Iter {iteration}: err={error:.2e}, contact={contact:.4f}")
            
            if error < tol:
                break
        
        # Results - contact is first point outside wall
        contact_idx = int(jnp.argmax(z >= R))
        contact = float(rho[contact_idx]) / rho_bulk
        
        if verbose:
            print(f"  Final: contact={contact:.4f} (CS={Z_CS:.4f}, ratio={contact/Z_CS*100:.1f}%)")
        
        return {
            'z': np.array(z),
            'rho': np.array(rho),
            'rho_norm': np.array(rho / rho_bulk),
            'eta_bulk': eta_bulk,
            'contact': contact,
            'contact_CS': Z_CS,
            'converged': error < tol,
            'functional': functional.name,
        }


# ============================================================================
# MD DATA (loaded from data/hswall/ files)
# ============================================================================

def get_mc_profile(eta: float) -> np.ndarray:
    """Get MD wall profile data as [z, rho/rho_bulk] array."""
    from solvers.wall_profile import get_mc_data
    data = get_mc_data(eta)
    if data is None:
        raise ValueError(f"No MD data found for eta={eta}")
    return np.column_stack([data['z'], data['rho'] / data['rho_bulk']])


# ============================================================================
# MAIN
# ============================================================================

def run_validation():
    """Run wall profile validation."""
    print("="*70)
    print("1D FMT WALL PROFILE VALIDATION WITH TENSOR TERMS")
    print("="*70)
    
    solver = WallSolver(nz=2048, Lz=6.0)
    
    # Functionals
    functionals = [
        RosenfeldFMT,
        WhiteBearIIFMT,
        ModifiedRSLT,
        esFMT_Tensor(A=1.0, B=-1.0),
        esFMT_Tensor(A=1.5, B=-1.5),
    ]
    
    eta = 0.367
    mc_data = get_mc_profile(eta)
    
    results = {}
    for func_cls in functionals:
        if isinstance(func_cls, type):
            func = func_cls()
        else:
            func = func_cls
        
        try:
            result = solver.solve(eta, func, max_iter=5000, verbose=True)
            results[func.name] = result
        except Exception as e:
            print(f"  {func.name}: FAILED - {e}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(mc_data[:, 0], mc_data[:, 1], 'ko', ms=6, label='MC', zorder=10)
    
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for i, (name, res) in enumerate(results.items()):
        ax.plot(res['z'], res['rho_norm'], '-', color=colors[i % len(colors)], 
               lw=2, label=f"{name} ({res['contact']:.2f})")
    
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('z/σ', fontsize=12)
    ax.set_ylabel('ρ(z)/ρ_bulk', fontsize=12)
    ax.set_title(f'1D FMT Wall Profiles with Tensor Terms (η = {eta})', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim([0.4, 2.0])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/fmt_1d_wbii_tensor.png', dpi=150)
    print(f"\nSaved: outputs/fmt_1d_wbii_tensor.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    Z_CS = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    mc_contact = mc_data[0, 1]
    
    print(f"\nη = {eta}")
    print(f"CS contact: {Z_CS:.4f}")
    print(f"MC contact: {mc_contact:.4f}")
    print()
    print(f"{'Functional':<25} {'Contact':>10} {'% of MC':>10} {'% of CS':>10}")
    print("-"*55)
    for name, res in results.items():
        pct_mc = res['contact'] / mc_contact * 100
        pct_cs = res['contact'] / Z_CS * 100
        print(f"{name:<25} {res['contact']:10.4f} {pct_mc:9.1f}% {pct_cs:9.1f}%")
    
    return results


if __name__ == "__main__":
    results = run_validation()
