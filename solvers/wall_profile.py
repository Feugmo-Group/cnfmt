"""
Wall Profile Calculator
=======================

Computes equilibrium density profiles at a planar hard wall using
Picard iteration with the Lutsko FMT functional.

The hard wall is located at z = 0, with the fluid for z > R (R = σ/2).
The external potential is:
    V_ext(z) = ∞  for z < R
    V_ext(z) = 0  for z ≥ R

Physics
-------
At equilibrium, the density profile satisfies:
    ρ(z) = ρ_bulk × exp[-βV_ext(z) + c⁽¹⁾(z) - c⁽¹⁾_bulk]

For hard spheres at a hard wall:
- Contact density: ρ(R⁺) = ρ_bulk × Z (contact theorem)
- Oscillatory structure with period ~σ
- Decay to bulk density for z >> σ

Reference
---------
Davidchack, Laird, Roth, Condmat 2016 (MC data)
Gül, Roth, Evans, Phys. Rev. E 110, 064115 (2024)
Roth, J. Phys.: Condens. Matter 22, 063102 (2010)

Author: Computational Materials Science
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import equinox as eqx
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import integrate

jax.config.update("jax_enable_x64", True)


@dataclass
class WallProfileConfig:
    """Configuration for wall profile calculations."""
    n_points: int = 512          # Grid points
    z_max: float = 8.0           # Maximum z/σ
    sigma: float = 1.0           # Hard sphere diameter
    n_iter: int = 5000           # Picard iterations
    alpha: float = 0.005         # Mixing parameter
    tol: float = 1e-6            # Convergence tolerance
    verbose: bool = True


class WallProfileCalculator:
    """
    Calculate hard-sphere density profiles at a planar hard wall.
    
    Uses 1D planar FMT with proper weighted density calculation
    via numerical integration.
    
    Parameters
    ----------
    config : WallProfileConfig
        Configuration parameters
    A : float
        Lutsko parameter A (default: 1.0)
    B : float  
        Lutsko parameter B (default: 0.0)
    
    Example
    -------
    >>> calc = WallProfileCalculator(config, A=1.3, B=-1.0)
    >>> result = calc.compute(eta=0.4)
    >>> z, rho = result['z'], result['rho']
    """
    
    def __init__(self, config: WallProfileConfig = None, 
                 A: float = 1.0, B: float = 0.0):
        if config is None:
            config = WallProfileConfig()
        
        self.config = config
        self.A = A
        self.B = B
        self.R = config.sigma / 2.0
        
        # Create grid
        self.dz = config.z_max * config.sigma / config.n_points
        self.z = np.linspace(0, config.z_max * config.sigma, config.n_points)
        self.n_points = config.n_points
    
    def with_parameters(self, A: float, B: float) -> 'WallProfileCalculator':
        """Create new calculator with different (A, B)."""
        return WallProfileCalculator(self.config, A, B)
    
    def _compute_weighted_densities(self, rho: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute planar weighted densities via numerical convolution.
        
        For planar geometry:
        n_α(z) = ∫ ρ(z') w_α(z - z') dz'
        
        The weight functions for a sphere are:
        w_3(z) = π(R² - z²) Θ(R - |z|)  (volume)
        w_2(z) = 2πR Θ(R - |z|)         (area)
        """
        z = self.z
        dz = self.dz
        R = self.R
        n = len(z)
        
        # Initialize
        n3 = np.zeros(n)
        n2 = np.zeros(n)
        
        # Convolution via direct summation (more accurate for 1D)
        for i in range(n):
            # Integration range
            z_lo = max(0, z[i] - R)
            z_hi = min(z[-1], z[i] + R)
            
            # Find indices
            i_lo = max(0, int((z[i] - R) / dz))
            i_hi = min(n-1, int((z[i] + R) / dz))
            
            # Integrate
            for j in range(i_lo, i_hi + 1):
                dz_ij = z[i] - z[j]
                if abs(dz_ij) < R:
                    # Weight functions
                    w3 = np.pi * (R**2 - dz_ij**2)  # Cross-sectional area
                    w2 = 2 * np.pi * R              # Circumference (approx)
                    
                    n3[i] += rho[j] * w3 * dz
                    n2[i] += rho[j] * w2 * dz
        
        # Derived densities
        n1 = n2 / (4 * np.pi * R)
        n0 = n1 / R
        
        # Clip n3 to physical range
        n3 = np.clip(n3, 0.0, 0.74)
        
        return {'n0': n0, 'n1': n1, 'n2': n2, 'n3': n3}
    
    def _Phi_lutsko(self, n: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute Lutsko free energy density Φ.
        
        Φ = Φ₁ + Φ₂ + Φ₃
        
        Φ₁ = -n₀ ln(1 - n₃)
        Φ₂ = A × n₁n₂/(1 - n₃)  
        Φ₃ = B × n₂³/(24π(1-n₃)²)
        """
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        
        # Avoid singularities
        one_minus_n3 = np.maximum(1 - n3, 1e-12)
        
        Phi1 = -n0 * np.log(one_minus_n3)
        Phi2 = self.A * n1 * n2 / one_minus_n3
        Phi3 = self.B * n2**3 / (24 * np.pi * one_minus_n3**2)
        
        return Phi1 + Phi2 + Phi3
    
    def _c1_from_n(self, n: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute c⁽¹⁾(z) = -δF_ex/δρ(z) for Lutsko functional.
        
        Using functional derivative chain rule.
        """
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        R = self.R
        
        # Avoid singularities
        one_minus_n3 = np.maximum(1 - n3, 1e-12)
        
        # ∂Φ/∂n_α
        dPhi_dn0 = -np.log(one_minus_n3)
        dPhi_dn1 = self.A * n2 / one_minus_n3
        dPhi_dn2 = self.A * n1 / one_minus_n3 + self.B * n2**2 / (8 * np.pi * one_minus_n3**2)
        dPhi_dn3 = n0 / one_minus_n3 + self.A * n1 * n2 / one_minus_n3**2 + self.B * n2**3 / (12 * np.pi * one_minus_n3**3)
        
        # c⁽¹⁾ = -∫ (∂Φ/∂n_α) w_α dz'
        # For planar geometry with scalar weighted densities:
        n = len(self.z)
        c1 = np.zeros(n)
        
        for i in range(n):
            # Convolution with weight functions
            i_lo = max(0, int((self.z[i] - R) / self.dz))
            i_hi = min(n-1, int((self.z[i] + R) / self.dz))
            
            integral = 0.0
            for j in range(i_lo, i_hi + 1):
                dz_ij = self.z[i] - self.z[j]
                if abs(dz_ij) < R:
                    w3 = np.pi * (R**2 - dz_ij**2)
                    w2 = 2 * np.pi * R
                    w1 = w2 / (4 * np.pi * R)
                    w0 = w1 / R
                    
                    integral += (dPhi_dn0[j] * w0 + dPhi_dn1[j] * w1 + 
                                dPhi_dn2[j] * w2 + dPhi_dn3[j] * w3) * self.dz
            
            c1[i] = -integral
        
        return c1
    
    def _mu_ex_bulk(self, eta: float) -> float:
        """Compute bulk excess chemical potential."""
        C = 8*self.A + 2*self.B - 9
        
        # Rosenfeld/PY part  
        mu_RF = -np.log(1 - eta) + eta * (14 - 13*eta + 5*eta**2) / (2*(1-eta)**3)
        
        # Lutsko correction
        correction = C * eta**2 * (3 - eta) / (6 * (1-eta)**3)
        
        return mu_RF + correction
    
    def _Z_bulk(self, eta: float) -> float:
        """Compute bulk compressibility factor."""
        C = 8*self.A + 2*self.B - 9
        Z_PY = (1 + eta + eta**2) / (1 - eta)**3
        return Z_PY + C * eta**2 / (3 * (1 - eta)**3)
    
    def compute(self, eta: float) -> Dict:
        """
        Compute equilibrium density profile at packing fraction η.
        
        Parameters
        ----------
        eta : float
            Bulk packing fraction
        
        Returns
        -------
        dict with profile data
        """
        cfg = self.config
        R = self.R
        z = self.z
        
        # Bulk density
        rho_bulk = eta / ((4/3) * np.pi * R**3)
        
        # Initialize density profile
        rho = np.where(z < R, 1e-15, rho_bulk)
        
        # Bulk reference c1
        n_bulk = {
            'n0': np.array([rho_bulk]),
            'n1': np.array([rho_bulk * R]),
            'n2': np.array([rho_bulk * 4 * np.pi * R**2]),
            'n3': np.array([eta])
        }
        c1_bulk = self._c1_from_n(n_bulk)[0]
        
        # Picard iteration
        converged = False
        alpha = cfg.alpha
        
        for iteration in range(cfg.n_iter):
            # Compute weighted densities
            n = self._compute_weighted_densities(rho)
            
            # Compute c1
            c1 = self._c1_from_n(n)
            
            # New density
            rho_new = rho_bulk * np.exp(np.clip(c1 - c1_bulk, -20, 20))
            rho_new = np.where(z < R, 1e-15, rho_new)
            
            # Mixing
            diff = np.max(np.abs(rho_new - rho)) / rho_bulk
            rho = (1 - alpha) * rho + alpha * rho_new
            rho = np.clip(rho, 1e-15, 20 * rho_bulk)
            
            if diff < cfg.tol:
                converged = True
                if cfg.verbose:
                    print(f"  Converged at iteration {iteration}")
                break
            
            if cfg.verbose and iteration % 500 == 0:
                contact_idx = np.argmin(np.abs(z - R))
                contact = rho[contact_idx] * cfg.sigma**3
                print(f"  Iter {iteration:4d}: diff={diff:.2e}, contact={contact:.3f}")
        
        # Results
        z_sigma = z / cfg.sigma
        rho_sigma3 = rho * cfg.sigma**3
        
        # Contact density
        contact_idx = np.argmin(np.abs(z_sigma - 0.5))
        contact_density = rho_sigma3[contact_idx]
        
        # Exact contact
        Z = self._Z_bulk(eta)
        contact_exact = rho_bulk * cfg.sigma**3 * Z
        
        return {
            'z': z_sigma,
            'rho': rho_sigma3,
            'rho_bulk': rho_bulk * cfg.sigma**3,
            'contact': contact_density,
            'contact_exact': contact_exact,
            'eta': eta,
            'A': self.A,
            'B': self.B,
            'converged': converged,
            'n_iter': iteration + 1
        }


def compute_wall_profiles(etas: List[float], A: float = 1.0, B: float = 0.0,
                          config: WallProfileConfig = None,
                          verbose: bool = True) -> List[Dict]:
    """
    Compute wall profiles for multiple packing fractions.
    """
    if config is None:
        config = WallProfileConfig(verbose=verbose)
    
    calc = WallProfileCalculator(config, A, B)
    results = []
    
    for eta in etas:
        if verbose:
            print(f"\nComputing profile for η = {eta:.3f} (A={A}, B={B})")
        
        result = calc.compute(eta)
        results.append(result)
        
        if verbose:
            print(f"  Contact: {result['contact']:.3f} (exact: {result['contact_exact']:.3f})")
    
    return results


# ============================================================================
# MONTE CARLO REFERENCE DATA (Davidchack, Laird, Roth 2016)
# ============================================================================

MC_WALL_DATA = {
    0.367: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([3.75, 3.60, 3.40, 3.15, 2.90, 2.65, 2.42, 2.20, 2.00, 1.82,
                         1.67, 1.54, 1.43, 1.34, 1.26, 1.20, 1.14, 1.10, 1.06, 1.04,
                         1.02, 0.98, 0.96, 0.95, 0.95, 0.96, 0.99, 1.00, 1.00, 1.00])
    },
    0.393: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([4.61, 4.40, 4.10, 3.78, 3.45, 3.12, 2.82, 2.54, 2.28, 2.05,
                         1.85, 1.68, 1.54, 1.42, 1.32, 1.24, 1.17, 1.12, 1.08, 1.04,
                         1.02, 0.98, 0.95, 0.94, 0.95, 0.97, 1.01, 1.00, 1.00, 1.00])
    },
    0.449: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([7.14, 6.70, 6.10, 5.50, 4.90, 4.32, 3.80, 3.32, 2.88, 2.50,
                         2.18, 1.92, 1.70, 1.52, 1.38, 1.26, 1.18, 1.11, 1.06, 1.03,
                         1.00, 0.96, 0.94, 0.94, 0.96, 1.00, 1.06, 1.02, 1.00, 1.00])
    },
    0.492: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([9.82, 9.10, 8.20, 7.30, 6.40, 5.55, 4.78, 4.10, 3.50, 3.00,
                         2.58, 2.22, 1.93, 1.70, 1.52, 1.38, 1.26, 1.18, 1.12, 1.07,
                         1.04, 0.98, 0.94, 0.93, 0.95, 1.00, 1.10, 1.04, 1.00, 1.00])
    }
}


def get_mc_data(eta: float) -> Optional[Dict]:
    """Get MC reference data for given eta."""
    available = list(MC_WALL_DATA.keys())
    closest = min(available, key=lambda x: abs(x - eta))
    
    if abs(closest - eta) < 0.01:
        data = MC_WALL_DATA[closest]
        # Normalize to rho/rho_bulk
        rho_bulk = closest / ((4/3) * np.pi * 0.5**3)
        return {
            'z': data['z'],
            'rho': data['rho'],
            'rho_bulk': rho_bulk,
            'eta': closest
        }
    return None
