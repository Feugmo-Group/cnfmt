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
Davidchack, Laird, Roth, Cond. Matt. Phys. 2016 (MD data)
Gül, Roth, Evans, Phys. Rev. E 110, 064115 (2024)
Roth, J. Phys.: Condens. Matter 22, 063102 (2010)

Author: Computational Materials Science
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import equinox as eqx
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


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

        Weight functions for a sphere of radius R:
        w₃(z) = π(R² - z²) Θ(R - |z|)   (volume slice)
        w₂(z) = 2πR Θ(R - |z|)           (surface)
        w₁(z) = 1/2 Θ(R - |z|)           (= w₂/(4πR))
        w₀(z) = 1/(2R) Θ(R - |z|)        (= w₁/R)
        wᵥ₂,z(z) = 2πz Θ(R - |z|)       (vector, z-component)
        wᵥ₁,z(z) = z/(2R) Θ(R - |z|)    (= wᵥ₂/(4πR))
        wT,zz(z) = (2πR/3)(3z²/R² - 1) Θ(R - |z|)  (tensor, zz-component)
        """
        z = self.z
        dz = self.dz
        R = self.R
        n = len(z)

        # Initialize all weighted densities
        n3 = np.zeros(n)
        n2 = np.zeros(n)
        n1 = np.zeros(n)
        n0 = np.zeros(n)
        nv2_z = np.zeros(n)
        nv1_z = np.zeros(n)
        T_zz = np.zeros(n)

        # Convolution via direct summation
        for i in range(n):
            i_lo = max(0, int((z[i] - R) / dz))
            i_hi = min(n-1, int((z[i] + R) / dz))

            for j in range(i_lo, i_hi + 1):
                dz_ij = z[i] - z[j]
                if abs(dz_ij) < R:
                    # Scalar weights
                    w3 = np.pi * (R**2 - dz_ij**2)
                    w2 = 2 * np.pi * R
                    w1 = 0.5
                    w0 = 1.0 / (2 * R)
                    # Vector weights (z-component)
                    wv2_z = 2 * np.pi * dz_ij
                    wv1_z = dz_ij / (2 * R)
                    # Tensor weight (zz-component, traceless)
                    wT_zz = (2 * np.pi * R / 3) * (3 * dz_ij**2 / R**2 - 1)

                    rho_dz = rho[j] * dz
                    n3[i] += rho_dz * w3
                    n2[i] += rho_dz * w2
                    n1[i] += rho_dz * w1
                    n0[i] += rho_dz * w0
                    nv2_z[i] += rho_dz * wv2_z
                    nv1_z[i] += rho_dz * wv1_z
                    T_zz[i] += rho_dz * wT_zz

        # Clip n3 to physical range
        n3 = np.clip(n3, 0.0, 0.9999)

        return {
            'n0': n0, 'n1': n1, 'n2': n2, 'n3': n3,
            'nv1_z': nv1_z, 'nv2_z': nv2_z, 'T_zz': T_zz
        }
    
    def _Phi_lutsko(self, n: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute esFMT free energy density Φ = Φ₁ + Φ₂ + Φ₃.

        Φ₁ = -n₀ ln(1 - n₃)
        Φ₂ = (n₁n₂ - nv1·nv2) / (1 - n₃)
        Φ₃ = (A·term_A + B·term_B) / (24π(1-n₃)²)

        In 1D planar geometry the tensor traces simplify:
          Tr(T²) = 3T_zz²/2,  Tr(T³) = 3T_zz³/4,  nv2·T·nv2 = nv2_z²·T_zz
        """
        n0, n1, n2, n3 = n['n0'], n['n1'], n['n2'], n['n3']
        nv1_z = n.get('nv1_z', np.zeros_like(n0))
        nv2_z = n.get('nv2_z', np.zeros_like(n0))
        T_zz = n.get('T_zz', np.zeros_like(n0))

        one_minus_n3 = np.maximum(1 - n3, 1e-12)

        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2

        # Tensor traces for 1D planar geometry
        T2 = 1.5 * T_zz**2       # Tr(T²)
        T3 = 0.75 * T_zz**3      # Tr(T³)
        vTv = nv2_sq * T_zz       # nv2·T·nv2

        Phi1 = -n0 * np.log(one_minus_n3)
        Phi2 = (n1 * n2 - nv1_dot_nv2) / one_minus_n3

        # A term: n₂³ - 3n₂nv₂² + 3vTv - T³
        term_A = n2**3 - 3*n2*nv2_sq + 3*vTv - T3
        # B term: n₂³ - 3n₂T² + 2T³
        term_B = n2**3 - 3*n2*T2 + 2*T3

        Phi3 = (self.A * term_A + self.B * term_B) / (24 * np.pi * one_minus_n3**2)

        return Phi1 + Phi2 + Phi3
    
    def _c1_from_n(self, n_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute c⁽¹⁾(z) = -δF_ex/δρ(z) for esFMT functional.

        Uses chain rule: c⁽¹⁾(z) = -Σ_α ∫ (∂Φ/∂n_α)(z') w_α(z-z') dz'
        Including scalar, vector, and tensor weighted density contributions.
        """
        n0, n1, n2, n3 = n_dict['n0'], n_dict['n1'], n_dict['n2'], n_dict['n3']
        nv1_z = n_dict.get('nv1_z', np.zeros_like(n0))
        nv2_z = n_dict.get('nv2_z', np.zeros_like(n0))
        T_zz = n_dict.get('T_zz', np.zeros_like(n0))
        R = self.R
        A, B = self.A, self.B

        one_minus_n3 = np.maximum(1 - n3, 1e-12)

        nv1_dot_nv2 = nv1_z * nv2_z
        nv2_sq = nv2_z**2
        T2 = 1.5 * T_zz**2
        T3 = 0.75 * T_zz**3
        vTv = nv2_sq * T_zz
        term_A = n2**3 - 3*n2*nv2_sq + 3*vTv - T3
        term_B = n2**3 - 3*n2*T2 + 2*T3

        # ∂Φ/∂n_α  (matches esFMT_Tensor.dPhi)
        dPhi_dn0 = -np.log(one_minus_n3)
        dPhi_dn1 = n2 / one_minus_n3
        dPhi_dn2 = (n1 / one_minus_n3 +
                     (A*(3*n2**2 - 3*nv2_sq) + B*(3*n2**2 - 3*T2))
                     / (24*np.pi*one_minus_n3**2))
        dPhi_dn3 = (n0 / one_minus_n3 +
                     (n1*n2 - nv1_dot_nv2) / one_minus_n3**2 +
                     2*(A*term_A + B*term_B) / (24*np.pi*one_minus_n3**3))
        dPhi_dnv1_z = -nv2_z / one_minus_n3
        dPhi_dnv2_z = (-nv1_z / one_minus_n3 +
                        A*(-6*n2*nv2_z + 6*T_zz*nv2_z) / (24*np.pi*one_minus_n3**2))
        dPhi_dT_zz = ((A*(3*nv2_sq - 2.25*T_zz**2) +
                        B*(-9*n2*T_zz + 4.5*T_zz**2))
                       / (24*np.pi*one_minus_n3**2))

        # c⁽¹⁾ = -Σ_α (∂Φ/∂n_α ★ w_α)
        ngrid = len(self.z)
        c1 = np.zeros(ngrid)

        for i in range(ngrid):
            i_lo = max(0, int((self.z[i] - R) / self.dz))
            i_hi = min(ngrid - 1, int((self.z[i] + R) / self.dz))

            integral = 0.0
            for j in range(i_lo, i_hi + 1):
                dz_ij = self.z[i] - self.z[j]
                if abs(dz_ij) < R:
                    # Scalar weights
                    w3 = np.pi * (R**2 - dz_ij**2)
                    w2 = 2 * np.pi * R
                    w1 = 0.5
                    w0 = 1.0 / (2 * R)
                    # Vector weights (note sign flip: w_v(z-z') vs w_v(z'-z))
                    wv2_z = -2 * np.pi * dz_ij
                    wv1_z = -dz_ij / (2 * R)
                    # Tensor weight (even function, no sign flip)
                    wT_zz = (2*np.pi*R/3) * (3*dz_ij**2/R**2 - 1)

                    integral += (dPhi_dn0[j] * w0 +
                                 dPhi_dn1[j] * w1 +
                                 dPhi_dn2[j] * w2 +
                                 dPhi_dn3[j] * w3 +
                                 dPhi_dnv1_z[j] * wv1_z +
                                 dPhi_dnv2_z[j] * wv2_z +
                                 dPhi_dT_zz[j] * wT_zz) * self.dz

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
        
        # Bulk reference c1 (vector/tensor terms vanish by symmetry in bulk)
        n_bulk = {
            'n0': np.array([rho_bulk]),
            'n1': np.array([rho_bulk * R]),
            'n2': np.array([rho_bulk * 4 * np.pi * R**2]),
            'n3': np.array([eta]),
            'nv1_z': np.array([0.0]),
            'nv2_z': np.array([0.0]),
            'T_zz': np.array([0.0]),
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
# MD REFERENCE DATA (Davidchack, Laird, Roth, Cond. Matt. Phys. 2016)
# ============================================================================

def _find_data_dir() -> Optional[Path]:
    """Locate the hswall data directory."""
    # Try relative to this file, then relative to cwd
    candidates = [
        Path(__file__).resolve().parent.parent / 'data' / 'hswall',
        Path('data') / 'hswall',
    ]
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _load_md_data_from_files() -> Dict[float, Dict]:
    """Load all MD wall profile data from data/hswall/*.dat files."""
    data_dir = _find_data_dir()
    if data_dir is None:
        return {}

    result = {}
    for dat_file in sorted(data_dir.glob('hswall*.dat')):
        # Parse bulk density from header
        rho_bulk = None
        with open(dat_file) as f:
            for line in f:
                if 'Bulk density' in line:
                    rho_bulk = float(line.split('=')[1].split('(')[0].strip())
                    break
        if rho_bulk is None:
            continue

        eta = rho_bulk * np.pi / 6.0
        data = np.loadtxt(dat_file, comments='%')
        # Columns: z/sigma, rho(z), 95% CI
        result[round(eta, 4)] = {
            'z': data[:, 0],
            'rho': data[:, 1],            # absolute density
            'rho_err': data[:, 2],         # 95% CI
            'rho_bulk': rho_bulk,
            'eta': eta,
            'file': str(dat_file.name),
        }
    return result


# Lazy-loaded cache
_MD_DATA_CACHE = None

def _get_md_cache() -> Dict[float, Dict]:
    global _MD_DATA_CACHE
    if _MD_DATA_CACHE is None:
        _MD_DATA_CACHE = _load_md_data_from_files()
    return _MD_DATA_CACHE


# Backward-compatible dict interface (lazy property)
class _MDWallDataProxy(dict):
    """Lazy dict that loads MD data from files on first access."""
    _loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.update(_get_md_cache())
            self._loaded = True

    def __contains__(self, key):
        self._ensure_loaded()
        # Allow approximate eta matching (within 0.002)
        if super().__contains__(key):
            return True
        for k in super().keys():
            if abs(k - key) < 0.002:
                return True
        return False

    def __getitem__(self, key):
        self._ensure_loaded()
        if super().__contains__(key):
            return super().__getitem__(key)
        # Approximate match
        for k in sorted(super().keys()):
            if abs(k - key) < 0.002:
                return super().__getitem__(k)
        raise KeyError(key)

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def items(self):
        self._ensure_loaded()
        return super().items()

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()


MC_WALL_DATA = _MDWallDataProxy()


def get_mc_data(eta: float) -> Optional[Dict]:
    """Get MD reference data for given eta (backward compatible)."""
    cache = _get_md_cache()
    if not cache:
        return None

    available = list(cache.keys())
    closest = min(available, key=lambda x: abs(x - eta))

    if abs(closest - eta) < 0.01:
        data = cache[closest]
        return {
            'z': data['z'],
            'rho': data['rho'],
            'rho_bulk': data['rho_bulk'],
            'eta': data['eta'],
        }
    return None
