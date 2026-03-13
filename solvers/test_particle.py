"""
Test Particle Calculator
========================

Computes thermodynamic quantities via test particle insertion method.

The test particle sum rules relate DFT properties to bulk thermodynamics:

Sum Rule 1 (Chemical Potential):
    Ω_s^test = μ_ex^DFT
    
Sum Rule 2 (Compressibility):
    ∫[ρ(r) - ρ_b]dr = -1 + S(k=0) = -1 + χ_T

For exact functionals, DFT and bulk routes agree.
For approximate functionals, the difference measures consistency.

Reference
---------
Gül, Roth, Evans, Phys. Rev. E 110, 064115 (2024)
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Dict, Any, Optional
from jaxtyping import Array

from cnfmt.core.grid import Grid
from cnfmt.core.weights import FMTKernels
from cnfmt.core.densities import WeightedDensityCalculator
from cnfmt.core.thermodynamics import BulkThermodynamics
from cnfmt.functionals.lutsko import LutskoFunctional
from cnfmt.functionals.potentials import GrandPotential, TestParticlePotential
from cnfmt.solvers.minimizer import DensityMinimizer


class TestParticleCalculator(eqx.Module):
    """
    Test particle calculations for sum rule verification.
    
    Parameters
    ----------
    grid : Grid
        Computational grid
    sigma : float
        Hard sphere diameter (default: 1.0)
    A, B : float
        Lutsko parameters
    
    Example
    -------
    >>> calc = TestParticleCalculator(grid, sigma=1.0, A=1.3, B=-1.0)
    >>> result = calc.compute(eta=0.4)
    >>> print(f"δμ = {result['delta_mu']:.4f}, δχ = {result['delta_chi']:.4f}")
    """
    
    grid: Grid
    kernels: FMTKernels
    functional: LutskoFunctional
    calculator: WeightedDensityCalculator
    grand_potential: GrandPotential
    minimizer: DensityMinimizer
    sigma: float = eqx.field(static=True)
    
    def __init__(self, grid: Grid, sigma: float = 1.0,
                 A: float = 1.0, B: float = 0.0):
        """Initialize test particle calculator."""
        self.grid = grid
        self.sigma = sigma
        R = sigma / 2.0
        
        # Build components
        self.kernels = FMTKernels(grid, R)
        self.calculator = WeightedDensityCalculator(self.kernels)
        self.functional = LutskoFunctional(A, B)
        self.grand_potential = GrandPotential(
            self.functional, self.calculator, grid, sigma
        )
        self.minimizer = DensityMinimizer(self.grand_potential)
    
    def with_parameters(self, A: float, B: float) -> 'TestParticleCalculator':
        """Create new calculator with different (A, B)."""
        return TestParticleCalculator(self.grid, self.sigma, A, B)
    
    def compute(self, eta: float, n_steps: int = 300, lr: float = 5e-4,
                verbose: bool = False) -> Dict[str, Any]:
        """
        Compute DFT quantities for given packing fraction.
        
        Parameters
        ----------
        eta : float
            Bulk packing fraction
        n_steps : int
            Minimization steps
        lr : float
            Learning rate for Adam
        verbose : bool
            Print progress
        
        Returns
        -------
        dict with:
            mu_ex_dft, mu_ex_bulk : Chemical potentials
            chi_T_dft, chi_T_bulk : Compressibilities
            delta_mu, delta_chi : Relative deviations
            rho_eq : Equilibrium density profile
            omega : Final grand potential
        """
        R = self.sigma / 2.0
        rho_bulk = eta / ((4.0/3.0) * jnp.pi * R**3)
        
        # Get parameters
        A = self.functional.A_default
        B = self.functional.B_default
        
        # ──────────────────────────────────────────────────────────
        # Bulk thermodynamics
        # ──────────────────────────────────────────────────────────
        mu_ex_bulk = float(BulkThermodynamics.mu_ex_bulk_lutsko(eta, A, B))
        chi_T_bulk = float(BulkThermodynamics.chi_T_bulk_lutsko(eta, A, B))
        Z_bulk = float(BulkThermodynamics.Z_lutsko(eta, A, B))
        P_bulk = rho_bulk * Z_bulk
        
        # Total chemical potential
        mu_total = mu_ex_bulk + jnp.log(rho_bulk * self.sigma**3)
        
        # ──────────────────────────────────────────────────────────
        # Test particle setup
        # ──────────────────────────────────────────────────────────
        test_potential = TestParticlePotential(self.grid, R, R)
        v_ext = test_potential()
        
        # Initialize density (zero inside exclusion zone)
        rho_init = jnp.where(v_ext > 100, 1e-12, rho_bulk)
        
        # Target particle number
        V_excl = test_potential.exclusion_volume
        V_fluid = self.grid.volume - V_excl
        N_target = rho_bulk * V_fluid
        
        # ──────────────────────────────────────────────────────────
        # Minimize grand potential
        # ──────────────────────────────────────────────────────────
        rho_eq, losses = self.minimizer.minimize_adam(
            rho_init, v_ext, mu_total,
            n_steps=n_steps, lr=lr,
            N_target=N_target,
            verbose=verbose
        )
        
        # Final grand potential
        omega_eq = float(self.grand_potential(rho_eq, v_ext, mu_total))
        
        # Reference: Ω₀ = -PV (no test particle)
        omega_0 = -P_bulk * self.grid.volume
        
        # ──────────────────────────────────────────────────────────
        # DFT quantities
        # ──────────────────────────────────────────────────────────
        
        # μ_ex^DFT = Ω_s (test particle grand potential)
        mu_ex_dft = float(omega_eq - omega_0)
        
        # χ_T from density integral
        mask = v_ext < 100  # Outside exclusion zone
        integral = jnp.sum((rho_eq - rho_bulk) * mask) * self.grid.dV
        chi_T_dft = float(jnp.maximum(1.0 + integral / (rho_bulk * self.grid.volume), 1e-6))
        
        # ──────────────────────────────────────────────────────────
        # Relative deviations (sum rule violations)
        # ──────────────────────────────────────────────────────────
        delta_mu = (mu_ex_dft - mu_ex_bulk) / (abs(mu_ex_bulk) + 1e-10)
        delta_chi = (chi_T_dft - chi_T_bulk) / (abs(chi_T_bulk) + 1e-10)
        
        return {
            'eta': eta,
            'mu_ex_dft': mu_ex_dft,
            'mu_ex_bulk': mu_ex_bulk,
            'chi_T_dft': chi_T_dft,
            'chi_T_bulk': chi_T_bulk,
            'delta_mu': delta_mu,
            'delta_chi': delta_chi,
            'rho_eq': rho_eq,
            'omega': omega_eq,
            'losses': losses,
            'A': A,
            'B': B
        }
