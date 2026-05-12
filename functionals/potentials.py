"""
Grand Potential and External Potentials
=======================================

Grand Canonical Potential
-------------------------
The grand potential functional is:

    Ω[ρ] = F_id[ρ] + F_ex[ρ] + ∫ρ(r)[V_ext(r) - μ]dr

where:
- F_id[ρ] = kT ∫ρ(r)[ln(ρ(r)Λ³) - 1]dr  (ideal gas)
- F_ex[ρ] = kT ∫Φ[{nα(r)}]dr  (excess from FMT)
- V_ext(r): external potential
- μ: chemical potential

At equilibrium: δΩ/δρ = 0

Test Particle Potential
-----------------------
For test particle calculations:

    V_ext(r) = ∞  for |r - r₀| < R_test + R_fluid
             = 0   otherwise

This represents a hard sphere centered at r₀.
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union
from jaxtyping import Array
from core.grid import Grid
from core.densities import WeightedDensityCalculator
from .lutsko import LutskoFunctional


class GrandPotential(eqx.Module):
    """
    Grand canonical potential functional Ω[ρ].
    
    Parameters
    ----------
    functional : LutskoFunctional
        Excess free energy functional
    calculator : WeightedDensityCalculator
        Weighted density calculator
    grid : Grid
        Computational grid
    sigma : float
        Hard sphere diameter
    
    Example
    -------
    >>> omega = GrandPotential(functional, calculator, grid, sigma=1.0)
    >>> Omega = omega(rho, v_ext, mu)
    """
    
    functional: LutskoFunctional
    calculator: WeightedDensityCalculator
    grid: Grid
    sigma: float = eqx.field(static=True)
    
    def __init__(self, functional: LutskoFunctional,
                 calculator: WeightedDensityCalculator,
                 grid: Grid, sigma: float = 1.0):
        self.functional = functional
        self.calculator = calculator
        self.grid = grid
        self.sigma = sigma
    
    def ideal_free_energy(self, rho: Array) -> float:
        """
        Ideal gas free energy.
        
        F_id = ∫ρ(r)[ln(ρΛ³) - 1]dr
        
        Using Λ = σ for hard spheres.
        """
        eps = 1e-12
        rho_safe = jnp.maximum(rho, eps)
        
        # f_id = ρ[ln(ρσ³) - 1]
        f_id = rho_safe * (jnp.log(rho_safe * self.sigma**3) - 1)
        
        return jnp.sum(f_id) * self.grid.dV
    
    def excess_free_energy(self, rho: Array,
                           A: Optional[Union[float, Array]] = None,
                           B: Optional[Union[float, Array]] = None) -> float:
        """
        Excess free energy from FMT.
        
        F_ex = ∫Φ(r)dr
        """
        measures = self.calculator(rho)
        Phi = self.functional.free_energy_density(measures, A, B)
        return jnp.sum(Phi) * self.grid.dV
    
    def external_contribution(self, rho: Array, v_ext: Array, mu: float) -> float:
        """
        External potential and chemical potential contribution.
        
        ∫ρ(r)[V_ext(r) - μ]dr
        """
        return jnp.sum(rho * (v_ext - mu)) * self.grid.dV
    
    def __call__(self, rho: Array, v_ext: Array, mu: float,
                 A: Optional[Union[float, Array]] = None,
                 B: Optional[Union[float, Array]] = None) -> float:
        """
        Compute grand potential Ω[ρ].
        
        Parameters
        ----------
        rho : Array
            Density field
        v_ext : Array
            External potential
        mu : float
            Chemical potential (βμ)
        A, B : optional
            Override Lutsko parameters
        
        Returns
        -------
        Omega : float
            Grand potential (βΩ)
        """
        F_id = self.ideal_free_energy(rho)
        F_ex = self.excess_free_energy(rho, A, B)
        ext = self.external_contribution(rho, v_ext, mu)
        
        return F_id + F_ex + ext
    
    def with_particle_constraint(self, rho: Array, v_ext: Array, mu: float,
                                  N_target: float, lambda_N: float = 10.0,
                                  A: Optional[float] = None,
                                  B: Optional[float] = None) -> float:
        """
        Grand potential with particle number constraint.
        
        Ω_constrained = Ω + λ(N - N_target)²
        """
        omega = self(rho, v_ext, mu, A, B)
        N = jnp.sum(rho) * self.grid.dV
        constraint = lambda_N * (N - N_target)**2
        return omega + constraint


class TestParticlePotential(eqx.Module):
    """
    External potential for test particle calculations.
    
    Creates a hard sphere exclusion zone at the box center.
    
    Parameters
    ----------
    grid : Grid
        Computational grid
    R_test : float
        Test particle radius
    R_fluid : float
        Fluid particle radius
    V_wall : float
        Value inside exclusion zone (effectively infinity)
    
    Example
    -------
    >>> potential = TestParticlePotential(grid, R_test=0.5, R_fluid=0.5)
    >>> v_ext = potential()
    """
    
    grid: Grid
    R_test: float = eqx.field(static=True)
    R_fluid: float = eqx.field(static=True)
    V_wall: float = eqx.field(static=True)
    
    def __init__(self, grid: Grid, R_test: float, R_fluid: float,
                 V_wall: float = 1e6):
        self.grid = grid
        self.R_test = R_test
        self.R_fluid = R_fluid
        self.V_wall = V_wall
    
    def __call__(self, center: Optional[tuple] = None) -> Array:
        """
        Create test particle potential.
        
        Parameters
        ----------
        center : tuple, optional
            Center position (cx, cy, cz). Default: box center.
        
        Returns
        -------
        v_ext : Array
            External potential field
        """
        if center is None:
            cx, cy, cz = self.grid.center
        else:
            cx, cy, cz = center
        
        # Distance from center
        r = jnp.sqrt((self.grid.X - cx)**2 + 
                     (self.grid.Y - cy)**2 + 
                     (self.grid.Z - cz)**2)
        
        # Exclusion radius
        R_exclusion = self.R_test + self.R_fluid
        
        # Hard wall potential
        v_ext = jnp.where(r < R_exclusion, self.V_wall, 0.0)
        
        return v_ext
    
    @property
    def exclusion_radius(self) -> float:
        """Total exclusion radius."""
        return self.R_test + self.R_fluid
    
    @property
    def exclusion_volume(self) -> float:
        """Volume of exclusion zone."""
        return (4.0 / 3.0) * jnp.pi * self.exclusion_radius**3


class PlanarWallPotential(eqx.Module):
    """
    External potential for planar hard wall.
    
    Wall at z = z_wall with exclusion zone z < z_wall + R.
    """
    
    grid: Grid
    z_wall: float = eqx.field(static=True)
    R_fluid: float = eqx.field(static=True)
    V_wall: float = eqx.field(static=True)
    
    def __init__(self, grid: Grid, z_wall: float = 0.0,
                 R_fluid: float = 0.5, V_wall: float = 1e6):
        self.grid = grid
        self.z_wall = z_wall
        self.R_fluid = R_fluid
        self.V_wall = V_wall
    
    def __call__(self) -> Array:
        """Create planar wall potential."""
        return jnp.where(self.grid.Z < self.z_wall + self.R_fluid, 
                        self.V_wall, 0.0)
