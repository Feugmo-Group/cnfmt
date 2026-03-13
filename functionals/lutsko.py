"""
Lutsko esFMT Functional
=======================

The Lutsko functional is a parameterized extension of Rosenfeld FMT
that allows adjustment of bulk thermodynamics while maintaining
the correct dimensional crossover properties.

Excess Free Energy Density
--------------------------
The excess free energy density Φ = Φ₁ + Φ₂ + Φ₃ where:

Φ₁ = -n₀ ln(1 - η)

Φ₂ = (n₁n₂ - nᵥ₁·nᵥ₂) / (1 - η)

Φ₃ = [1 / 24π(1-η)²] × {
    (A+B) n₂³ 
    - 3A n₂(nᵥ₂·nᵥ₂) 
    + 3A (nᵥ₂·T·nᵥ₂) 
    - 3B n₂ Tr(T²) 
    + (2B-A) Tr(T³)
}

Parameters
----------
A, B : Lutsko parameters controlling EOS deviation from PY
- A = 1, B = 0: Lutsko's original (C = -1)
- A = 1.3, B = -1.0: Optimal from test particle sum rules (C = -0.6)
- A = 3/2, B = -3/2: Rosenfeld tensor (C = 0, PY line)

Constraint: C = 8A + 2B - 9 determines pressure correction

Reference
---------
J.F. Lutsko, Phys. Rev. E 102, 062137 (2020)
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union
from jaxtyping import Array
from cnfmt.core.densities import WeightedDensities


class LutskoFunctional(eqx.Module):
    """
    Lutsko esFMT functional with parameters (A, B).
    
    Supports both scalar and field-valued parameters.
    
    Parameters
    ----------
    A : float
        First Lutsko parameter (default: 1.0)
    B : float
        Second Lutsko parameter (default: 0.0)
    
    Attributes
    ----------
    A_default, B_default : float
        Default parameter values
    
    Example
    -------
    >>> functional = LutskoFunctional(A=1.3, B=-1.0)
    >>> Phi = functional.free_energy_density(measures)
    >>> F_ex = jnp.sum(Phi) * dV
    """
    
    A_default: float = eqx.field(static=True)
    B_default: float = eqx.field(static=True)
    
    def __init__(self, A: float = 1.0, B: float = 0.0):
        """Initialize Lutsko functional with parameters."""
        self.A_default = A
        self.B_default = B
    
    def free_energy_density(self, 
                            measures: WeightedDensities,
                            A: Optional[Union[float, Array]] = None,
                            B: Optional[Union[float, Array]] = None) -> Array:
        """
        Compute excess free energy density Φ(r).
        
        Parameters
        ----------
        measures : WeightedDensities
            Weighted densities from density calculator
        A, B : float or Array, optional
            Override default parameters. Can be spatially varying.
        
        Returns
        -------
        Phi : Array
            Excess free energy density βΦ(r)
        """
        # Use provided or default parameters
        A = self.A_default if A is None else A
        B = self.B_default if B is None else B
        
        # Unpack weighted densities
        eta = measures.eta
        n0, n1, n2 = measures.n0, measures.n1, measures.n2
        nv1_dot_nv2 = measures.nv1_dot_nv2
        nv2_sq = measures.nv2_sq
        T2, T3 = measures.T2, measures.T3
        nvTnv = measures.nvTnv
        
        # Regularization for stability
        eta_safe = jnp.clip(eta, 1e-12, 1 - 1e-8)
        one_minus_eta = 1 - eta_safe
        
        # ──────────────────────────────────────────────────────────
        # Φ₁: Ideal gas contribution
        # ──────────────────────────────────────────────────────────
        Phi1 = -n0 * jnp.log(one_minus_eta)
        
        # ──────────────────────────────────────────────────────────
        # Φ₂: Two-body contribution
        # ──────────────────────────────────────────────────────────
        Phi2 = (n1 * n2 - nv1_dot_nv2) / one_minus_eta
        
        # ──────────────────────────────────────────────────────────
        # Φ₃: Three-body (Lutsko parameterized)
        # ──────────────────────────────────────────────────────────
        # 
        # Φ₃ = [1 / 24π(1-η)²] × {
        #     (A+B) n₂³ 
        #     - 3A n₂(nᵥ₂·nᵥ₂) 
        #     + 3A (nᵥ₂·T·nᵥ₂) 
        #     - 3B n₂ Tr(T²) 
        #     + (2B-A) Tr(T³)
        # }
        
        prefactor = 1.0 / (24 * jnp.pi * one_minus_eta**2)
        
        term1 = (A + B) * n2**3
        term2 = -3 * A * n2 * nv2_sq
        term3 = 3 * A * nvTnv
        term4 = -3 * B * n2 * T2
        term5 = (2*B - A) * T3
        
        Phi3 = prefactor * (term1 + term2 + term3 + term4 + term5)
        
        # Total free energy density
        Phi = Phi1 + Phi2 + Phi3
        
        # Handle unphysical regions (η ≥ 1)
        Phi = jnp.where(eta >= 0.999, 1e10, Phi)
        
        return Phi
    
    def constraint_value(self, A: Optional[float] = None, B: Optional[float] = None) -> float:
        """
        Compute constraint C = 8A + 2B - 9.
        
        C determines EOS deviation from Percus-Yevick:
        - C = 0: PY (Rosenfeld)
        - C = -0.6: Optimal (Gül et al.)
        - C = -1: Lutsko original
        - C = -3: Carnahan-Starling (White Bear)
        """
        A = self.A_default if A is None else A
        B = self.B_default if B is None else B
        return 8*A + 2*B - 9
    
    def with_parameters(self, A: float, B: float) -> 'LutskoFunctional':
        """Create new functional with different parameters."""
        return LutskoFunctional(A, B)
    
    def __repr__(self) -> str:
        C = self.constraint_value()
        return f"LutskoFunctional(A={self.A_default}, B={self.B_default}, C={C:.2f})"
