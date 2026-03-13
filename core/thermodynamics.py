"""
Bulk Thermodynamics
===================

Analytical formulas for bulk hard-sphere thermodynamics.

Equations of State
------------------
Percus-Yevick (PY):
    Z_PY = (1 + η + η²) / (1 - η)³

Carnahan-Starling (CS):
    Z_CS = (1 + η + η² - η³) / (1 - η)³

Lutsko (parameterized):
    Z_LK = Z_PY + [η² / 3(1-η)³] × (8A + 2B - 9)

Chemical Potential
------------------
Rosenfeld:
    βμ_ex^RF = -ln(1-η) + η(14-13η+5η²) / [2(1-η)³]

Lutsko:
    βμ_ex^LK = βμ_ex^RF + [η²(3-η) / 6(1-η)³] × (8A + 2B - 9)

Compressibility
---------------
    χ_T^RF = (1-η)⁴ / (1+2η)²
    χ_T^LK = χ_T^RF × (1+2η)² / [(1+2η)² - (8A+2B-9)η²]

Reference
---------
Gül, Roth, Evans, Phys. Rev. E 110, 064115 (2024)
"""

import jax.numpy as jnp
from jaxtyping import Array


class BulkThermodynamics:
    """
    Analytical bulk thermodynamic formulas for hard spheres.
    
    All methods are static and work with both scalars and arrays.
    
    Example
    -------
    >>> eta = 0.4
    >>> Z = BulkThermodynamics.Z_CS(eta)
    >>> mu = BulkThermodynamics.mu_ex_CS(eta)
    >>> print(f"Z={Z:.4f}, βμ_ex={mu:.4f}")
    """
    
    # ══════════════════════════════════════════════════════════════
    # EQUATIONS OF STATE
    # ══════════════════════════════════════════════════════════════
    
    @staticmethod
    def Z_PY(eta: Array) -> Array:
        """
        Percus-Yevick compressibility factor.
        
        Z_PY = (1 + η + η²) / (1 - η)³
        
        Slightly overestimates pressure at high η.
        """
        return (1 + eta + eta**2) / (1 - eta)**3
    
    @staticmethod
    def Z_CS(eta: Array) -> Array:
        """
        Carnahan-Starling compressibility factor.
        
        Z_CS = (1 + η + η² - η³) / (1 - η)³
        
        Most accurate EOS for hard spheres.
        """
        return (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    
    @staticmethod
    def Z_lutsko(eta: Array, A: Array, B: Array) -> Array:
        """
        Lutsko equation of state (Eq. 20 in paper).
        
        Z_LK = Z_PY + [η² / 3(1-η)³] × C
        
        where C = 8A + 2B - 9 is the constraint parameter.
        
        Special cases:
        - C = 0 (PY line): Z_LK = Z_PY
        - C = -3 (White Bear): Z_LK = Z_CS
        """
        Z_py = BulkThermodynamics.Z_PY(eta)
        C = 8*A + 2*B - 9
        correction = eta**2 / (3 * (1-eta)**3) * C
        return Z_py + correction
    
    # ══════════════════════════════════════════════════════════════
    # CHEMICAL POTENTIAL
    # ══════════════════════════════════════════════════════════════
    
    @staticmethod
    def mu_ex_RF(eta: Array) -> Array:
        """
        Rosenfeld excess chemical potential (Eq. 25).
        
        βμ_ex^RF = -ln(1-η) + η(14 - 13η + 5η²) / [2(1-η)³]
        
        Derived from PY direct correlation function.
        """
        return -jnp.log(1 - eta) + eta * (14 - 13*eta + 5*eta**2) / (2*(1-eta)**3)
    
    @staticmethod
    def mu_ex_CS(eta: Array) -> Array:
        """
        Carnahan-Starling excess chemical potential.
        
        βμ_ex^CS = η(8 - 9η + 3η²) / (1 - η)³
        """
        return eta * (8 - 9*eta + 3*eta**2) / (1 - eta)**3
    
    @staticmethod
    def mu_ex_bulk_lutsko(eta: Array, A: Array, B: Array) -> Array:
        """
        Lutsko excess chemical potential - BULK route (Eq. 24).
        
        βμ_ex^LK = βμ_ex^RF + [η²(3-η) / 6(1-η)³] × C
        
        where C = 8A + 2B - 9.
        """
        mu_RF = BulkThermodynamics.mu_ex_RF(eta)
        C = 8*A + 2*B - 9
        correction = (eta**2 * (3 - eta)) / (6 * (1-eta)**3) * C
        return mu_RF + correction
    
    # ══════════════════════════════════════════════════════════════
    # COMPRESSIBILITY
    # ══════════════════════════════════════════════════════════════
    
    @staticmethod
    def chi_T_RF(eta: Array) -> Array:
        """
        Rosenfeld isothermal compressibility (Eq. 27).
        
        χ_T^RF = (1 - η)⁴ / (1 + 2η)²
        
        From PY compressibility route.
        """
        return (1 - eta)**4 / (1 + 2*eta)**2
    
    @staticmethod
    def chi_T_CS(eta: Array) -> Array:
        """
        Carnahan-Starling isothermal compressibility.
        
        χ_T^CS = (1 - η)⁴ / (1 + 4η + 4η² - 4η³ + η⁴)
        """
        numerator = (1 - eta)**4
        denominator = 1 + 4*eta + 4*eta**2 - 4*eta**3 + eta**4
        return numerator / jnp.maximum(denominator, 1e-10)
    
    @staticmethod
    def chi_T_bulk_lutsko(eta: Array, A: Array, B: Array) -> Array:
        """
        Lutsko isothermal compressibility - BULK route (Eq. 26).
        
        χ_T^LK = χ_T^RF × (1+2η)² / [(1+2η)² - C·η²]
        
        where C = 8A + 2B - 9.
        
        Note: Uses MINUS sign in denominator (see paper Figure 4b).
        """
        chi_RF = BulkThermodynamics.chi_T_RF(eta)
        C = 8*A + 2*B - 9
        numerator = (1 + 2*eta)**2
        denominator = (1 + 2*eta)**2 - C * eta**2
        return chi_RF * numerator / jnp.maximum(denominator, 1e-10)
    
    # ══════════════════════════════════════════════════════════════
    # PRESSURE
    # ══════════════════════════════════════════════════════════════
    
    @staticmethod
    def pressure_CS(eta: Array, rho: Array) -> Array:
        """
        Carnahan-Starling pressure.
        
        βP = ρ × Z_CS
        """
        return rho * BulkThermodynamics.Z_CS(eta)
    
    @staticmethod
    def pressure_lutsko(eta: Array, rho: Array, A: Array, B: Array) -> Array:
        """
        Lutsko pressure.
        
        βP = ρ × Z_LK
        """
        return rho * BulkThermodynamics.Z_lutsko(eta, A, B)
