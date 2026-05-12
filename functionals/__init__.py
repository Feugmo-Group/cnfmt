"""
Functionals Module
==================

Free energy functionals for cDFT.

- LutskoFunctional: Lutsko esFMT with parameters (A, B)
- GrandPotential: Grand canonical potential Ω[ρ]
- TestParticlePotential: External potential for test particle geometry
"""

from .lutsko import LutskoFunctional
from .potentials import GrandPotential, TestParticlePotential

__all__ = ['LutskoFunctional', 'GrandPotential', 'TestParticlePotential']
