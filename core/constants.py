"""
FMT Constants
=============

Reference parameter sets for Fundamental Measure Theory functionals.

Each functional is defined by (A, B) where the constraint parameter
C = 8A + 2B - 9.

References
----------
- Rosenfeld: Phys. Rev. Lett. 63, 980 (1989)
- Lutsko: Phys. Rev. E 74, 021121 (2006)
- White Bear II: J. Phys.: Condens. Matter 18, 8413 (2006)
- Gül et al.: Phys. Rev. E 110, 064115 (2024)
"""

from typing import NamedTuple


class FMTParams(NamedTuple):
    """(A, B) parameter pair for an FMT functional."""
    A: float
    B: float

    @property
    def C(self) -> float:
        """Constraint parameter C = 8A + 2B - 9."""
        return 8 * self.A + 2 * self.B - 9


# Named reference functionals
ROSENFELD = FMTParams(A=1.5, B=0.0)       # C = 3   (PY c(r))
LUTSKO = FMTParams(A=1.0, B=0.0)          # C = -1  (PY line)
WHITE_BEAR_II = FMTParams(A=1.125, B=-1.125)  # C = -3  (CS EOS)
GUL = FMTParams(A=1.3, B=-1.0)            # C = -0.6 (test particle optimized)

# Convenience dict for lookup by name
FMT_PARAMS = {
    'rosenfeld': ROSENFELD,
    'lutsko': LUTSKO,
    'white_bear_ii': WHITE_BEAR_II,
    'wbii': WHITE_BEAR_II,
    'gul': GUL,
}

# Numerical safety constants
EPS_NUMERICAL = 1e-10   # General numerical safety floor
EPS_K_SPACE = 1e-12     # Fourier space regularization
