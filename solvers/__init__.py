"""
Solvers Module
==============

Density minimizers, test particle, and wall profile calculations.

- DensityMinimizer: Adam and L-BFGS optimizers for Ω[ρ]
- TestParticleCalculator: Sum rule calculations via test particle
- WallProfileCalculator: Hard sphere profiles at planar hard wall

Validated FMT implementations:
- fmt_1d_wbii_tensor: 1D planar with tensor terms (RECOMMENDED)
- fmt_3d_tensor: Full 3D with tensor terms
"""

from cnfmt.solvers.minimizer import DensityMinimizer
from cnfmt.solvers.test_particle import TestParticleCalculator
from cnfmt.solvers.wall_profile import (
    WallProfileCalculator,
    WallProfileConfig,
    compute_wall_profiles,
    get_mc_data,
    MC_WALL_DATA
)

# Validated 1D FMT with tensor terms
from cnfmt.solvers.fmt_1d_wbii_tensor import (
    Weights1D,
    RosenfeldFMT,
    WhiteBearIIFMT,
    ModifiedRSLT,
    esFMT_Tensor,
    WallSolver,
    phi2_WBII,
    phi3_WBII,
    get_mc_profile
)

# Validated 3D FMT with tensor terms  
from cnfmt.solvers.fmt_3d_tensor import (
    Grid3D,
    FMTWeights3D,
    WeightedDensities,
    RosenfeldFMT as RosenfeldFMT3D,
    WhiteBearIIFMT as WhiteBearIIFMT3D,
    esFMT_Tensor as esFMT_Tensor3D,
    WBII_Tensor,
    DFTSolver3D,
    get_mc_wall_profile
)

__all__ = [
    # Original modules
    'DensityMinimizer', 
    'TestParticleCalculator',
    'WallProfileCalculator',
    'WallProfileConfig',
    'compute_wall_profiles',
    'get_mc_data',
    'MC_WALL_DATA',
    # Validated 1D FMT (recommended)
    'Weights1D',
    'RosenfeldFMT',
    'WhiteBearIIFMT',
    'ModifiedRSLT',
    'esFMT_Tensor',
    'WallSolver',
    'phi2_WBII',
    'phi3_WBII',
    'get_mc_profile',
    # Validated 3D FMT
    'Grid3D',
    'FMTWeights3D',
    'WeightedDensities',
    'RosenfeldFMT3D',
    'WhiteBearIIFMT3D',
    'esFMT_Tensor3D',
    'WBII_Tensor',
    'DFTSolver3D',
    'get_mc_wall_profile'
]
