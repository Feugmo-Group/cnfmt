"""
Conditional Neural Fundamental Measure Theory (CNFMT)
=====================================================

A JAX-based implementation of classical Density Functional Theory (cDFT)
with neural network-parameterized Lutsko functional.

Modules
-------
core : Grid, weighted densities, bulk thermodynamics
functionals : FMT and Lutsko functionals
neural : Conditional neural networks for parameter prediction
solvers : Density minimizers and test particle calculations
training : Loss functions, optimizers, checkpointing
utils : Plotting and analysis utilities

Example
-------
>>> from cnfmt import Grid, LutskoFunctional, ConditionalNetwork
>>> from cnfmt.training import train_bulk_phase
>>> 
>>> grid = Grid((32, 32, 32), 12.0)
>>> network = ConditionalNetwork(key, n_features=5)
>>> network, losses = train_bulk_phase(network, config)
"""

__version__ = "1.0.0"
__author__ = "Cony"

# Core components
from cnfmt.core.grid import Grid
from cnfmt.core.weights import FMTKernels
from cnfmt.core.densities import WeightedDensities, WeightedDensityCalculator
from cnfmt.core.thermodynamics import BulkThermodynamics

# Functionals
from cnfmt.functionals.lutsko import LutskoFunctional
from cnfmt.functionals.potentials import GrandPotential, TestParticlePotential

# Neural networks
from cnfmt.neural.network import ConditionalNetwork

# Solvers
from cnfmt.solvers.minimizer import DensityMinimizer
from cnfmt.solvers.test_particle import TestParticleCalculator

# Training
from cnfmt.training.config import TrainingConfig

__all__ = [
    # Core
    'Grid', 'FMTKernels', 'WeightedDensities', 'WeightedDensityCalculator',
    'BulkThermodynamics',
    # Functionals
    'LutskoFunctional', 'GrandPotential', 'TestParticlePotential',
    # Neural
    'ConditionalNetwork',
    # Solvers
    'DensityMinimizer', 'TestParticleCalculator',
    # Training
    'TrainingConfig',
]
