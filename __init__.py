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

# Enable double precision globally — required for all FMT physics
import jax
jax.config.update("jax_enable_x64", True)

# Core components
from core.grid import Grid
from core.weights import FMTKernels
from core.densities import WeightedDensities, WeightedDensityCalculator
from core.thermodynamics import BulkThermodynamics

# Functionals
from functionals.lutsko import LutskoFunctional
from functionals.potentials import GrandPotential, TestParticlePotential

# Neural networks
from neural.network import ConditionalNetwork

# Solvers
from solvers.minimizer import DensityMinimizer
from solvers.test_particle import TestParticleCalculator

# Training
from training.config import TrainingConfig

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
