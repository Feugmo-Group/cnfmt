"""
Nonlocal Extension
==================

Extends the Lutsko FMT functional with spatially varying parameters
A(r), B(r) predicted by a neural network from nonlocal density features.

Modules
-------
kernels : Learnable radial convolution kernels
features : Nonlocal feature extraction (η, η̄, ∇η, ∇²η)
functional : NonlocalLutskoFunctional combining all components

Example
-------
>>> from nonlocal_ext import LearnableKernel, NonlocalLutskoFunctional
>>> kernel = LearnableKernel(key, n_gaussians=8)
>>> nl_func = NonlocalLutskoFunctional(network, kernel, calculator, grid)
>>> F_exc = nl_func.excess_free_energy(rho)
>>> c1 = nl_func.compute_c1(rho)
"""

from nonlocal_ext.kernels import LearnableKernel
from nonlocal_ext.features import NonlocalFeatureExtractor
from nonlocal_ext.functional import NonlocalLutskoFunctional

__all__ = [
    'LearnableKernel',
    'NonlocalFeatureExtractor',
    'NonlocalLutskoFunctional',
]
