"""
Correlations
============

Correlation function computation for hard-sphere fluids.

Modules
-------
direct : Direct correlation functions c_1, c_2 from functional derivatives
ornstein_zernike : OZ equation solver, pair correlation g(r), structure factor S(k)

Example
-------
>>> from correlations import compute_c2_bulk, solve_oz_fourier
>>> c2_k = compute_c2_fourier(functional, rho_bulk, grid)
>>> g_r = compute_g_radial(c2_k, rho_bulk, grid)
"""

from correlations.direct import (
    compute_c2_bulk,
    compute_c2_fourier,
    compute_c1,
)

from correlations.ornstein_zernike import (
    solve_oz_fourier,
    compute_pair_correlation,
    compute_structure_factor,
    radial_average,
    compute_g_radial,
)

__all__ = [
    'compute_c2_bulk',
    'compute_c2_fourier',
    'compute_c1',
    'solve_oz_fourier',
    'compute_pair_correlation',
    'compute_structure_factor',
    'radial_average',
    'compute_g_radial',
]
