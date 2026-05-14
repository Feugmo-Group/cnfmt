"""
Validation
==========

Analytical reference solutions for validating neural density functionals.

Modules
-------
percus_yevick : Exact Percus-Yevick solutions for hard-sphere direct
    correlation function, structure factor, and contact value

Example
-------
>>> from validation import py_direct_correlation, py_structure_factor
>>> c_r = py_direct_correlation(r, eta)
>>> S_k = py_structure_factor(k, eta)
"""

from validation.percus_yevick import (
    py_direct_correlation,
    py_direct_correlation_fourier,
    py_structure_factor,
    py_compressibility,
    py_contact_value,
    compute_py_g_numerical,
)

__all__ = [
    'py_direct_correlation',
    'py_direct_correlation_fourier',
    'py_structure_factor',
    'py_compressibility',
    'py_contact_value',
    'compute_py_g_numerical',
]
