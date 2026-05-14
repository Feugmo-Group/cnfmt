"""
Constraints
===========

Physics constraint losses for training neural density functionals
without simulation data.

Modules
-------
sum_rules : Contact and compressibility sum rule losses
consistency : Pressure and Ornstein-Zernike consistency losses
noether : Translational invariance from Noether's theorem
scaled_particle : Scaled-particle theory exact relations and positivity

Example
-------
>>> from constraints import contact_sum_rule_loss, pressure_consistency_loss
>>> loss_sr = contact_sum_rule_loss(functional, rho, grid, eta)
>>> loss_pc = pressure_consistency_loss(functional, rho, grid)
"""

from constraints.sum_rules import (
    contact_sum_rule_loss,
    compressibility_sum_rule_loss,
    gibbs_adsorption_loss,
)

from constraints.consistency import (
    pressure_consistency_loss,
    oz_consistency_loss,
    c2_reference_loss,
    structure_factor_loss,
)

from constraints.noether import (
    translational_invariance_loss,
)

from constraints.scaled_particle import (
    low_density_limit_loss,
    close_packing_limit_loss,
    spt_exact_relations_loss,
    positivity_loss,
)

__all__ = [
    'contact_sum_rule_loss',
    'compressibility_sum_rule_loss',
    'gibbs_adsorption_loss',
    'pressure_consistency_loss',
    'oz_consistency_loss',
    'c2_reference_loss',
    'structure_factor_loss',
    'translational_invariance_loss',
    'low_density_limit_loss',
    'close_packing_limit_loss',
    'spt_exact_relations_loss',
    'positivity_loss',
]
