"""
Utilities Module
================

Plotting and analysis utilities.

- plotting: Publication-quality figures
- analysis: Parameter analysis and comparison
"""

from cnfmt.utils.plotting import (
    plot_training_losses,
    plot_learned_parameters,
    plot_thermodynamics,
    plot_parameter_space,
    create_publication_figure
)
from cnfmt.utils.analysis import (
    evaluate_network,
    compare_functionals,
    compute_errors
)

__all__ = [
    'plot_training_losses', 'plot_learned_parameters', 
    'plot_thermodynamics', 'plot_parameter_space',
    'create_publication_figure',
    'evaluate_network', 'compare_functionals', 'compute_errors'
]
