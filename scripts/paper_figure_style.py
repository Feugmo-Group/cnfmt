"""
Shared matplotlib style for paper figures.

Sets font sizes appropriate for PRE single/double-column figures
(minimum 8pt at print size).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def apply_paper_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
