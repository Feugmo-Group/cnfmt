"""
Training Module
===============

Training infrastructure for conditional neural functional.

- TrainingConfig: Configuration dataclass
- Loss functions: Bulk thermodynamic and DFT losses
- Optimizers: Adam with scheduling, L-BFGS refinement
- Checkpointing: Save/load model states
"""

from .config import TrainingConfig
from .losses import compute_bulk_loss, compute_dft_loss
from .optimizers import (
    train_bulk_adam, train_bulk_lbfgs, train_dft_phase
)
from .checkpoints import save_checkpoint, load_checkpoint

__all__ = [
    'TrainingConfig',
    'compute_bulk_loss', 'compute_dft_loss',
    'train_bulk_adam', 'train_bulk_lbfgs', 'train_dft_phase',
    'save_checkpoint', 'load_checkpoint'
]
