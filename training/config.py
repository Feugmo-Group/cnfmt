"""
Training Configuration
======================

Dataclass for all training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """
    Configuration for neural functional training.
    
    Attributes
    ----------
    mode : str
        Training mode: 'bulk', 'dft', or 'hybrid'
    n_features : int
        Number of input features for network
    hidden_dim : int
        Hidden layer dimension
    n_hidden : int
        Number of hidden layers
    optimizer : str
        Optimizer type: 'adam', 'adamw'
    learning_rate : float
        Initial learning rate
    weight_decay : float
        Weight decay for AdamW
    warmup_steps : int
        LR warmup steps
    use_cosine_decay : bool
        Use cosine annealing for LR
    min_lr_ratio : float
        Final LR = initial * min_lr_ratio
    n_iter_bulk : int
        Adam iterations for bulk training
    n_iter_lbfgs : int
        L-BFGS iterations for refinement
    n_iter_dft : int
        DFT fine-tuning iterations
    grid_size : int
        DFT grid points per dimension
    box_length : float
        DFT box size (in σ units)
    n_dft_steps : int
        Minimization steps per DFT calculation
    eta_train : List[float]
        Training packing fractions
    eta_test : List[float]
        Test packing fractions for DFT
    weight_Z, weight_mu, weight_chi : float
        Loss function weights
    weight_smooth : float
        Smoothness regularization weight
    weight_constraint : float
        Constraint penalty weight
    grad_clip : float
        Gradient clipping threshold
    use_ema : bool
        Use exponential moving average
    ema_decay : float
        EMA decay rate
    checkpoint_dir : str
        Directory for checkpoints
    save_every : int
        Checkpoint frequency
    log_every : int
        Logging frequency
    """
    
    # Training mode
    mode: str = "hybrid"
    
    # Network architecture
    n_features: int = 5
    hidden_dim: int = 64
    n_hidden: int = 4
    
    # Optimizer settings
    optimizer: str = "adamw"
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    
    # Learning rate schedule
    warmup_steps: int = 50
    use_cosine_decay: bool = True
    min_lr_ratio: float = 0.01
    
    # Training iterations
    n_iter_bulk: int = 500
    n_iter_lbfgs: int = 100
    n_iter_dft: int = 100
    
    # DFT settings
    grid_size: int = 32
    box_length: float = 12.0
    n_dft_steps: int = 300
    
    # Training packing fractions
    eta_train: List[float] = field(default_factory=lambda: [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.48
    ])
    eta_test: List[float] = field(default_factory=lambda: [0.2, 0.35, 0.45])
    
    # Loss function weights
    weight_Z: float = 1.0
    weight_mu: float = 1.0
    weight_chi: float = 0.5
    weight_smooth: float = 0.05
    weight_constraint: float = 0.1
    
    # Gradient settings
    grad_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.99
    
    # DFT gradient settings
    dft_grad_eps: float = 0.01
    dft_n_grad_coords: int = 50
    
    # Nonlocal constraint loss weights
    weight_contact: float = 1.0
    weight_noether: float = 0.1
    weight_spt: float = 0.5
    weight_positivity: float = 1.0
    weight_oz_consistency: float = 0.5  # for future use

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 100
    log_every: int = 10
