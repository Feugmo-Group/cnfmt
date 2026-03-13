"""
Checkpoint Management
=====================

Save and load model checkpoints with Equinox serialization.
"""

import jax
import equinox as eqx
import json
from pathlib import Path
from cnfmt.neural.network import ConditionalNetwork
from cnfmt.training.config import TrainingConfig


def save_checkpoint(network: ConditionalNetwork, config: TrainingConfig, 
                    name: str) -> Path:
    """
    Save network checkpoint.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Trained network
    config : TrainingConfig
        Training configuration
    name : str
        Checkpoint name (e.g., 'after_bulk', 'final')
    
    Returns
    -------
    filepath : Path
        Path to saved checkpoint
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save model
    filepath = checkpoint_dir / f"model_{name}.eqx"
    eqx.tree_serialise_leaves(filepath, network)
    
    # Save config
    config_path = checkpoint_dir / f"config_{name}.json"
    config_dict = {
        'n_features': config.n_features,
        'hidden_dim': config.hidden_dim,
        'n_hidden': config.n_hidden,
        'optimizer': config.optimizer,
        'learning_rate': config.learning_rate,
        'n_iter_bulk': config.n_iter_bulk,
        'n_iter_lbfgs': config.n_iter_lbfgs,
        'n_iter_dft': config.n_iter_dft,
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"  Saved checkpoint: {filepath}")
    return filepath


def load_checkpoint(filepath: str, config: TrainingConfig) -> ConditionalNetwork:
    """
    Load network from checkpoint.
    
    Parameters
    ----------
    filepath : str
        Path to checkpoint file
    config : TrainingConfig
        Configuration (for architecture)
    
    Returns
    -------
    network : ConditionalNetwork
        Loaded network
    """
    key = jax.random.PRNGKey(0)
    network = ConditionalNetwork(
        key, config.n_features, config.hidden_dim, config.n_hidden
    )
    network = eqx.tree_deserialise_leaves(filepath, network)
    print(f"Loaded checkpoint: {filepath}")
    return network
