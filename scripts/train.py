#!/usr/bin/env python
"""
CNFMT Training Script
=====================

Train a conditional neural functional for Lutsko esFMT.

Usage
-----
    # Full training (recommended)
    python -m cnfmt.scripts.train --mode hybrid
    
    # Just bulk training
    python -m cnfmt.scripts.train --mode bulk --n_iter_bulk 1000
    
    # From checkpoint
    python -m cnfmt.scripts.train --mode dft --checkpoint checkpoints/model_after_bulk.eqx
"""

import jax
import jax.numpy as jnp
import argparse
from pathlib import Path

from neural.network import ConditionalNetwork
from training.config import TrainingConfig
from training.optimizers import train_bulk_adam, train_bulk_lbfgs, train_dft_phase
from training.checkpoints import save_checkpoint, load_checkpoint
from utils.plotting import create_publication_figure
from utils.analysis import evaluate_network


def main():
    parser = argparse.ArgumentParser(
        description='Train Conditional Neural Functional',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='hybrid',
                       choices=['bulk', 'dft', 'hybrid'],
                       help='Training mode')
    
    # Network architecture
    parser.add_argument('--n_features', type=int, default=5,
                       help='Number of input features')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer dimension')
    parser.add_argument('--n_hidden', type=int, default=4,
                       help='Number of hidden layers')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--lr', type=float, default=3e-3,
                       help='Learning rate')
    
    # Iterations
    parser.add_argument('--n_iter_bulk', type=int, default=500,
                       help='Bulk training iterations')
    parser.add_argument('--n_iter_lbfgs', type=int, default=100,
                       help='L-BFGS refinement iterations')
    parser.add_argument('--n_iter_dft', type=int, default=50,
                       help='DFT fine-tuning iterations')
    
    # DFT settings
    parser.add_argument('--grid_size', type=int, default=32,
                       help='DFT grid points per dimension')
    parser.add_argument('--n_dft_steps', type=int, default=200,
                       help='DFT minimization steps')
    
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Load from checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for figures')
    
    # Misc
    parser.add_argument('--no_ema', action='store_true',
                       help='Disable EMA')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        mode=args.mode,
        n_features=args.n_features,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        n_iter_bulk=args.n_iter_bulk,
        n_iter_lbfgs=args.n_iter_lbfgs,
        n_iter_dft=args.n_iter_dft,
        grid_size=args.grid_size,
        n_dft_steps=args.n_dft_steps,
        use_ema=not args.no_ema,
    )
    
    # Print header
    print("\n" + "="*70)
    print("CONDITIONAL NEURAL FUNDAMENTAL MEASURE THEORY")
    print("="*70)
    print(f"Mode: {config.mode}")
    print(f"Network: {config.n_features} → {config.hidden_dim} × {config.n_hidden}")
    print(f"Optimizer: {config.optimizer}, lr={config.learning_rate}")
    print(f"Iterations: Bulk={config.n_iter_bulk}, L-BFGS={config.n_iter_lbfgs}, DFT={config.n_iter_dft}")
    print("="*70)
    
    # Initialize or load network
    if args.checkpoint:
        network = load_checkpoint(args.checkpoint, config)
    else:
        key = jax.random.PRNGKey(args.seed)
        network = ConditionalNetwork(
            key, config.n_features, config.hidden_dim, config.n_hidden
        )
    
    bulk_losses = []
    dft_results = []
    eta_values = jnp.array(config.eta_train)
    
    # Phase 1: Bulk training
    if config.mode in ['bulk', 'hybrid']:
        print("\n" + "="*70)
        print("PHASE 1: BULK THERMODYNAMIC TRAINING")
        print("="*70)
        
        # Phase 1A: Adam
        network, adam_losses = train_bulk_adam(network, config, eta_values)
        save_checkpoint(network, config, "after_adam")
        
        # Phase 1B: L-BFGS
        if config.n_iter_lbfgs > 0:
            network, lbfgs_losses = train_bulk_lbfgs(network, config, eta_values)
            bulk_losses = adam_losses + lbfgs_losses
        else:
            bulk_losses = adam_losses
        
        save_checkpoint(network, config, "after_bulk")
    
    # Phase 2: DFT fine-tuning
    if config.mode in ['dft', 'hybrid']:
        if config.n_iter_dft > 0:
            network, dft_results = train_dft_phase(network, config)
            save_checkpoint(network, config, "after_dft")
    
    # Final save
    save_checkpoint(network, config, "final")
    
    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    evaluate_network(network)
    
    # Create figures
    output_dir = Path(args.output_dir)
    create_publication_figure(network, bulk_losses, dft_results, output_dir)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: {config.checkpoint_dir}/model_final.eqx")
    print(f"Figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
