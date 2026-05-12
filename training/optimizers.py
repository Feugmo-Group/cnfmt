"""
Training Optimizers
===================

Multi-phase optimization for conditional neural functional:

Phase 1A: Adam with warmup + cosine decay
Phase 1B: L-BFGS refinement
Phase 2: DFT fine-tuning with coordinate-wise gradients
"""

import jax
import jax.numpy as jnp
from jax import value_and_grad
from jax.flatten_util import ravel_pytree
import optax
import equinox as eqx
from typing import List, Tuple, Any, NamedTuple

from core.grid import Grid
from neural.network import ConditionalNetwork
from solvers.test_particle import TestParticleCalculator
from .config import TrainingConfig
from .losses import compute_bulk_loss, compute_dft_loss


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================

def create_lr_schedule(config: TrainingConfig):
    """Create learning rate schedule with warmup and cosine decay."""
    if config.use_cosine_decay:
        warmup_fn = optax.linear_schedule(
            init_value=config.learning_rate * 0.1,
            end_value=config.learning_rate,
            transition_steps=config.warmup_steps
        )
        decay_fn = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.n_iter_bulk - config.warmup_steps,
            alpha=config.min_lr_ratio
        )
        return optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[config.warmup_steps]
        )
    return optax.constant_schedule(config.learning_rate)


def create_optimizer(config: TrainingConfig):
    """Create optimizer with gradient clipping."""
    schedule = create_lr_schedule(config)
    
    if config.optimizer == "adamw":
        base_opt = optax.adamw(schedule, weight_decay=config.weight_decay)
    else:
        base_opt = optax.adam(schedule)
    
    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        base_opt
    )


# ============================================================================
# EMA (EXPONENTIAL MOVING AVERAGE)
# ============================================================================

class EMAState(NamedTuple):
    """State for exponential moving average."""
    ema_params: Any
    step: int


def init_ema(params, decay=0.99):
    return EMAState(ema_params=params, step=0)


def update_ema(ema_state, params, decay=0.99):
    new_ema = jax.tree_util.tree_map(
        lambda e, p: decay * e + (1 - decay) * p,
        ema_state.ema_params, params
    )
    return EMAState(ema_params=new_ema, step=ema_state.step + 1)


# ============================================================================
# PHASE 1A: ADAM TRAINING
# ============================================================================

def train_bulk_adam(network: ConditionalNetwork, config: TrainingConfig,
                    eta_values: jnp.ndarray, verbose: bool = True
                    ) -> Tuple[ConditionalNetwork, List[float]]:
    """
    Phase 1A: Train using Adam with LR scheduling.
    
    Parameters
    ----------
    network : ConditionalNetwork
        Neural network to train
    config : TrainingConfig
        Training configuration
    eta_values : array
        Training packing fractions
    verbose : bool
        Print progress
    
    Returns
    -------
    network : ConditionalNetwork
        Trained network
    losses : List[float]
        Loss history
    """
    if verbose:
        print("\n" + "-"*60)
        print("Phase 1A: Adam Training with LR Schedule")
        print("-"*60)
    
    optimizer = create_optimizer(config)
    params = eqx.filter(network, eqx.is_array)
    opt_state = optimizer.init(params)
    
    ema_state = init_ema(params, config.ema_decay) if config.use_ema else None
    
    @eqx.filter_value_and_grad
    def loss_fn(net):
        return compute_bulk_loss(net, eta_values, config)
    
    losses = []
    best_network = network
    best_loss = float('inf')
    
    if verbose:
        print(f"\n{'Iter':>6} {'Loss':>12} {'A(0.3)':>8} {'B(0.3)':>8} {'C(0.3)':>10}")
        print("-"*55)
    
    for i in range(config.n_iter_bulk):
        loss, grads = loss_fn(network)
        
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(network, eqx.is_array)
        )
        network = eqx.apply_updates(network, updates)
        losses.append(float(loss))
        
        if config.use_ema:
            ema_state = update_ema(ema_state, eqx.filter(network, eqx.is_array),
                                   config.ema_decay)
        
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_network = network
        
        if verbose and (i % config.log_every == 0 or i == config.n_iter_bulk - 1):
            A, B = network.from_eta(0.3)
            C = 8 * float(A) + 2 * float(B) - 9
            print(f"{i:6d} {float(loss):12.4e} {float(A):8.4f} {float(B):8.4f} {C:10.4f}")
    
    # Use EMA weights if enabled
    if config.use_ema and ema_state is not None:
        final_network = eqx.combine(
            ema_state.ema_params,
            eqx.filter(network, lambda x: not eqx.is_array(x))
        )
    else:
        final_network = best_network
    
    return final_network, losses


# ============================================================================
# PHASE 1B: L-BFGS REFINEMENT
# ============================================================================

def train_bulk_lbfgs(network: ConditionalNetwork, config: TrainingConfig,
                     eta_values: jnp.ndarray, verbose: bool = True
                     ) -> Tuple[ConditionalNetwork, List[float]]:
    """
    Phase 1B: Refine using L-BFGS optimizer.
    """
    if verbose:
        print("\n" + "-"*60)
        print("Phase 1B: L-BFGS Refinement")
        print("-"*60)
    
    params = eqx.filter(network, eqx.is_array)
    flat_params, unflatten = ravel_pytree(params)
    
    def loss_from_flat(flat_p):
        p = unflatten(flat_p)
        net = eqx.combine(p, eqx.filter(network, lambda x: not eqx.is_array(x)))
        return compute_bulk_loss(net, eta_values, config)
    
    opt = optax.lbfgs(learning_rate=1.0, memory_size=20, scale_init_precond=True)
    opt_state = opt.init(flat_params)
    
    losses = []
    best_params = flat_params
    best_loss = float('inf')
    
    if verbose:
        print(f"\n{'Iter':>6} {'Loss':>12} {'|grad|':>10} {'A(0.3)':>8} {'B(0.3)':>8}")
        print("-"*55)
    
    for i in range(config.n_iter_lbfgs):
        loss, grads = value_and_grad(loss_from_flat)(flat_params)
        grad_norm = float(jnp.linalg.norm(grads))
        
        updates, opt_state = opt.update(
            grads, opt_state, flat_params,
            value=loss, grad=grads, value_fn=loss_from_flat
        )
        flat_params = optax.apply_updates(flat_params, updates)
        losses.append(float(loss))
        
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_params = flat_params
        
        if verbose and (i % config.log_every == 0 or i == config.n_iter_lbfgs - 1):
            params_i = unflatten(flat_params)
            net_i = eqx.combine(params_i, eqx.filter(network, lambda x: not eqx.is_array(x)))
            A, B = net_i.from_eta(0.3)
            print(f"{i:6d} {float(loss):12.4e} {grad_norm:10.2e} "
                  f"{float(A):8.4f} {float(B):8.4f}")
        
        if grad_norm < 1e-6:
            if verbose:
                print(f"Converged at iteration {i}")
            break
    
    best_p = unflatten(best_params)
    best_network = eqx.combine(best_p, eqx.filter(network, lambda x: not eqx.is_array(x)))
    
    return best_network, losses


# ============================================================================
# PHASE 2: DFT FINE-TUNING
# ============================================================================

def train_dft_phase(network: ConditionalNetwork, config: TrainingConfig,
                    verbose: bool = True) -> Tuple[ConditionalNetwork, List[dict]]:
    """
    Phase 2: Fine-tune with DFT calculations.
    
    Uses coordinate-wise gradient estimation for accurate gradients.
    """
    if verbose:
        print("\n" + "="*60)
        print("PHASE 2: DFT FINE-TUNING")
        print("="*60)
    
    grid = Grid((config.grid_size,)*3, config.box_length)
    base_calc = TestParticleCalculator(grid, sigma=1.0, A=1.0, B=0.0)
    
    opt = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(config.learning_rate * 0.01)
    )
    
    params = eqx.filter(network, eqx.is_array)
    flat_params, unflatten = ravel_pytree(params)
    opt_state = opt.init(flat_params)
    
    results_history = []
    best_network = network
    best_loss = float('inf')
    eta_values = config.eta_test
    
    if verbose:
        print(f"\n{'Iter':>5} {'Loss':>12} {'A(0.35)':>8} {'B(0.35)':>8}")
        print("-"*45)
    
    for i in range(config.n_iter_dft):
        # Compute aggregated loss
        total_loss = 0.0
        for eta in eta_values:
            A, B = network.from_eta(eta)
            calc = base_calc.with_parameters(float(A), float(B))
            result = calc.compute(eta, n_steps=config.n_dft_steps, verbose=False)
            total_loss += compute_dft_loss(result['delta_mu'], result['delta_chi'])
        total_loss /= len(eta_values)
        
        results_history.append({'iter': i, 'loss': total_loss})
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_network = network
        
        # Coordinate-wise gradient estimation
        eps = config.dft_grad_eps
        n_coords = min(config.dft_n_grad_coords, len(flat_params))
        coords = jax.random.permutation(jax.random.PRNGKey(i), len(flat_params))[:n_coords]
        grad_estimate = jnp.zeros_like(flat_params)
        
        def eval_loss(flat_p):
            p = unflatten(flat_p)
            net = eqx.combine(p, eqx.filter(network, lambda x: not eqx.is_array(x)))
            loss = 0.0
            for eta in eta_values:
                A, B = net.from_eta(eta)
                calc = base_calc.with_parameters(float(A), float(B))
                r = calc.compute(eta, n_steps=config.n_dft_steps // 2, verbose=False)
                loss += compute_dft_loss(r['delta_mu'], r['delta_chi'])
            return loss / len(eta_values)
        
        for coord in coords:
            e_i = jnp.zeros_like(flat_params).at[int(coord)].set(1.0)
            loss_plus = eval_loss(flat_params + eps * e_i)
            loss_minus = eval_loss(flat_params - eps * e_i)
            grad_estimate = grad_estimate.at[int(coord)].set((loss_plus - loss_minus) / (2 * eps))
        
        # Update
        updates, opt_state = opt.update(grad_estimate, opt_state, flat_params)
        flat_params = optax.apply_updates(flat_params, updates)
        
        new_params = unflatten(flat_params)
        network = eqx.combine(new_params, eqx.filter(network, lambda x: not eqx.is_array(x)))
        
        if verbose:
            A, B = network.from_eta(0.35)
            print(f"{i:5d} {total_loss:12.4e} {float(A):8.4f} {float(B):8.4f}")
    
    return best_network, results_history
