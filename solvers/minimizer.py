"""
Density Minimization
====================

Direct minimization of grand potential Ω[ρ] using gradient-based
optimizers (Adam, L-BFGS).

The equilibrium density satisfies:
    δΩ/δρ = 0

We use log-density parametrization for positivity:
    ρ(r) = exp(ψ(r))

This ensures ρ > 0 without constrained optimization.
"""

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import equinox as eqx
from typing import Tuple, List, Optional
from jaxtyping import Array
from functionals.potentials import GrandPotential


class DensityMinimizer(eqx.Module):
    """
    Minimizes grand potential Ω[ρ] to find equilibrium density.
    
    Parameters
    ----------
    grand_potential : GrandPotential
        Grand potential functional
    
    Example
    -------
    >>> minimizer = DensityMinimizer(grand_potential)
    >>> rho_eq, losses = minimizer.minimize_adam(rho_init, v_ext, mu)
    """
    
    grand_potential: GrandPotential
    
    def __init__(self, grand_potential: GrandPotential):
        self.grand_potential = grand_potential
    
    def minimize_adam(self, rho_init: Array, v_ext: Array, mu: float,
                      n_steps: int = 500, lr: float = 1e-3,
                      N_target: Optional[float] = None,
                      A: Optional[float] = None, B: Optional[float] = None,
                      verbose: bool = False) -> Tuple[Array, List[float]]:
        """
        Minimize using Adam optimizer.
        
        Uses log-density parametrization: ρ = exp(ψ)
        
        Parameters
        ----------
        rho_init : Array
            Initial density guess
        v_ext : Array
            External potential
        mu : float
            Chemical potential (βμ)
        n_steps : int
            Number of optimization steps
        lr : float
            Learning rate
        N_target : float, optional
            Target particle number (for constraint)
        A, B : float, optional
            Lutsko parameters
        verbose : bool
            Print progress
        
        Returns
        -------
        rho_eq : Array
            Equilibrium density
        losses : List[float]
            Loss history
        """
        eps = 1e-10
        log_rho = jnp.log(jnp.maximum(rho_init, eps))
        
        # Loss function
        def loss_fn(log_rho):
            rho = jnp.exp(log_rho)
            rho = jnp.clip(rho, eps, 100.0)
            
            if N_target is not None:
                return self.grand_potential.with_particle_constraint(
                    rho, v_ext, mu, N_target, lambda_N=10.0, A=A, B=B
                )
            else:
                return self.grand_potential(rho, v_ext, mu, A, B)
        
        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(log_rho)
        
        losses = []
        
        @jit
        def step(log_rho, opt_state):
            loss, grads = value_and_grad(loss_fn)(log_rho)
            grads = jnp.clip(grads, -10.0, 10.0)  # Gradient clipping
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_log_rho = optax.apply_updates(log_rho, updates)
            return new_log_rho, new_opt_state, loss
        
        for i in range(n_steps):
            log_rho, opt_state, loss = step(log_rho, opt_state)
            losses.append(float(loss))
            
            if verbose and i % 50 == 0:
                rho = jnp.exp(log_rho)
                N = jnp.sum(rho) * self.grand_potential.grid.dV
                print(f"  Adam iter {i:4d}: Ω = {float(loss):.6e}, N = {float(N):.2f}")
        
        rho_final = jnp.exp(log_rho)
        rho_final = jnp.clip(rho_final, eps, 100.0)
        
        return rho_final, losses
    
    def minimize_lbfgs(self, rho_init: Array, v_ext: Array, mu: float,
                       max_iter: int = 100,
                       N_target: Optional[float] = None,
                       A: Optional[float] = None, B: Optional[float] = None,
                       verbose: bool = False) -> Tuple[Array, float]:
        """
        Minimize using L-BFGS optimizer.
        
        Better for fine-tuning near a minimum.
        """
        from jax.scipy.optimize import minimize as jax_minimize
        
        eps = 1e-10
        
        def loss_fn(log_rho_flat):
            log_rho = log_rho_flat.reshape(rho_init.shape)
            rho = jnp.exp(log_rho)
            rho = jnp.clip(rho, eps, 100.0)
            
            if N_target is not None:
                return self.grand_potential.with_particle_constraint(
                    rho, v_ext, mu, N_target, lambda_N=10.0, A=A, B=B
                )
            else:
                return self.grand_potential(rho, v_ext, mu, A, B)
        
        log_rho_init = jnp.log(jnp.maximum(rho_init, eps)).ravel()
        
        result = jax_minimize(
            loss_fn,
            log_rho_init,
            method='BFGS',
            options={'maxiter': max_iter}
        )
        
        log_rho_final = result.x.reshape(rho_init.shape)
        rho_final = jnp.exp(log_rho_final)
        rho_final = jnp.clip(rho_final, eps, 100.0)
        
        if verbose:
            print(f"  L-BFGS: converged={result.success}, Ω={float(result.fun):.6e}")
        
        return rho_final, float(result.fun)
