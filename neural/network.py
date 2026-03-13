"""
Conditional Neural Network
==========================

Neural network that predicts Lutsko parameters (A, B) from 
packing fraction or local density features.

Architecture
------------
- Input: Features derived from η (or local density environment)
- Hidden: MLP with GELU activation and LayerNorm
- Output: (A, B) with soft constraints via tanh

The network learns η-dependent parameters A(η), B(η) that
optimize bulk thermodynamic consistency.

Features
--------
Default features from packing fraction η:
1. η (linear)
2. η² (quadratic)
3. η³ (cubic)
4. η/(1-η) (divergent near close packing)
5. ln(1-η) (logarithmic)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple
from jaxtyping import Array


class ConditionalNetwork(eqx.Module):
    """
    Neural network for conditional (A, B) prediction.
    
    Parameters
    ----------
    key : PRNGKey
        Random key for initialization
    n_features : int
        Number of input features (default: 5)
    hidden_dim : int
        Hidden layer dimension (default: 64)
    n_hidden : int
        Number of hidden layers (default: 4)
    A_bounds : tuple
        (min, max) for A parameter
    B_bounds : tuple
        (min, max) for B parameter
    
    Example
    -------
    >>> key = jax.random.PRNGKey(42)
    >>> network = ConditionalNetwork(key)
    >>> A, B = network.from_eta(0.4)
    >>> print(f"A={A:.4f}, B={B:.4f}")
    """
    
    # Network layers
    input_proj: eqx.nn.Linear
    hidden_layers: List[eqx.nn.Linear]
    layer_norms: List[eqx.nn.LayerNorm]
    output_layer: eqx.nn.Linear
    
    # Architecture (static)
    n_features: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    n_hidden: int = eqx.field(static=True)
    
    # Output bounds (static)
    A_center: float = eqx.field(static=True)
    A_scale: float = eqx.field(static=True)
    B_center: float = eqx.field(static=True)
    B_scale: float = eqx.field(static=True)
    
    def __init__(self, key: jax.random.PRNGKey,
                 n_features: int = 5,
                 hidden_dim: int = 64,
                 n_hidden: int = 4,
                 A_bounds: Tuple[float, float] = (0.5, 2.0),
                 B_bounds: Tuple[float, float] = (-2.0, 1.0)):
        """Initialize conditional network."""
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        
        # Parameterize output as: center + scale * tanh(raw)
        self.A_center = (A_bounds[0] + A_bounds[1]) / 2
        self.A_scale = (A_bounds[1] - A_bounds[0]) / 2
        self.B_center = (B_bounds[0] + B_bounds[1]) / 2
        self.B_scale = (B_bounds[1] - B_bounds[0]) / 2
        
        # Split keys
        keys = jax.random.split(key, n_hidden + 3)
        
        # Input projection
        self.input_proj = eqx.nn.Linear(n_features, hidden_dim, key=keys[0])
        
        # Hidden layers with layer normalization
        self.hidden_layers = []
        self.layer_norms = []
        
        for i in range(n_hidden):
            self.hidden_layers.append(
                eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[i + 1])
            )
            self.layer_norms.append(eqx.nn.LayerNorm(hidden_dim))
        
        # Output layer
        self.output_layer = eqx.nn.Linear(hidden_dim, 2, key=keys[-1])
    
    def __call__(self, features: Array) -> Tuple[Array, Array]:
        """
        Forward pass: features → (A, B).
        
        Parameters
        ----------
        features : Array
            Input features, shape (..., n_features)
        
        Returns
        -------
        A, B : Array
            Lutsko parameters with soft constraints
        """
        # Input projection
        x = self.input_proj(features)
        x = jax.nn.gelu(x)
        
        # Hidden layers with residual connections and layer norm
        for i, (layer, norm) in enumerate(zip(self.hidden_layers, self.layer_norms)):
            x_res = x
            x = layer(x)
            x = norm(x)
            x = jax.nn.gelu(x)
            # Scaled residual connection
            if i > 0:
                x = x + 0.5 * x_res
        
        # Output layer
        out = self.output_layer(x)
        A_raw, B_raw = out[..., 0], out[..., 1]
        
        # Soft bounds via tanh (smooth, differentiable)
        A = self.A_center + self.A_scale * jnp.tanh(A_raw)
        B = self.B_center + self.B_scale * jnp.tanh(B_raw)
        
        return A, B
    
    def from_eta(self, eta: Array) -> Tuple[Array, Array]:
        """
        Predict (A, B) from packing fraction.
        
        Parameters
        ----------
        eta : float or Array
            Packing fraction (0 < η < 0.74)
        
        Returns
        -------
        A, B : Array
            Predicted Lutsko parameters
        """
        eta_val = jnp.atleast_1d(jnp.asarray(eta)).flatten()[0]
        eta_safe = jnp.clip(eta_val, 1e-6, 0.74 - 1e-6)
        
        # Create feature vector
        features = jnp.array([
            eta_safe,                           # Linear
            eta_safe**2,                        # Quadratic
            eta_safe**3,                        # Cubic
            eta_safe / (1 - eta_safe + 1e-8),   # Divergent
            jnp.log(1 - eta_safe + 1e-8)        # Logarithmic
        ])[:self.n_features]
        
        # Pad if needed
        if self.n_features > 5:
            features = jnp.concatenate([features, jnp.zeros(self.n_features - 5)])
        
        return self(features)
    
    def constraint_value(self, eta: Array) -> Array:
        """Compute constraint C = 8A + 2B - 9."""
        A, B = self.from_eta(eta)
        return 8*A + 2*B - 9
    
    def __repr__(self) -> str:
        return (f"ConditionalNetwork(n_features={self.n_features}, "
                f"hidden_dim={self.hidden_dim}, n_hidden={self.n_hidden})")
