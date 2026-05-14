"""
Nonlocal Lutsko Functional
===========================

Extends the Lutsko esFMT functional with spatially varying parameters
A(r), B(r) predicted by a neural network from nonlocal density features.

    F_exc[ρ] = ∫ Φ(n_α(r); A(r), B(r)) dr

where A(r), B(r) = NN(η(r), η̄(r), |∇η|, ∇²η, η-η̄)

The key advantage: c₁(r) = -δF_exc/δρ(r) is computed via JAX autodiff,
which automatically accounts for the dependence of A, B on ρ through
the neural network and the learnable kernel.

Architecture
------------
    ρ(r) → WeightedDensities → η(r) = n₃(r)
                                  ↓
                            LearnableKernel → η̄(r)
                                  ↓
                         NonlocalFeatures(η, η̄, ∇η, ∇²η, η-η̄)
                                  ↓
                      NonlocalConditionalNetwork → A(r), B(r)
                                  ↓
                      LutskoFunctional.free_energy_density(measures, A, B)
                                  ↓
                            F_exc = ∫Φ dr
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from jaxtyping import Array
from core.grid import Grid
from core.densities import WeightedDensityCalculator, WeightedDensities
from functionals.lutsko import LutskoFunctional
from nonlocal_ext.kernels import LearnableKernel
from nonlocal_ext.features import NonlocalFeatureExtractor
from neural.network import NonlocalConditionalNetwork


class NonlocalLutskoFunctional(eqx.Module):
    """
    Lutsko functional with spatially varying A(r), B(r) from neural network.

    Combines:
    - LearnableKernel for nonlocal density smoothing
    - NonlocalConditionalNetwork for parameter prediction
    - LutskoFunctional for free energy computation

    The c₁ = -δF_exc/δρ is computed via JAX autodiff through the
    entire pipeline, including the neural network and kernel convolution.

    Parameters
    ----------
    network : NonlocalConditionalNetwork
        Neural network mapping features → (A, B)
    kernel : LearnableKernel
        Learnable convolution kernel for nonlocal features
    calculator : WeightedDensityCalculator
        FMT weighted density calculator
    grid : Grid
        Computational grid
    base_functional : LutskoFunctional
        Base Lutsko functional (used for Φ computation)

    Example
    -------
    >>> key = jax.random.PRNGKey(42)
    >>> k1, k2 = jax.random.split(key)
    >>> network = NonlocalConditionalNetwork(k1)
    >>> kernel = LearnableKernel(k2)
    >>> nl_func = NonlocalLutskoFunctional(network, kernel, calculator, grid)
    >>> F_exc = nl_func.excess_free_energy(rho)
    >>> c1 = nl_func.compute_c1(rho)
    """

    network: NonlocalConditionalNetwork
    kernel: LearnableKernel
    feature_extractor: NonlocalFeatureExtractor
    calculator: WeightedDensityCalculator
    grid: Grid
    base_functional: LutskoFunctional

    def __init__(self, network: NonlocalConditionalNetwork,
                 kernel: LearnableKernel,
                 calculator: WeightedDensityCalculator,
                 grid: Grid,
                 base_functional: Optional[LutskoFunctional] = None):
        """Initialize nonlocal Lutsko functional."""
        self.network = network
        self.kernel = kernel
        self.calculator = calculator
        self.grid = grid
        self.base_functional = base_functional or LutskoFunctional(A=1.0, B=0.0)

        # Build feature extractor from components
        self.feature_extractor = NonlocalFeatureExtractor(
            kernel, calculator, grid
        )

    def predict_parameters(self, rho: Array) -> tuple:
        """
        Predict spatially varying A(r), B(r) from density field.

        Parameters
        ----------
        rho : Array
            Density field, shape (nx, ny, nz)

        Returns
        -------
        A : Array
            Spatially varying A parameter, shape (nx, ny, nz)
        B : Array
            Spatially varying B parameter, shape (nx, ny, nz)
        """
        # Extract nonlocal features: (nx, ny, nz, 5)
        features = self.feature_extractor(rho)

        # Flatten spatial dimensions for pointwise MLP
        shape = features.shape[:-1]  # (nx, ny, nz)
        flat_features = features.reshape(-1, features.shape[-1])  # (N, 5)

        # Pointwise prediction
        A_flat, B_flat = self.network(flat_features)

        # Reshape back to spatial grid
        A = A_flat.reshape(shape)
        B = B_flat.reshape(shape)

        return A, B

    def free_energy_density(self, rho: Array) -> Array:
        """
        Compute excess free energy density Φ(r) with neural A(r), B(r).

        Parameters
        ----------
        rho : Array
            Density field, shape (nx, ny, nz)

        Returns
        -------
        Phi : Array
            Free energy density, shape (nx, ny, nz)
        """
        # Predict spatially varying parameters
        A, B = self.predict_parameters(rho)

        # Compute weighted densities
        measures = self.calculator(rho)

        # Evaluate Lutsko functional with field-valued A, B
        Phi = self.base_functional.free_energy_density(measures, A, B)

        return Phi

    def excess_free_energy(self, rho: Array) -> float:
        """
        Compute total excess free energy F_exc = ∫ Φ(r) dr.

        Parameters
        ----------
        rho : Array
            Density field, shape (nx, ny, nz)

        Returns
        -------
        F_exc : float
            Total excess free energy (βF_exc)
        """
        Phi = self.free_energy_density(rho)
        return jnp.sum(Phi) * self.grid.dV

    def compute_c1(self, rho: Array) -> Array:
        """
        Compute one-body direct correlation function via JAX autodiff.

        c₁(r) = -δF_exc/δρ(r)

        This automatically differentiates through:
        - FFT convolutions (weighted densities)
        - Learnable kernel convolution (nonlocal features)
        - Neural network (A, B prediction)
        - Lutsko free energy density

        Parameters
        ----------
        rho : Array
            Density field, shape (nx, ny, nz)

        Returns
        -------
        c1 : Array
            One-body DCF, shape (nx, ny, nz)
        """
        c1 = -jax.grad(self.excess_free_energy)(rho)
        return c1

    def compute_c1_bulk(self, eta: float) -> float:
        """
        Compute bulk c₁ at uniform density.

        For uniform ρ, c₁ is constant. This returns that constant value.

        Parameters
        ----------
        eta : float
            Bulk packing fraction

        Returns
        -------
        c1_bulk : float
            Bulk one-body DCF
        """
        rho_bulk = 6.0 * eta / jnp.pi
        rho_uniform = jnp.ones((self.grid.nx, self.grid.ny, self.grid.nz)) * rho_bulk
        c1_field = self.compute_c1(rho_uniform)

        # Average over all points (should be uniform)
        return jnp.mean(c1_field)

    def constraint_field(self, rho: Array) -> Array:
        """
        Compute spatially varying constraint C(r) = 8A(r) + 2B(r) - 9.

        Parameters
        ----------
        rho : Array
            Density field

        Returns
        -------
        C : Array
            Constraint field, shape (nx, ny, nz)
        """
        A, B = self.predict_parameters(rho)
        return 8 * A + 2 * B - 9

    def bulk_parameters(self, eta: float) -> tuple:
        """
        Get (A, B) predicted for uniform density at given η.

        Parameters
        ----------
        eta : float
            Packing fraction

        Returns
        -------
        A, B : float
            Predicted parameters in bulk
        """
        rho_bulk = 6.0 * eta / jnp.pi
        rho_uniform = jnp.ones((self.grid.nx, self.grid.ny, self.grid.nz)) * rho_bulk
        A, B = self.predict_parameters(rho_uniform)

        # In uniform system, A and B should be constant
        return jnp.mean(A), jnp.mean(B)

    def __repr__(self) -> str:
        return (f"NonlocalLutskoFunctional(\n"
                f"  network={self.network},\n"
                f"  kernel={self.kernel},\n"
                f"  grid={self.grid}\n"
                f")")
