"""
Nonlocal Feature Extraction
============================

Computes nonlocal structural features from density fields for
the NonlocalConditionalNetwork.

Features
--------
From the local packing fraction η(r) = n₃(r):

1. η(r)           — local packing fraction (normalized by 0.5)
2. η̄(r)           — nonlocal smoothed packing fraction (via learnable kernel)
3. |∇η(r)|        — gradient magnitude (interface indicator)
4. ∇²η(r)         — Laplacian (curvature / oscillation indicator)
5. η(r) - η̄(r)    — local deviation from nonlocal average

All features are computed via spectral (FFT) methods for efficiency.
The nonlocal feature η̄ uses the LearnableKernel for convolution.
"""

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from core.grid import Grid
from core.densities import WeightedDensityCalculator
from nonlocal_ext.kernels import LearnableKernel


class NonlocalFeatureExtractor(eqx.Module):
    """
    Extracts nonlocal structural features from density field.

    Combines local packing fraction, nonlocal smoothing via a learnable
    kernel, and spectral gradient/Laplacian features.

    Parameters
    ----------
    kernel : LearnableKernel
        Learnable convolution kernel for nonlocal features
    calculator : WeightedDensityCalculator
        For computing FMT weighted densities (to get η = n₃)
    grid : Grid
        Computational grid with Fourier-space wavevectors

    Example
    -------
    >>> extractor = NonlocalFeatureExtractor(kernel, calculator, grid)
    >>> features = extractor(rho)  # Shape (nx, ny, nz, 5)
    """

    kernel: LearnableKernel
    calculator: WeightedDensityCalculator
    grid: Grid

    def __init__(self, kernel: LearnableKernel,
                 calculator: WeightedDensityCalculator,
                 grid: Grid):
        self.kernel = kernel
        self.calculator = calculator
        self.grid = grid

    def __call__(self, rho: Array) -> Array:
        """
        Extract nonlocal features from density field.

        Parameters
        ----------
        rho : Array
            Density field ρ(r), shape (nx, ny, nz)

        Returns
        -------
        features : Array
            Feature tensor, shape (nx, ny, nz, 5)
        """
        # ──────────────────────────────────────────────────────────
        # Local packing fraction from FMT weighted densities
        # ──────────────────────────────────────────────────────────
        measures = self.calculator(rho)
        eta = measures.eta  # n₃(r), shape (nx, ny, nz)

        # ──────────────────────────────────────────────────────────
        # Nonlocal smoothed packing fraction via learnable kernel
        # η̄(r) = ∫ K(r-r') η(r') dr'
        # ──────────────────────────────────────────────────────────
        eta_bar = self.kernel.convolve(eta, self.grid)

        # ──────────────────────────────────────────────────────────
        # Spectral gradient and Laplacian of η
        # ──────────────────────────────────────────────────────────
        eta_hat = jnp.fft.fftn(eta)

        # ∇η via spectral differentiation
        grad_eta_x = jnp.real(jnp.fft.ifftn(1j * self.grid.Kx * eta_hat))
        grad_eta_y = jnp.real(jnp.fft.ifftn(1j * self.grid.Ky * eta_hat))
        grad_eta_z = jnp.real(jnp.fft.ifftn(1j * self.grid.Kz * eta_hat))
        grad_eta_mag = jnp.sqrt(
            grad_eta_x**2 + grad_eta_y**2 + grad_eta_z**2 + 1e-20
        )

        # ∇²η via spectral Laplacian
        laplacian_eta = jnp.real(jnp.fft.ifftn(-self.grid.k_sq * eta_hat))

        # ──────────────────────────────────────────────────────────
        # Normalize and stack features
        # ──────────────────────────────────────────────────────────

        # 1. Local η normalized to O(1) range
        f_eta = eta / 0.5

        # 2. Nonlocal η̄ normalized similarly
        f_eta_bar = eta_bar / 0.5

        # 3. Gradient magnitude (scale by particle diameter σ=1)
        f_grad = grad_eta_mag

        # 4. Laplacian (scale by σ²=1)
        f_laplacian = laplacian_eta

        # 5. Local deviation from nonlocal average
        f_deviation = (eta - eta_bar) / (jnp.abs(eta_bar) + 1e-10)

        features = jnp.stack([
            f_eta,
            f_eta_bar,
            f_grad,
            f_laplacian,
            f_deviation
        ], axis=-1)

        return features

    def __repr__(self) -> str:
        return (f"NonlocalFeatureExtractor("
                f"kernel={self.kernel}, "
                f"grid={self.grid})")
