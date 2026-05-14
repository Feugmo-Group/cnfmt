"""
Learnable Nonlocal Kernels
==========================

Radial convolution kernels for nonlocal weighted density computation.

The kernel K(r) smooths the local packing fraction η(r) to produce
a nonlocal average η̄(r) = ∫ K(r-r') η(r') dr' that captures
packing correlations beyond the local FMT weights.

Parameterization
----------------
Sum of Gaussians with learnable amplitudes and widths:

    K(r) = Σᵢ aᵢ exp(-r²/2σᵢ²) × f_cut(r)

where f_cut is a smooth cutoff and amplitudes are normalized so ∫K(r)dr = 1.

Fourier Transform (analytical)
------------------------------
Each Gaussian component has:

    K̂ᵢ(k) = aᵢ (2π σᵢ²)^(3/2) exp(-σᵢ²k²/2)

This allows efficient FFT-based convolution.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
from core.grid import Grid


class LearnableKernel(eqx.Module):
    """
    Learnable radial kernel for nonlocal weighted density.

    Parameterized as a normalized sum of Gaussians with smooth cutoff.
    All parameters are differentiable for end-to-end training.

    Parameters
    ----------
    key : PRNGKey
        Random key for initialization
    n_gaussians : int
        Number of Gaussian components
    r_cut : float
        Cutoff radius in units of σ (particle diameter)
    init_width_range : tuple
        (min, max) for initial Gaussian widths

    Attributes
    ----------
    log_raw_amplitudes : Array
        Raw (pre-softmax) amplitudes, shape (n_gaussians,)
    log_widths : Array
        Log of Gaussian widths, shape (n_gaussians,)

    Example
    -------
    >>> key = jax.random.PRNGKey(0)
    >>> kernel = LearnableKernel(key, n_gaussians=8)
    >>> K_hat = kernel.fourier(grid.k_abs)
    """

    # Learnable parameters
    log_raw_amplitudes: Array
    log_widths: Array

    # Static configuration
    n_gaussians: int = eqx.field(static=True)
    r_cut: float = eqx.field(static=True)

    def __init__(self, key: jax.random.PRNGKey,
                 n_gaussians: int = 8,
                 r_cut: float = 2.0,
                 init_width_range: tuple = (0.2, 1.5)):
        """
        Initialize learnable kernel.

        Widths initialized log-uniformly in init_width_range.
        Amplitudes initialized uniformly (equal weight per Gaussian).
        """
        self.n_gaussians = n_gaussians
        self.r_cut = r_cut

        k1, k2 = jax.random.split(key)

        # Initialize widths log-uniformly
        log_min = jnp.log(init_width_range[0])
        log_max = jnp.log(init_width_range[1])
        self.log_widths = jax.random.uniform(
            k1, (n_gaussians,), minval=log_min, maxval=log_max
        )

        # Initialize amplitudes near uniform (all Gaussians contribute equally)
        self.log_raw_amplitudes = jnp.zeros(n_gaussians) + jax.random.normal(
            k2, (n_gaussians,)
        ) * 0.1

    @property
    def widths(self) -> Array:
        """Gaussian widths σᵢ (always positive via exp)."""
        return jnp.exp(self.log_widths)

    @property
    def amplitudes(self) -> Array:
        """Normalized amplitudes (sum to 1 via softmax)."""
        return jax.nn.softmax(self.log_raw_amplitudes)

    def real_space(self, r: Array) -> Array:
        """
        Evaluate kernel in real space.

        K(r) = Σᵢ aᵢ × N(σᵢ) × exp(-r²/2σᵢ²) × f_cut(r)

        where N(σᵢ) normalizes each Gaussian and f_cut is a smooth cutoff.

        Parameters
        ----------
        r : Array
            Radial distances, any shape

        Returns
        -------
        K : Array
            Kernel values, same shape as r
        """
        a = self.amplitudes      # (n_gaussians,)
        sigma = self.widths      # (n_gaussians,)

        # Gaussian normalization: (2πσ²)^(-3/2) for 3D
        norm = (2 * jnp.pi * sigma**2) ** (-1.5)

        # Smooth cutoff: cosine taper from r_cut - Δ to r_cut
        delta = 0.2 * self.r_cut
        f_cut = jnp.where(
            r < self.r_cut - delta,
            1.0,
            jnp.where(
                r < self.r_cut,
                0.5 * (1 + jnp.cos(jnp.pi * (r - self.r_cut + delta) / delta)),
                0.0
            )
        )

        # Sum of Gaussians
        # Broadcast: r -> (1,) + r.shape, sigma -> (n_gaussians,) + (1,)*r.ndim
        r_exp = jnp.expand_dims(r, axis=0)                       # (1, *r.shape)
        sigma_exp = sigma.reshape((-1,) + (1,) * r.ndim)         # (n_g, 1, ...)
        a_exp = a.reshape((-1,) + (1,) * r.ndim)                 # (n_g, 1, ...)
        norm_exp = norm.reshape((-1,) + (1,) * r.ndim)           # (n_g, 1, ...)

        gaussians = norm_exp * jnp.exp(-r_exp**2 / (2 * sigma_exp**2))
        K = jnp.sum(a_exp * gaussians, axis=0) * f_cut

        return K

    def fourier(self, k: Array) -> Array:
        """
        Evaluate kernel Fourier transform (analytical).

        K̂(k) = Σᵢ aᵢ exp(-σᵢ²k²/2)

        The (2πσ²)^(3/2) normalization from the Gaussian FT cancels
        with the (2πσ²)^(-3/2) real-space normalization, leaving just
        the exponential decay.

        Parameters
        ----------
        k : Array
            Wavevector magnitudes |k|, any shape

        Returns
        -------
        K_hat : Array
            Fourier-space kernel, same shape as k. Real-valued.
        """
        a = self.amplitudes      # (n_gaussians,)
        sigma = self.widths      # (n_gaussians,)

        # Broadcast for vectorized computation
        k_exp = jnp.expand_dims(k, axis=0)                       # (1, *k.shape)
        sigma_exp = sigma.reshape((-1,) + (1,) * k.ndim)         # (n_g, 1, ...)
        a_exp = a.reshape((-1,) + (1,) * k.ndim)                 # (n_g, 1, ...)

        # Analytical FT of normalized Gaussian
        components = a_exp * jnp.exp(-0.5 * sigma_exp**2 * k_exp**2)
        K_hat = jnp.sum(components, axis=0)

        return K_hat

    def convolve(self, field: Array, grid: Grid) -> Array:
        """
        Convolve a 3D field with this kernel via FFT.

        result(r) = ∫ K(r-r') field(r') dr'

        Parameters
        ----------
        field : Array
            Input field, shape (nx, ny, nz)
        grid : Grid
            Computational grid

        Returns
        -------
        result : Array
            Convolved field, shape (nx, ny, nz)
        """
        K_hat = self.fourier(grid.k_abs)
        field_hat = jnp.fft.fftn(field)
        result = jnp.real(jnp.fft.ifftn(field_hat * K_hat))
        return result

    def __repr__(self) -> str:
        return (f"LearnableKernel(n_gaussians={self.n_gaussians}, "
                f"r_cut={self.r_cut})")
