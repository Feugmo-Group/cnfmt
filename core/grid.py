"""
3D Computational Grid
=====================

Provides the spatial discretization for DFT calculations with FFT support.

The grid is periodic in all three dimensions with:
- Real-space coordinates: x, y, z ∈ [0, L)
- Fourier-space wavevectors: k_x, k_y, k_z

Example
-------
>>> grid = Grid((32, 32, 32), 12.0)
>>> print(f"Volume: {grid.volume:.2f}, dV: {grid.dV:.6f}")
"""

import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Union
from jaxtyping import Array


class Grid(eqx.Module):
    """
    3D computational grid with FFT support.
    
    Parameters
    ----------
    n_grid : Tuple[int, int, int]
        Number of grid points in each dimension (nx, ny, nz)
    length : float or Tuple[float, float, float]
        Box length(s). If scalar, same length in all dimensions.
    
    Attributes
    ----------
    nx, ny, nz : int
        Grid dimensions
    Lx, Ly, Lz : float
        Box lengths
    dx, dy, dz : float
        Grid spacings
    dV : float
        Volume element
    x, y, z : Array
        1D coordinate arrays
    X, Y, Z : Array
        3D meshgrid arrays
    kx, ky, kz : Array
        1D wavevector arrays
    Kx, Ky, Kz : Array
        3D wavevector meshgrids
    k_sq : Array
        |k|² for each point
    k_abs : Array
        |k| for each point
    """
    
    # Grid dimensions (static - don't change)
    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    nz: int = eqx.field(static=True)
    
    # Box lengths (static)
    Lx: float = eqx.field(static=True)
    Ly: float = eqx.field(static=True)
    Lz: float = eqx.field(static=True)
    
    # Grid spacings (static)
    dx: float = eqx.field(static=True)
    dy: float = eqx.field(static=True)
    dz: float = eqx.field(static=True)
    dV: float = eqx.field(static=True)
    
    # Real-space coordinates
    x: Array
    y: Array
    z: Array
    X: Array
    Y: Array
    Z: Array
    
    # Fourier-space wavevectors
    kx: Array
    ky: Array
    kz: Array
    Kx: Array
    Ky: Array
    Kz: Array
    k_sq: Array
    k_abs: Array
    
    def __init__(self, n_grid: Tuple[int, int, int], 
                 length: Union[float, Tuple[float, float, float]]):
        """Initialize computational grid."""
        self.nx, self.ny, self.nz = n_grid
        
        # Handle scalar or tuple length
        if isinstance(length, (int, float)):
            self.Lx = self.Ly = self.Lz = float(length)
        else:
            self.Lx, self.Ly, self.Lz = length
        
        # Grid spacings
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        self.dV = self.dx * self.dy * self.dz
        
        # Real-space coordinates [0, L)
        self.x = jnp.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = jnp.linspace(0, self.Ly, self.ny, endpoint=False)
        self.z = jnp.linspace(0, self.Lz, self.nz, endpoint=False)
        self.X, self.Y, self.Z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Fourier-space wavevectors
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(self.nx, self.dx)
        self.ky = 2 * jnp.pi * jnp.fft.fftfreq(self.ny, self.dy)
        self.kz = 2 * jnp.pi * jnp.fft.fftfreq(self.nz, self.dz)
        self.Kx, self.Ky, self.Kz = jnp.meshgrid(self.kx, self.ky, self.kz, indexing='ij')
        
        # Wavevector magnitude
        self.k_sq = self.Kx**2 + self.Ky**2 + self.Kz**2
        self.k_abs = jnp.sqrt(self.k_sq + 1e-14)  # Small regularization
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Grid shape (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)
    
    @property
    def volume(self) -> float:
        """Total box volume."""
        return self.Lx * self.Ly * self.Lz
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Box center coordinates."""
        return (self.Lx / 2, self.Ly / 2, self.Lz / 2)
    
    @property
    def n_points(self) -> int:
        """Total number of grid points."""
        return self.nx * self.ny * self.nz
    
    def distance_from_center(self) -> Array:
        """Compute distance from box center for each grid point."""
        cx, cy, cz = self.center
        return jnp.sqrt((self.X - cx)**2 + (self.Y - cy)**2 + (self.Z - cz)**2)
    
    def __repr__(self) -> str:
        return f"Grid({self.shape}, L=({self.Lx}, {self.Ly}, {self.Lz}))"
