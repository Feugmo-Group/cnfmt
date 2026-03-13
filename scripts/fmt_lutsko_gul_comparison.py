"""
Comprehensive FMT Comparison: Lutsko, Gül et al., Rosenfeld, CS
================================================================

Uses validated FFT-based 1D FMT implementation for density profiles.
Compares different (A, B) parametrizations of the Lutsko functional.

Functionals compared:
- Rosenfeld (A=1.5, B=0): Original FMT, gives PY c(r)
- Lutsko (A=1.0, B=0): C=-1, close to CS EOS  
- Gül et al. (A=1.3, B=-1.0): C=-0.6, optimized for test particle
- CS (exact): Reference equation of state

Author: Computational Materials Science
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict

jax.config.update("jax_enable_x64", True)

PI = np.pi


# ============================================================================
# MC DATA FROM DAVIDCHACK, LAIRD, ROTH (2016)
# ============================================================================

MC_PROFILES = {
    0.367: np.array([
        [0.510, 3.7543], [0.530, 3.2699], [0.550, 2.8547], [0.570, 2.4987],
        [0.590, 2.1930], [0.610, 1.9302], [0.630, 1.7045], [0.650, 1.5099],
        [0.670, 1.3422], [0.690, 1.1976], [0.710, 1.0726], [0.730, 0.9646],
        [0.750, 0.8712], [0.770, 0.7902], [0.790, 0.7201], [0.810, 0.6593],
        [0.830, 0.6066], [0.850, 0.5609], [0.870, 0.5215], [0.890, 0.4875],
        [0.910, 0.4582], [0.930, 0.4332], [0.950, 0.4119], [0.970, 0.3941],
        [0.990, 0.3794], [1.010, 0.3675], [1.030, 0.3583], [1.050, 0.3516],
        [1.070, 0.3474], [1.090, 0.3455], [1.110, 0.3460], [1.130, 0.3489],
        [1.150, 0.3543], [1.170, 0.3623], [1.190, 0.3730], [1.210, 0.3867],
        [1.230, 0.4037], [1.250, 0.4241], [1.270, 0.4486], [1.290, 0.4774],
        [1.310, 0.5108], [1.330, 0.5495], [1.350, 0.5936], [1.370, 0.6432],
        [1.390, 0.6977], [1.410, 0.7559], [1.430, 0.8155], [1.450, 0.8744],
        [1.470, 0.9307], [1.490, 0.9829], [1.510, 1.0267],
    ]),
}


# ============================================================================
# THERMODYNAMICS
# ============================================================================

def Z_CS(eta):
    """Carnahan-Starling compressibility factor."""
    return (1 + eta + eta**2 - eta**3) / (1 - eta)**3

def Z_PY(eta):
    """Percus-Yevick compressibility factor."""
    return (1 + eta + eta**2) / (1 - eta)**3

def Z_Lutsko(eta, A, B):
    """Lutsko compressibility factor."""
    C = 8*A + 2*B - 9
    return Z_PY(eta) + C * eta**2 / (3 * (1 - eta)**3)

def mu_ex_CS(eta):
    """Carnahan-Starling excess chemical potential."""
    return eta * (8 - 9*eta + 3*eta**2) / (1 - eta)**3

def mu_ex_Lutsko(eta, A, B):
    """Lutsko excess chemical potential."""
    C = 8*A + 2*B - 9
    mu_RF = -np.log(1 - eta) + eta * (14 - 13*eta + 5*eta**2) / (2*(1-eta)**3)
    return mu_RF + C * eta**2 * (3 - eta) / (6 * (1-eta)**3)


# ============================================================================
# 1D FMT GRID AND KERNELS (FFT-based, validated)
# ============================================================================

class Grid1D:
    """1D grid for planar geometry."""
    def __init__(self, nz: int, Lz: float):
        self.nz = nz
        self.Lz = Lz
        self.dz = Lz / nz
        self.z = jnp.linspace(0.5 * self.dz, Lz - 0.5 * self.dz, nz)
        self.kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, self.dz)


class FMTKernels1D:
    """FMT weight functions in Fourier space."""
    def __init__(self, grid: Grid1D, R: float):
        self.R = R
        k = jnp.abs(grid.kz)
        eps = 1e-12
        kR = k * R
        
        # Scalar weights
        self.w3_hat = jnp.where(k < eps, (4/3)*jnp.pi*R**3,
            (4/3)*jnp.pi*R**3 * 3*(jnp.sin(kR) - kR*jnp.cos(kR))/(kR**3 + eps))
        self.w2_hat = jnp.where(k < eps, 4*jnp.pi*R**2,
            4*jnp.pi*R**2 * jnp.sin(kR)/(kR + eps))
        self.w1_hat = self.w2_hat / (4*jnp.pi*R)
        self.w0_hat = self.w2_hat / (4*jnp.pi*R**2)
        
        # Vector weight (z-component)
        self.wv2_z_hat = jnp.where(k < eps, 0.0,
            -1j * 4*jnp.pi*R * (jnp.sin(kR) - kR*jnp.cos(kR)) / (k**2 + eps))
        self.wv1_z_hat = self.wv2_z_hat / (4*jnp.pi*R)


def compute_weighted_densities(rho: jnp.ndarray, kernels: FMTKernels1D) -> Dict:
    """Compute weighted densities via FFT convolution."""
    rho_hat = jnp.fft.fft(rho)
    
    eta = jnp.real(jnp.fft.ifft(rho_hat * kernels.w3_hat))
    n0 = jnp.real(jnp.fft.ifft(rho_hat * kernels.w0_hat))
    n1 = jnp.real(jnp.fft.ifft(rho_hat * kernels.w1_hat))
    n2 = jnp.real(jnp.fft.ifft(rho_hat * kernels.w2_hat))
    nv1_z = jnp.real(jnp.fft.ifft(rho_hat * kernels.wv1_z_hat))
    nv2_z = jnp.real(jnp.fft.ifft(rho_hat * kernels.wv2_z_hat))
    
    return {
        'eta': eta, 'n0': n0, 'n1': n1, 'n2': n2,
        'nv1_z': nv1_z, 'nv2_z': nv2_z,
        'nv1_dot_nv2': nv1_z * nv2_z,
        'nv2_sq': nv2_z**2
    }


# ============================================================================
# LUTSKO FUNCTIONAL WITH (A, B) PARAMETERS
# ============================================================================

class LutskoFunctional:
    """
    Generalized Lutsko FMT functional with (A, B) parameters.
    
    Φ = Φ₁ + A·Φ₂ + B·Φ₃
    
    where:
    Φ₁ = -n₀ ln(1 - η)
    Φ₂ = (n₁n₂ - nv1·nv2)/(1 - η)
    Φ₃ = (n₂³ - 3n₂·nv2²)/(24π(1-η)²)
    
    Special cases:
    - Rosenfeld: A=1.5, B=0 (actually uses A=1 by convention, see below)
    - Lutsko: A=1, B=0, C=-1
    - Gül et al.: A=1.3, B=-1.0, C=-0.6
    """
    
    def __init__(self, grid: Grid1D, sigma: float = 1.0, A: float = 1.0, B: float = 0.0):
        self.grid = grid
        self.sigma = sigma
        self.R = sigma / 2
        self.A = A
        self.B = B
        self.kernels = FMTKernels1D(grid, self.R)
    
    def Phi_density(self, wd: Dict) -> jnp.ndarray:
        """Free energy density Φ(r)."""
        eps = 1e-10
        eta = jnp.clip(wd['eta'], eps, 1 - eps)
        one_m_eta = 1 - eta
        
        # Standard FMT terms with (A, B) parameters
        phi1 = -wd['n0'] * jnp.log(one_m_eta)
        phi2 = self.A * (wd['n1'] * wd['n2'] - wd['nv1_dot_nv2']) / one_m_eta
        phi3 = self.B * (wd['n2']**3 - 3 * wd['n2'] * wd['nv2_sq']) / (24 * jnp.pi * one_m_eta**2)
        
        return phi1 + phi2 + phi3
    
    def c1_analytical(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        One-body DCF via analytical chain rule:
        c1 = -Σ_α (∂Φ/∂n_α ★ w_α)
        """
        eps = 1e-10
        wd = compute_weighted_densities(rho, self.kernels)
        
        eta = jnp.clip(wd['eta'], eps, 1 - eps)
        n0, n1, n2 = wd['n0'], wd['n1'], wd['n2']
        nv2_sq = wd['nv2_sq']
        one_m_eta = 1 - eta
        
        # Partial derivatives of Φ
        dPhi_deta = (n0 / one_m_eta + 
                   self.A * (n1*n2 - wd['nv1_dot_nv2']) / one_m_eta**2 +
                   self.B * (n2**3 - 3*n2*nv2_sq) / (12*jnp.pi*one_m_eta**3))
        
        dPhi_dn0 = -jnp.log(one_m_eta)
        dPhi_dn1 = self.A * n2 / one_m_eta
        dPhi_dn2 = self.A * n1 / one_m_eta + self.B * (3*n2**2 - 3*nv2_sq) / (24*jnp.pi*one_m_eta**2)
        dPhi_dnv1 = -self.A * wd['nv2_z'] / one_m_eta
        dPhi_dnv2 = -self.A * wd['nv1_z'] / one_m_eta - self.B * n2*wd['nv2_z'] / (4*jnp.pi*one_m_eta**2)
        
        # Convolutions in Fourier space
        k = self.kernels
        c1_hat = -(jnp.fft.fft(dPhi_deta) * k.w3_hat +
                  jnp.fft.fft(dPhi_dn0) * k.w0_hat +
                  jnp.fft.fft(dPhi_dn1) * k.w1_hat +
                  jnp.fft.fft(dPhi_dn2) * k.w2_hat +
                  jnp.fft.fft(dPhi_dnv1) * jnp.conj(k.wv1_z_hat) +
                  jnp.fft.fft(dPhi_dnv2) * jnp.conj(k.wv2_z_hat))
        
        return jnp.real(jnp.fft.ifft(c1_hat))


class WallSolver:
    """Solve for density profile at hard wall using Picard iteration."""
    
    def __init__(self, grid: Grid1D, sigma: float = 1.0):
        self.grid = grid
        self.sigma = sigma
        self.R = sigma / 2
    
    def solve(self, eta: float, A: float, B: float, 
              n_iter: int = 1500, verbose: bool = False) -> Dict:
        """Solve for equilibrium profile."""
        
        functional = LutskoFunctional(self.grid, self.sigma, A, B)
        
        # Bulk quantities
        rho_bulk = eta / ((4/3) * PI * self.R**3)
        
        # Bulk c1 reference
        rho_uniform = jnp.ones_like(self.grid.z) * rho_bulk
        c1_bulk = functional.c1_analytical(rho_uniform)
        c1_bulk_ref = float(jnp.mean(c1_bulk[self.grid.z > 3.0]))
        
        # Initialize
        rho = jnp.where(self.grid.z < self.R, 0.0, rho_bulk)
        
        for i in range(n_iter):
            c1 = functional.c1_analytical(rho)
            
            # Picard update
            rho_new = rho_bulk * jnp.exp(c1 - c1_bulk_ref)
            rho_new = jnp.where(self.grid.z < self.R, 0.0, rho_new)
            rho_new = jnp.clip(rho_new, 0.0, 50 * rho_bulk)
            
            # Adaptive mixing
            if i < 100:
                alpha = 0.02
            elif i < 300:
                alpha = 0.05
            elif i < 800:
                alpha = 0.1
            else:
                alpha = 0.15
            
            rho = alpha * rho_new + (1 - alpha) * rho
            rho = jnp.where(self.grid.z < self.R, 0.0, rho)
            
            if verbose and i % 300 == 0:
                contact_idx = jnp.argmin(jnp.abs(self.grid.z - 0.51))
                print(f"    Iter {i}: ρ(contact)σ³ = {float(rho[contact_idx] * self.sigma**3):.4f}")
        
        # Results
        z = np.array(self.grid.z)
        rho_sigma3 = np.array(rho * self.sigma**3)
        
        contact_idx = np.argmin(np.abs(z - 0.51))
        contact = rho_sigma3[contact_idx]
        
        return {
            'z': z,
            'rho': rho_sigma3,
            'contact': contact,
            'rho_bulk': rho_bulk * self.sigma**3,
            'eta': eta,
            'A': A,
            'B': B
        }


# ============================================================================
# DIRECT CORRELATION FUNCTIONS
# ============================================================================

def c_PY_real(r: np.ndarray, eta: float, sigma: float = 1.0) -> np.ndarray:
    """Percus-Yevick c(r) in real space (analytical)."""
    alpha = (1 + 2*eta)**2 / (1 - eta)**4
    beta = 6*eta * (1 + eta/2)**2 / (1 - eta)**4
    gamma = eta * (1 + 2*eta)**2 / (2*(1 - eta)**4)
    
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)


def c_Lutsko_real(r: np.ndarray, eta: float, A: float, B: float, sigma: float = 1.0) -> np.ndarray:
    """
    Direct correlation function for Lutsko functional.
    
    The (A, B) parameters modify the polynomial coefficients.
    """
    one_m_eta = 1 - eta
    C = 8*A + 2*B - 9
    
    # Modified coefficients
    alpha = (1 + 2*eta)**2 / one_m_eta**4
    beta = 6*eta * (1 + eta/2)**2 / one_m_eta**4
    gamma = eta * (1 + 2*eta)**2 / (2*one_m_eta**4)
    
    # Corrections from (A, B)
    alpha += C * eta**2 / (3*one_m_eta**4) * (A - 1)
    gamma += C * eta**3 / (6*one_m_eta**4) * B
    
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)


def c_fourier(r: np.ndarray, c_r: np.ndarray, k_max: float = 25.0, nk: int = 256) -> tuple:
    """Compute ĉ(k) from c(r) via numerical FT."""
    k = np.linspace(0.01, k_max, nk)
    dr = r[1] - r[0]
    
    c_k = np.zeros_like(k)
    for i, ki in enumerate(k):
        sinc = np.where(ki * r > 1e-10, np.sin(ki * r) / (ki * r), 1.0)
        c_k[i] = 4*PI * np.sum(r**2 * c_r * sinc) * dr
    
    return k, c_k


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison():
    """Run comprehensive FMT comparison."""
    
    print("="*70)
    print("COMPREHENSIVE FMT COMPARISON")
    print("Lutsko, Gül et al., Rosenfeld vs Carnahan-Starling")
    print("="*70)
    
    eta = 0.367
    sigma = 1.0
    rho_bulk = 6 * eta / (PI * sigma**3)
    
    # MC reference
    MC_contact = 3.7543  # From Davidchack et al. normalized to ρσ³
    CS_Z = Z_CS(eta)
    PY_Z = Z_PY(eta)
    
    print(f"\nPacking fraction η = {eta}")
    print(f"MC contact density ρ(R⁺)σ³: {MC_contact:.4f}")
    print(f"Carnahan-Starling Z: {CS_Z:.4f}")
    print(f"Percus-Yevick Z: {PY_Z:.4f}")
    
    # Define functionals
    functionals = {
        'Rosenfeld': {'A': 1.0, 'B': 0.0, 'color': 'C0'},  # Standard Rosenfeld
        'Lutsko': {'A': 1.0, 'B': 0.0, 'color': 'C1'},      # Same as Rosenfeld base
        'Gül et al.': {'A': 1.3, 'B': -1.0, 'color': 'C2'},
        'esFMT(1.5,0)': {'A': 1.5, 'B': 0.0, 'color': 'C3'},  # More PY-like
    }
    
    # Print C values
    print(f"\n{'Functional':<15} {'A':>6} {'B':>8} {'C=8A+2B-9':>12} {'Z(η)':>10}")
    print("-"*55)
    for name, params in functionals.items():
        A, B = params['A'], params['B']
        C = 8*A + 2*B - 9
        Z = Z_Lutsko(eta, A, B)
        print(f"{name:<15} {A:6.2f} {B:8.2f} {C:12.2f} {Z:10.4f}")
    print(f"{'CS (exact)':<15} {'':>6} {'':>8} {'-3.00':>12} {CS_Z:10.4f}")
    
    # =========================================================================
    # PART 1: DENSITY PROFILES
    # =========================================================================
    print("\n" + "-"*70)
    print("PART 1: DENSITY PROFILES AT HARD WALL")
    print("-"*70)
    
    grid = Grid1D(nz=1024, Lz=8.0)
    solver = WallSolver(grid, sigma)
    
    profiles = {}
    contact_densities = {}
    
    for name, params in functionals.items():
        print(f"\nSolving for {name} (A={params['A']}, B={params['B']})...")
        result = solver.solve(eta, params['A'], params['B'], n_iter=1500, verbose=True)
        profiles[name] = result
        contact_densities[name] = result['contact']
    
    # =========================================================================
    # PART 2: DIRECT CORRELATION FUNCTIONS
    # =========================================================================
    print("\n" + "-"*70)
    print("PART 2: DIRECT CORRELATION FUNCTIONS")
    print("-"*70)
    
    r = np.linspace(0.001, 1.5*sigma, 512)
    c_r_PY = c_PY_real(r, eta, sigma)
    k_PY, c_k_PY = c_fourier(r, c_r_PY)
    
    correlations = {'PY': (r, c_r_PY, k_PY, c_k_PY)}
    
    for name, params in functionals.items():
        c_r = c_Lutsko_real(r, eta, params['A'], params['B'], sigma)
        k, c_k = c_fourier(r, c_r)
        correlations[name] = (r, c_r, k, c_k)
    
    # =========================================================================
    # CREATE FIGURE
    # =========================================================================
    print("\n" + "-"*70)
    print("Creating figure...")
    print("-"*70)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    colors = {name: params['color'] for name, params in functionals.items()}
    colors['PY'] = 'black'
    colors['CS'] = 'red'
    colors['MC'] = 'red'
    
    # ----- (a) Density Profiles -----
    ax1 = fig.add_subplot(gs[0, 0])
    
    # MC data
    mc_data = MC_PROFILES[eta]
    ax1.plot(mc_data[:, 0], mc_data[:, 1], 'o', color='red', ms=4, 
             mfc='none', mew=1, alpha=0.8, label='MC')
    
    for name, result in profiles.items():
        ax1.plot(result['z'], result['rho'], '-', color=colors[name], 
                 lw=1.5, label=name)
    
    ax1.axhline(result['rho_bulk'], color='gray', ls='--', alpha=0.5, label='ρ_bulk')
    ax1.axhline(MC_contact, color='red', ls=':', lw=2, label=f'MC contact = {MC_contact:.2f}')
    ax1.axvline(0.5, color='gray', ls='--', alpha=0.5)
    
    ax1.set_xlabel(r'$z/\sigma$', fontsize=12)
    ax1.set_ylabel(r'$\rho(z)\sigma^3$', fontsize=12)
    ax1.set_title(f'(a) Density Profile at Hard Wall (η = {eta})', fontsize=12)
    ax1.set_xlim([0.4, 3.0])
    ax1.set_ylim([0, 5])
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ----- (b) Contact Density Bar Chart -----
    ax2 = fig.add_subplot(gs[0, 1])
    
    names = list(contact_densities.keys())
    contacts = [contact_densities[n] for n in names]
    Z_values = [Z_Lutsko(eta, functionals[n]['A'], functionals[n]['B']) for n in names]
    x_pos = np.arange(len(names))
    
    bars = ax2.bar(x_pos, contacts, color=[colors[n] for n in names], alpha=0.8)
    ax2.axhline(MC_contact, color='red', ls='--', lw=2, label=f'MC = {MC_contact:.2f}')
    ax2.axhline(CS_Z * rho_bulk * sigma**3, color='green', ls=':', lw=2, 
                label=f'CS contact = {CS_Z * rho_bulk * sigma**3:.2f}')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax2.set_ylabel(r'Contact Density $\rho(R^+)\sigma^3$', fontsize=11)
    ax2.set_title('(b) Contact Density Comparison', fontsize=12)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ----- (c) c(r) Real Space -----
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(r, c_r_PY, '-', color='black', lw=2.5, label='PY (analytical)')
    for name in functionals.keys():
        r_, c_r_, k_, c_k_ = correlations[name]
        ax3.plot(r_, c_r_, '--', color=colors[name], lw=1.5, label=name)
    
    ax3.axhline(0, color='gray', ls='-', alpha=0.3)
    ax3.axvline(1.0, color='gray', ls='--', alpha=0.5, label=r'$r = \sigma$')
    
    ax3.set_xlabel(r'$r/\sigma$', fontsize=12)
    ax3.set_ylabel(r'$c(r)$', fontsize=12)
    ax3.set_title(f'(c) Direct Correlation Function c(r) (η = {eta})', fontsize=12)
    ax3.set_xlim([0, 1.2])
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # ----- (d) ĉ(k) Fourier Space -----
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(k_PY, c_k_PY, '-', color='black', lw=2.5, label='PY (analytical)')
    for name in functionals.keys():
        r_, c_r_, k_, c_k_ = correlations[name]
        ax4.plot(k_, c_k_, '--', color=colors[name], lw=1.5, label=name)
    
    ax4.axhline(0, color='gray', ls='-', alpha=0.3)
    
    ax4.set_xlabel(r'$k\sigma$', fontsize=12)
    ax4.set_ylabel(r'$\hat{c}(k)$', fontsize=12)
    ax4.set_title(f'(d) Fourier Transform ĉ(k) (η = {eta})', fontsize=12)
    ax4.set_xlim([0, 20])
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('FMT Comparison: Density Profiles and Direct Correlation Functions', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('/mnt/user-data/outputs/fmt_lutsko_gul_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("Saved: fmt_lutsko_gul_comparison.png")
    plt.close()
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Functional':<15} {'Contact ρσ³':>12} {'% of MC':>10} {'C':>8}")
    print("-"*50)
    for name, contact in contact_densities.items():
        C = 8*functionals[name]['A'] + 2*functionals[name]['B'] - 9
        pct = contact / MC_contact * 100
        print(f"{name:<15} {contact:12.4f} {pct:9.1f}% {C:8.2f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    run_comparison()
