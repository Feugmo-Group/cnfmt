"""
Comprehensive FMT Comparison - Using Validated Implementation
=============================================================

This script uses the validated 1D planar FMT solver with proper
chain rule for c⁽¹⁾(z) calculation to produce accurate density profiles.

Compares: Rosenfeld, Lutsko, Gül et al., CS

Author: Computational Materials Science
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple, List

PI = np.pi


# ============================================================================
# MC WALL PROFILE DATA (Davidchack, Laird, Roth 2016)
# ============================================================================

MC_PROFILES = {
    0.367: np.array([
        [0.510, 3.7543085], [0.530, 3.2698767], [0.550, 2.8546749],
        [0.570, 2.4986631], [0.590, 2.1929623], [0.610, 1.9302458],
        [0.630, 1.7044568], [0.650, 1.5098530], [0.670, 1.3422220],
        [0.690, 1.1976265], [0.710, 1.0726264], [0.730, 0.9646101],
        [0.750, 0.8711540], [0.770, 0.7901845], [0.790, 0.7200606],
        [0.810, 0.6592646], [0.830, 0.6065577], [0.850, 0.5609323],
        [0.870, 0.5215091], [0.890, 0.4874595], [0.910, 0.4582073],
        [0.930, 0.4331748], [0.950, 0.4119227], [0.970, 0.3940790],
        [0.990, 0.3793644], [1.010, 0.3675033], [1.030, 0.3583127],
        [1.050, 0.3516432], [1.070, 0.3474103], [1.090, 0.3455326],
        [1.110, 0.3460356], [1.130, 0.3489446], [1.150, 0.3543193],
        [1.170, 0.3622934], [1.190, 0.3729875], [1.210, 0.3866957],
        [1.230, 0.4036505], [1.250, 0.4241440], [1.270, 0.4485735],
        [1.290, 0.4773881], [1.310, 0.5108371], [1.330, 0.5495006],
        [1.350, 0.5935853], [1.370, 0.6431678], [1.390, 0.6977321],
        [1.410, 0.7558864], [1.430, 0.8155218], [1.450, 0.8744027],
        [1.470, 0.9307192], [1.490, 0.9828855], [1.510, 1.0267341],
        [1.530, 1.0516409], [1.550, 1.0589684], [1.570, 1.0532432],
        [1.590, 1.0376837], [1.610, 1.0151697], [1.630, 0.9879103],
        [1.650, 0.9575533], [1.670, 0.9254464], [1.690, 0.8927105],
        [1.710, 0.8601011], [1.730, 0.8282547], [1.750, 0.7975791],
        [1.770, 0.7683927], [1.790, 0.7409146], [1.810, 0.7152955],
        [1.830, 0.6916027], [1.850, 0.6699323], [1.870, 0.6503300],
        [1.890, 0.6326470], [1.910, 0.6170403], [1.930, 0.6033605],
        [1.950, 0.5916013], [1.970, 0.5817735], [1.990, 0.5737707],
        [2.010, 0.5675866], [2.030, 0.5631688], [2.050, 0.5605595],
        [2.070, 0.5596782], [2.090, 0.5604436], [2.110, 0.5629116],
        [2.130, 0.5670371], [2.150, 0.5728226], [2.170, 0.5800886],
        [2.190, 0.5889496], [2.210, 0.5992337], [2.230, 0.6108698],
        [2.250, 0.6237762], [2.270, 0.6377732], [2.290, 0.6526653],
        [2.310, 0.6682662], [2.330, 0.6842706], [2.350, 0.7004005],
    ]),
}


# ============================================================================
# 1D PLANAR GRID AND KERNELS
# ============================================================================

class PlanarGrid:
    """1D grid for planar geometry."""
    def __init__(self, nz: int = 2048, Lz: float = 10.0):
        self.nz = nz
        self.Lz = Lz
        self.dz = Lz / nz
        self.z = jnp.linspace(0.5 * self.dz, Lz - 0.5 * self.dz, nz)
        self.kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, self.dz)


class PlanarFMTKernels:
    """1D FMT weight functions for planar geometry."""
    def __init__(self, grid: PlanarGrid, R: float):
        self.R = R
        k = jnp.abs(grid.kz)
        eps = 1e-12
        kR = k * R
        
        self.w3_hat = jnp.where(
            k < eps, (4.0/3.0) * jnp.pi * R**3,
            (4.0/3.0) * jnp.pi * R**3 * 3 * (jnp.sin(kR) - kR * jnp.cos(kR)) / (kR**3 + eps)
        )
        self.w2_hat = jnp.where(
            k < eps, 4.0 * jnp.pi * R**2,
            4.0 * jnp.pi * R**2 * jnp.sin(kR) / (kR + eps)
        )
        self.w1_hat = self.w2_hat / (4.0 * jnp.pi * R)
        self.w0_hat = self.w2_hat / (4.0 * jnp.pi * R**2)
        
        self.wv2_z_hat = jnp.where(
            k < eps, 0.0,
            -1j * 4.0 * jnp.pi * R * (jnp.sin(kR) - kR * jnp.cos(kR)) / (k**2 + eps)
        )
        self.wv1_z_hat = self.wv2_z_hat / (4.0 * jnp.pi * R)


def compute_weighted_densities(rho: jnp.ndarray, kernels: PlanarFMTKernels) -> Dict:
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
# LUTSKO FMT FUNCTIONAL
# ============================================================================

class LutskoFunctional:
    """
    Lutsko esFMT functional for 1D planar geometry.
    
    Φ = -n₀ ln(1-η) + (n₁n₂ - nv1·nv2)/(1-η) 
        + [A(n₂³ - 3n₂·nv2²) + B·n₂³]/(24π(1-η)²)
    """
    
    def __init__(self, grid: PlanarGrid, sigma: float = 1.0, 
                 A: float = 1.0, B: float = 0.0):
        self.grid = grid
        self.sigma = sigma
        self.R = sigma / 2
        self.A = A
        self.B = B
        self.kernels = PlanarFMTKernels(grid, self.R)
    
    def c1(self, rho: jnp.ndarray) -> jnp.ndarray:
        """
        One-body direct correlation c⁽¹⁾(z) via chain rule:
        c⁽¹⁾ = -Σ_α (∂Φ/∂n_α ★ w_α)
        """
        eps = 1e-10
        wd = compute_weighted_densities(rho, self.kernels)
        A, B = self.A, self.B
        
        eta = jnp.clip(wd['eta'], 0, 1 - eps)
        n0, n1, n2 = wd['n0'], wd['n1'], wd['n2']
        nv2_sq = wd['nv2_sq']
        one_m_eta = 1 - eta
        
        # Partial derivatives
        term3_eta = (A * (n2**3 - 3*n2*nv2_sq) + B * n2**3) / (12 * jnp.pi * one_m_eta**3)
        dPhi_deta = n0 / one_m_eta + (n1*n2 - wd['nv1_dot_nv2']) / one_m_eta**2 + term3_eta
        
        dPhi_dn0 = -jnp.log(one_m_eta)
        dPhi_dn1 = n2 / one_m_eta
        
        term3_n2 = (3*A*(n2**2 - nv2_sq) + 3*B*n2**2) / (24 * jnp.pi * one_m_eta**2)
        dPhi_dn2 = n1 / one_m_eta + term3_n2
        
        dPhi_dnv1 = -wd['nv2_z'] / one_m_eta
        term3_nv2 = -3*A*n2*wd['nv2_z'] / (12 * jnp.pi * one_m_eta**2)
        dPhi_dnv2 = -wd['nv1_z'] / one_m_eta + term3_nv2
        
        # c1 in Fourier space
        kernels = self.kernels
        c1_hat = -(
            jnp.fft.fft(dPhi_deta) * kernels.w3_hat + 
            jnp.fft.fft(dPhi_dn0) * kernels.w0_hat + 
            jnp.fft.fft(dPhi_dn1) * kernels.w1_hat + 
            jnp.fft.fft(dPhi_dn2) * kernels.w2_hat +
            jnp.fft.fft(dPhi_dnv1) * kernels.wv1_z_hat +
            jnp.fft.fft(dPhi_dnv2) * kernels.wv2_z_hat
        )
        
        return jnp.real(jnp.fft.ifft(c1_hat))


# ============================================================================
# WALL SOLVER
# ============================================================================

class WallSolver:
    """Solve for density profile at hard wall using Picard iteration."""
    
    def __init__(self, nz: int = 2048, Lz: float = 10.0, R: float = 0.5):
        self.grid = PlanarGrid(nz, Lz)
        self.R = R
        self.sigma = 2 * R
    
    def solve(self, eta: float, A: float = 1.0, B: float = 0.0,
              max_iter: int = 5000, tol: float = 1e-9,
              verbose: bool = True) -> Dict:
        """Solve for equilibrium density profile."""
        
        functional = LutskoFunctional(self.grid, self.sigma, A, B)
        rho_bulk = eta / ((4/3) * PI * self.R**3)
        
        # Initialize
        rho = jnp.ones(self.grid.nz) * rho_bulk
        rho = jnp.where(self.grid.z < self.R, 0.0, rho)
        
        # Bulk c1 reference
        rho_uniform = jnp.ones(self.grid.nz) * rho_bulk
        c1_uniform = functional.c1(rho_uniform)
        c1_bulk_ref = float(c1_uniform[self.grid.nz // 2])
        
        if verbose:
            print(f"  Solving η={eta:.3f}, A={A:.2f}, B={B:.2f}")
        
        # Picard iteration
        for i in range(max_iter):
            c1 = functional.c1(rho)
            
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
            
            diff = float(jnp.max(jnp.abs(rho_new - rho)) / rho_bulk)
            rho = alpha * rho_new + (1 - alpha) * rho
            rho = jnp.where(self.grid.z < self.R, 0.0, rho)
            
            if diff < tol:
                if verbose:
                    print(f"    Converged at iteration {i}")
                break
            
            if verbose and i % 500 == 0:
                contact_idx = int(jnp.argmin(jnp.abs(self.grid.z - (self.R + 0.01))))
                contact_norm = float(rho[contact_idx] / rho_bulk)
                print(f"    Iter {i}: contact = {contact_norm:.3f}, diff = {diff:.2e}")
        
        # Results
        z = np.array(self.grid.z)
        rho_norm = np.array(rho / rho_bulk)
        
        contact_idx = int(np.argmin(np.abs(z - (self.R + 0.01))))
        contact = rho_norm[contact_idx]
        
        return {
            'z': z,
            'rho_norm': rho_norm,
            'contact': contact,
            'eta': eta,
            'A': A,
            'B': B
        }


# ============================================================================
# DIRECT CORRELATION FUNCTIONS
# ============================================================================

def c_PY_real(r: np.ndarray, eta: float, sigma: float = 1.0) -> np.ndarray:
    """Percus-Yevick c(r) - EXACT for Rosenfeld FMT."""
    alpha = (1 + 2*eta)**2 / (1 - eta)**4
    beta = 6*eta * (1 + eta/2)**2 / (1 - eta)**4
    gamma = eta * (1 + 2*eta)**2 / (2*(1 - eta)**4)
    
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)


def c_fourier(r: np.ndarray, c_r: np.ndarray, k_max: float = 30.0, nk: int = 256) -> Tuple:
    """Compute ĉ(k) from c(r) via numerical FT."""
    k = np.linspace(0.01, k_max, nk)
    dr = r[1] - r[0]
    
    c_k = np.zeros_like(k)
    for i, ki in enumerate(k):
        sinc = np.sin(ki * r) / (ki * r + 1e-30)
        c_k[i] = 4*PI * np.sum(r**2 * c_r * sinc) * dr
    
    return k, c_k


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison():
    """Run comprehensive FMT comparison."""
    
    print("="*70)
    print("COMPREHENSIVE FMT COMPARISON")
    print("Density Profiles AND Direct Correlation Functions")
    print("="*70)
    
    eta = 0.367
    sigma = 1.0
    rho_bulk = 6 * eta / (PI * sigma**3)
    
    # MC reference
    MC_contact = 5.36  # ρ(R⁺)/ρ_bulk from MC
    CS_Z = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    
    print(f"\nPacking fraction η = {eta}")
    print(f"Monte Carlo contact: {MC_contact}")
    print(f"Carnahan-Starling Z: {CS_Z:.3f}")
    
    # =========================================================================
    # DEFINE FUNCTIONALS
    # =========================================================================
    
    functionals = {
        'Rosenfeld': {'A': 1.0, 'B': 0.0, 'color': 'C0'},
        'Lutsko': {'A': 1.0, 'B': 0.0, 'color': 'C1'},  # Same as Rosenfeld base
        'Gül et al.': {'A': 1.3, 'B': -1.0, 'color': 'C2'},
        'esFMT(1.5,0)': {'A': 1.5, 'B': 0.0, 'color': 'C3'},
        'esFMT(1,-1)': {'A': 1.0, 'B': -1.0, 'color': 'C4'},
    }
    
    # =========================================================================
    # PART 1: COMPUTE DENSITY PROFILES
    # =========================================================================
    print("\n" + "-"*70)
    print("PART 1: DENSITY PROFILES AT HARD WALL")
    print("-"*70)
    
    solver = WallSolver(nz=2048, Lz=10.0, R=0.5)
    
    profiles = {}
    contact_densities = {}
    
    for name, params in functionals.items():
        print(f"\nSolving for {name}...")
        result = solver.solve(eta, params['A'], params['B'], 
                             max_iter=5000, tol=1e-9, verbose=True)
        profiles[name] = result
        contact_densities[name] = result['contact']
        print(f"  Contact: {result['contact']:.3f}")
    
    # =========================================================================
    # PART 2: DIRECT CORRELATION FUNCTIONS  
    # =========================================================================
    print("\n" + "-"*70)
    print("PART 2: DIRECT CORRELATION FUNCTIONS")
    print("-"*70)
    
    r = np.linspace(0.001, 1.5*sigma, 512)
    
    # PY reference (= Rosenfeld c(r))
    c_r_PY = c_PY_real(r, eta, sigma)
    k_PY, c_k_PY = c_fourier(r, c_r_PY)
    
    print(f"  PY c(0) = {c_r_PY[0]:.4f}")
    
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
    colors['MC'] = 'red'
    
    # ----- (a) Density Profiles -----
    ax1 = fig.add_subplot(gs[0, 0])
    
    # MC data
    mc_data = MC_PROFILES[eta]
    mc_z = mc_data[:, 0]
    mc_rho_bulk = eta / ((4/3) * PI * 0.5**3)
    mc_rho_norm = mc_data[:, 1] / mc_rho_bulk
    
    ax1.plot(mc_z, mc_rho_norm, 'o', color='red', ms=3, mfc='white', mew=1, 
             alpha=0.8, label='MC (Davidchack et al.)')
    
    # FMT profiles
    for name, result in profiles.items():
        ax1.plot(result['z'], result['rho_norm'], '-', color=colors[name], 
                 lw=1.5, label=name)
    
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax1.axhline(MC_contact, color='red', ls=':', lw=2, label=f'MC contact = {MC_contact}')
    ax1.axvline(0.5, color='gray', ls='--', alpha=0.5, label='z = R')
    
    ax1.set_xlabel(r'$z/\sigma$', fontsize=12)
    ax1.set_ylabel(r'$\rho(z)/\rho_{\mathrm{bulk}}$', fontsize=12)
    ax1.set_title(f'(a) Density Profile at Hard Wall (η = {eta})', fontsize=12)
    ax1.set_xlim([0.4, 3])
    ax1.set_ylim([0, 8])
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ----- (b) Contact Density Bar Chart -----
    ax2 = fig.add_subplot(gs[0, 1])
    
    names = list(contact_densities.keys())
    contacts = [contact_densities[n] for n in names]
    x_pos = np.arange(len(names))
    
    bars = ax2.bar(x_pos, contacts, color=[colors[n] for n in names], alpha=0.8)
    ax2.axhline(MC_contact, color='red', ls='--', lw=2, label=f'MC = {MC_contact}')
    ax2.axhline(CS_Z, color='green', ls=':', lw=2, label=f'CS = {CS_Z:.2f}')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel(r'Contact Density $\rho(R^+)/\rho_{\mathrm{bulk}}$', fontsize=11)
    ax2.set_title('(b) Contact Density Comparison', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ----- (c) c(r) Real Space -----
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Rosenfeld = PY exactly
    ax3.plot(r, c_r_PY, '-', color='black', lw=2.5, label='PY (analytical)')
    ax3.plot(r, c_r_PY, '--', color=colors['Rosenfeld'], lw=1.5, label='Rosenfeld')
    
    ax3.axhline(0, color='gray', ls='-', alpha=0.3)
    ax3.axvline(1.0, color='gray', ls='--', alpha=0.5)
    
    ax3.set_xlabel(r'$r/\sigma$', fontsize=12)
    ax3.set_ylabel(r'$c(r)$', fontsize=12)
    ax3.set_title(f'(c) Direct Correlation Function c(r) (η = {eta})', fontsize=12)
    ax3.set_xlim([0, 1.2])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ----- (d) ĉ(k) Fourier Space -----
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(k_PY, c_k_PY, '-', color='black', lw=2.5, label='PY (analytical)')
    
    ax4.axhline(0, color='gray', ls='-', alpha=0.3)
    
    ax4.set_xlabel(r'$k\sigma$', fontsize=12)
    ax4.set_ylabel(r'$\hat{c}(k)$', fontsize=12)
    ax4.set_title(f'(d) Fourier Transform ĉ(k) (η = {eta})', fontsize=12)
    ax4.set_xlim([0, 20])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('FMT Comparison: Density Profiles and Direct Correlation Functions', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('outputs/fmt_validated_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("Saved: fmt_validated_comparison.png")
    plt.close()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Functional':<15} {'Contact':>10} {'% of MC':>10} {'% of CS':>10}")
    print("-"*50)
    for name, contact in contact_densities.items():
        pct_MC = contact / MC_contact * 100
        pct_CS = contact / CS_Z * 100
        print(f"{name:<15} {contact:10.3f} {pct_MC:9.1f}% {pct_CS:9.1f}%")
    
    print("="*70)


if __name__ == "__main__":
    run_comparison()
