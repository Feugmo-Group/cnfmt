"""Quick comprehensive FMT figure with all c(r) methods."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from core.thermodynamics import BulkThermodynamics as BT

import sys
sys.path.insert(0, '/mnt/user-data/uploads')
from fmt_1d_wbii_tensor import (
    WallSolver, RosenfeldFMT, WhiteBearIIFMT, ModifiedRSLT, esFMT_Tensor,
    phi2_WBII, phi3_WBII
)

PI = np.pi

# c(r) functions
def c_PY_real(r, eta, sigma=1.0):
    alpha = (1 + 2*eta)**2 / (1 - eta)**4
    beta = 6*eta * (1 + eta/2)**2 / (1 - eta)**4
    gamma = eta * (1 + 2*eta)**2 / (2*(1 - eta)**4)
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)

def c_WBII_real(r, eta, sigma=1.0):
    phi2 = float(phi2_WBII(jnp.array(eta)))
    phi3 = float(phi3_WBII(jnp.array(eta)))
    alpha = (1 + 2*eta)**2 / (1 - eta)**4 * phi2
    beta = 6*eta * (1 + eta/2)**2 / (1 - eta)**4 * phi2
    gamma = eta * (1 + 2*eta)**2 / (2*(1 - eta)**4) * phi3
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)

def c_mRSLT_real(r, eta, sigma=1.0):
    phi2 = float(phi2_WBII(jnp.array(eta)))
    alpha = (1 + 2*eta)**2 / (1 - eta)**4
    beta = 6*eta * (1 + eta/2)**2 / (1 - eta)**4
    gamma = eta * (1 + 2*eta)**2 / (2*(1 - eta)**4) * phi2
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)

def c_esFMT_real(r, eta, A, B, sigma=1.0):
    delta = 8*A + 2*B - 9
    alpha = (1 + 2*eta)**2 / (1 - eta)**4 + delta * eta**2 / (6*(1 - eta)**4)
    beta = 6*eta * (1 + eta/2)**2 / (1 - eta)**4
    gamma = eta * (1 + 2*eta)**2 / (2*(1 - eta)**4) * (1 + delta/9)
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)

def c_fourier(r, c_r, k_max=30.0, nk=256):
    k = np.linspace(0.01, k_max, nk)
    dr = r[1] - r[0]
    c_k = np.zeros_like(k)
    for i, ki in enumerate(k):
        sinc = np.sin(ki * r) / (ki * r + 1e-30)
        c_k[i] = 4*PI * np.sum(r**2 * c_r * sinc) * dr
    return k, c_k

# MC data
MC_DATA = np.array([
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
])

print("="*60)
print("COMPREHENSIVE FMT COMPARISON - ALL METHODS")
print("="*60)

eta = 0.367
sigma = 1.0
rho_bulk = 6 * eta / PI
MC_contact = 5.36
CS_Z = float(BT.Z_CS(eta))

# Solve density profiles
solver = WallSolver(nz=1024, Lz=6.0, R=0.5)

functionals = [
    ('Rosenfeld', RosenfeldFMT()),
    ('White Bear II', WhiteBearIIFMT()),
    ('Modified RSLT', ModifiedRSLT()),
    ('esFMT(1,-1)', esFMT_Tensor(A=1.0, B=-1.0)),
    ('Gül et al.', esFMT_Tensor(A=1.3, B=-1.0)),
]

colors = {'Rosenfeld': 'C0', 'White Bear II': 'C1', 'Modified RSLT': 'C2',
          'esFMT(1,-1)': 'C3', 'Gül et al.': 'C4'}

profiles = {}
print(f"\nComputing density profiles at η = {eta}...")
for name, func in functionals:
    result = solver.solve(eta, func, max_iter=2500, tol=1e-7, verbose=False)
    profiles[name] = result
    print(f"  {name}: contact = {result['contact']:.3f}")

# MC normalization
mc_norm = MC_DATA.copy()
mc_norm[:, 1] = MC_DATA[:, 1] / rho_bulk

# Compute c(r) for all methods
print("\nComputing c(r) for all methods...")
r = np.linspace(0.001, 1.5*sigma, 512)

c_r_all = {
    'Rosenfeld': c_PY_real(r, eta, sigma),
    'White Bear II': c_WBII_real(r, eta, sigma),
    'Modified RSLT': c_mRSLT_real(r, eta, sigma),
    'esFMT(1,-1)': c_esFMT_real(r, eta, 1.0, -1.0, sigma),
    'Gül et al.': c_esFMT_real(r, eta, 1.3, -1.0, sigma),
}

c_k_all = {}
for name, c_r in c_r_all.items():
    k, c_k = c_fourier(r, c_r)
    c_k_all[name] = (k, c_k)
    print(f"  {name}: c(0) = {c_r[0]:.4f}")

# Create figure
print("\nCreating figure...")
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

# (a) Density Profiles
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(mc_norm[:, 0], mc_norm[:, 1], 'o', color='red', ms=4, 
         mfc='white', mew=1.5, alpha=0.8, label='MC')
for name, result in profiles.items():
    ax1.plot(result['z'], result['rho_norm'], '-', color=colors[name], 
             lw=1.5, label=f"{name}: {result['contact']:.2f}")
ax1.axhline(1.0, color='gray', ls='--', alpha=0.5)
ax1.axhline(MC_contact, color='red', ls=':', lw=2, alpha=0.7)
ax1.axvline(0.5, color='gray', ls='--', alpha=0.5)
ax1.set_xlabel(r'$z/\sigma$', fontsize=12)
ax1.set_ylabel(r'$\rho(z)/\rho_{\mathrm{bulk}}$', fontsize=12)
ax1.set_title(f'(a) Density Profile at Hard Wall (η = {eta})', fontsize=12)
ax1.set_xlim([0.4, 2.0])
ax1.set_ylim([0, 8])
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# (b) Contact Density Bar Chart
ax2 = fig.add_subplot(gs[0, 1])
names = list(profiles.keys())
contacts = [profiles[n]['contact'] for n in names]
x_pos = np.arange(len(names))
ax2.bar(x_pos, contacts, color=[colors[n] for n in names], alpha=0.8)
ax2.axhline(MC_contact, color='red', ls='--', lw=2, label=f'MC = {MC_contact}')
ax2.axhline(CS_Z, color='green', ls=':', lw=2, label=f'CS = {CS_Z:.2f}')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel(r'Contact Density $\rho(R^+)/\rho_{\mathrm{bulk}}$', fontsize=11)
ax2.set_title('(b) Contact Density Comparison', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# (c) c(r) Real Space - ALL METHODS
ax3 = fig.add_subplot(gs[1, 0])
for name, c_r in c_r_all.items():
    ax3.plot(r, c_r, '-', color=colors[name], lw=1.5, label=name)
ax3.axhline(0, color='gray', ls='-', alpha=0.3)
ax3.axvline(1.0, color='gray', ls='--', alpha=0.5)
ax3.set_xlabel(r'$r/\sigma$', fontsize=12)
ax3.set_ylabel(r'$c(r)$', fontsize=12)
ax3.set_title(f'(c) Direct Correlation Function c(r) (η = {eta})', fontsize=12)
ax3.set_xlim([0, 1.2])
ax3.legend(fontsize=9, loc='lower right')
ax3.grid(True, alpha=0.3)

# (d) ĉ(k) Fourier Space - ALL METHODS
ax4 = fig.add_subplot(gs[1, 1])
for name, (k, c_k) in c_k_all.items():
    ax4.plot(k, c_k, '-', color=colors[name], lw=1.5, label=name)
ax4.axhline(0, color='gray', ls='-', alpha=0.3)
ax4.set_xlabel(r'$k\sigma$', fontsize=12)
ax4.set_ylabel(r'$\hat{c}(k)$', fontsize=12)
ax4.set_title(f'(d) Fourier Transform ĉ(k) (η = {eta})', fontsize=12)
ax4.set_xlim([0, 20])
ax4.legend(fontsize=9, loc='lower right')
ax4.grid(True, alpha=0.3)

plt.suptitle('FMT Comparison: Density Profiles and Direct Correlation Functions (All Methods)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig('outputs/fmt_comprehensive_all_methods.png', 
            dpi=150, bbox_inches='tight')
print("Saved: fmt_comprehensive_all_methods.png")
plt.close()

print("\nDone!")
