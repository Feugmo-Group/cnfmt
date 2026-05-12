#!/usr/bin/env python3
"""
LJ Phase Diagram - Lutsko Method + Neural Network
==================================================

Based on working Lutsko implementation with proper:
1. WCA decomposition of LJ potential
2. Numerical integration for Barker-Henderson diameter (Eq. 38)
3. Numerical integration for van der Waals parameter (Eq. 41)
4. Thermodynamic relation P = ρμ - f

Reference: Lutsko & Lam, Phys. Rev. E
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from paper_figure_style import apply_paper_style
apply_paper_style()
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
PI = np.pi


# =============================================================================
# LJ POTENTIAL WITH WCA DECOMPOSITION (Eq. 35-37)
# =============================================================================

class LJPotential:
    """LJ potential with WCA decomposition following Lutsko."""
    
    def __init__(self, sigma=1.0, epsilon=1.0, r_cut=3.0):
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_cut = r_cut * sigma
        self.r_min = 2**(1/6) * sigma  # Potential minimum
        
        self.v_at_rmin = self._v_lj_raw(self.r_min)
        self.v_at_rcut = self._v_lj_raw(self.r_cut)
        self.v_shifted_at_rmin = self.v_at_rmin - self.v_at_rcut
    
    def _v_lj_raw(self, r):
        if r < 0.1 * self.sigma:
            return 1e10
        y = self.sigma / r
        y6 = y**6
        return 4 * self.epsilon * (y6**2 - y6)
    
    def v_shifted(self, r):
        if r >= self.r_cut:
            return 0.0
        return self._v_lj_raw(r) - self.v_at_rcut
    
    def v_repulsive(self, r):
        """WCA repulsive part for BH integral."""
        if r >= self.r_min:
            return 0.0
        return self.v_shifted(r) - self.v_shifted_at_rmin
    
    def w_attractive(self, r):
        """WCA attractive part."""
        if r >= self.r_cut:
            return 0.0
        if r < self.r_min:
            return self.v_shifted_at_rmin
        return self.v_shifted(r)
    
    def barker_henderson_diameter(self, kT):
        """BH diameter via numerical integration (Eq. 38)."""
        def integrand(r):
            if r < 0.01 * self.sigma:
                return 1.0
            v0 = self.v_repulsive(r)
            beta_v0 = v0 / kT
            if beta_v0 > 100:
                return 1.0
            return 1.0 - np.exp(-beta_v0)
        
        d_T, _ = quad(integrand, 0.01 * self.sigma, self.r_min, limit=200)
        d_T += 0.01 * self.sigma
        return d_T
    
    def vdw_parameter(self):
        """van der Waals parameter via numerical integration (Eq. 41)."""
        def integrand(r):
            return 4 * PI * r**2 * self.w_attractive(r)
        a, _ = quad(integrand, 0.01 * self.sigma, self.r_cut, limit=200)
        return a


# =============================================================================
# HARD-SPHERE FUNCTIONS (CS + Lutsko correction)
# =============================================================================

def f_ex_CS(eta):
    """Carnahan-Starling excess free energy per particle (Eq. 43)."""
    if eta <= 0:
        return 0.0
    if eta >= 0.64:
        eta = 0.64
    return eta * (4 - 3*eta) / (1 - eta)**2


def mu_ex_CS(eta):
    """CS excess chemical potential."""
    if eta <= 0:
        return 0.0
    if eta >= 0.64:
        eta = 0.64
    return eta * (8 - 9*eta + 3*eta**2) / (1 - eta)**3


def f_ex_lutsko(eta, A=1.0, B=0.0):
    """Lutsko excess free energy with correction."""
    if eta <= 0:
        return 0.0
    eta = min(eta, 0.64)
    Delta = 8*A + 2*B - 9
    return f_ex_CS(eta) + Delta * eta**2 / (6 * (1 - eta)**2)


def mu_ex_lutsko(eta, A=1.0, B=0.0):
    """Lutsko excess chemical potential."""
    if eta <= 0:
        return 0.0
    eta = min(eta, 0.64)
    Delta = 8*A + 2*B - 9
    return mu_ex_CS(eta) + Delta * eta * (2 - eta) / (6 * (1 - eta)**3)


# =============================================================================
# UNIFORM LJ FLUID (Eq. 40-44)
# =============================================================================

class UniformLJFluid:
    """Thermodynamics of uniform LJ fluid."""
    
    def __init__(self, potential, T_star, A=1.0, B=0.0, nn=None):
        self.potential = potential
        self.T_star = T_star
        self.kT = T_star * potential.epsilon
        self.sigma = potential.sigma
        
        # Temperature-dependent BH diameter
        self.d_T = potential.barker_henderson_diameter(self.kT)
        
        # van der Waals parameter (temperature-independent)
        self.a = potential.vdw_parameter()
        
        # Lutsko parameters
        self.A_fixed = A
        self.B_fixed = B
        self.nn = nn
    
    def get_AB(self, eta):
        if self.nn is not None:
            A, B = self.nn.from_eta(float(eta))
            return float(A), float(B)
        return self.A_fixed, self.B_fixed
    
    def eta(self, rho):
        """Packing fraction η = (π/6) ρ d_T³"""
        return (PI / 6) * rho * self.d_T**3
    
    def f_total(self, rho):
        """Total free energy density βf (Eq. 40)."""
        if rho <= 0:
            return 0.0
        eta = self.eta(rho)
        A, B = self.get_AB(eta)
        
        # Ideal: ρ[ln(ρσ³) - 1]
        f_id = rho * (np.log(rho * self.sigma**3) - 1)
        # HS excess
        f_ex = rho * f_ex_lutsko(eta, A, B)
        # Attractive mean-field
        f_att = 0.5 * self.a * rho**2 / self.kT
        return f_id + f_ex + f_att
    
    def mu_total(self, rho):
        """Chemical potential βμ (Eq. 44)."""
        if rho <= 0:
            return -1e10
        eta = self.eta(rho)
        A, B = self.get_AB(eta)
        
        mu_id = np.log(rho * self.sigma**3)
        mu_ex = mu_ex_lutsko(eta, A, B)
        mu_att = self.a * rho / self.kT
        return mu_id + mu_ex + mu_att
    
    def P_total(self, rho):
        """Pressure βP from thermodynamic relation P = ρμ - f."""
        if rho <= 0:
            return 0.0
        return rho * self.mu_total(rho) - self.f_total(rho)


# =============================================================================
# PHASE DIAGRAM
# =============================================================================

class LJPhaseDiagram:
    """Compute vapor-liquid coexistence."""
    
    def __init__(self, sigma=1.0, epsilon=1.0, r_cut=3.0, A=1.0, B=0.0, nn=None):
        self.potential = LJPotential(sigma, epsilon, r_cut)
        self.sigma = sigma
        self.epsilon = epsilon
        self.A = A
        self.B = B
        self.nn = nn
    
    def find_coexistence(self, T_star, rho_v_guess=0.02, rho_l_guess=0.6):
        fluid = UniformLJFluid(self.potential, T_star, self.A, self.B, self.nn)
        
        def equations(x):
            rho_v, rho_l = x
            if rho_v <= 0.001 or rho_l <= 0.001 or rho_v >= rho_l:
                return [1e10, 1e10]
            if fluid.eta(rho_l) > 0.6:
                return [1e10, 1e10]
            
            return [fluid.mu_total(rho_v) - fluid.mu_total(rho_l),
                    fluid.P_total(rho_v) - fluid.P_total(rho_l)]
        
        try:
            sol, info, ier, msg = fsolve(equations, [rho_v_guess, rho_l_guess], full_output=True)
            if ier == 1:
                rho_v, rho_l = sol
                if 0 < rho_v < rho_l < 1.0:
                    residual = np.max(np.abs(equations([rho_v, rho_l])))
                    if residual < 0.01:
                        return float(rho_v), float(rho_l)
        except:
            pass
        return None, None
    
    def find_spinodal(self, T_star):
        """Find spinodal points where ∂P/∂ρ = 0."""
        fluid = UniformLJFluid(self.potential, T_star, self.A, self.B, self.nn)
        rho_vals = np.linspace(0.01, 0.8, 500)
        P_vals = np.array([fluid.P_total(r) for r in rho_vals])
        dP = np.diff(P_vals)
        sign_changes = np.where(dP[:-1] * dP[1:] < 0)[0]
        if len(sign_changes) >= 2:
            return rho_vals[sign_changes[0]+1], rho_vals[sign_changes[-1]+1]
        return None, None
    
    def compute_coexistence_curve(self, T_min=0.7, T_max=1.35, n_points=50):
        T_vals = np.linspace(T_min, T_max, n_points)
        rho_v_list, rho_l_list, T_valid = [], [], []
        
        rv_guess, rl_guess = 0.01, 0.65
        
        for T in T_vals:
            rv, rl = self.find_coexistence(T, rv_guess, rl_guess)
            if rv is not None:
                rho_v_list.append(rv)
                rho_l_list.append(rl)
                T_valid.append(T)
                rv_guess, rl_guess = rv, rl
            else:
                # Try spinodal-based guess
                sp_lo, sp_hi = self.find_spinodal(T)
                if sp_lo is not None:
                    rv, rl = self.find_coexistence(T, sp_lo * 0.5, sp_hi * 1.2)
                    if rv is not None:
                        rho_v_list.append(rv)
                        rho_l_list.append(rl)
                        T_valid.append(T)
                        rv_guess, rl_guess = rv, rl
        
        return {'T': np.array(T_valid), 'rho_v': np.array(rho_v_list), 'rho_l': np.array(rho_l_list)}


# =============================================================================
# NEURAL NETWORK
# =============================================================================

from neural.network import ConditionalNetwork


def train_nn(network, potential, ref_coex, n_epochs=300, lr=0.01):
    """Train NN to match reference coexistence."""
    print("\nTraining NN A(η, T*), B(η, T*)...")
    
    T_train = ref_coex['T'][::3]
    rv_train = ref_coex['rho_v'][::3]
    rl_train = ref_coex['rho_l'][::3]
    print(f"  Training on {len(T_train)} points")
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    
    # Precompute temperature-dependent quantities
    sigma = potential.sigma
    a = potential.vdw_parameter()
    
    cache = {}
    for T in T_train:
        kT = T * potential.epsilon
        d_T = potential.barker_henderson_diameter(kT)
        cache[T] = (d_T, kT)
    
    def loss_fn(net, T, rv, rl):
        d_T, kT = cache[T]
        
        eta_v = (PI / 6) * rv * d_T**3
        eta_l = (PI / 6) * rl * d_T**3
        
        A_v, B_v = net.from_eta(eta_v)
        A_l, B_l = net.from_eta(eta_l)
        
        mu_v = jnp.log(rv * sigma**3) + mu_ex_lutsko(eta_v, A_v, B_v) + a * rv / kT
        mu_l = jnp.log(rl * sigma**3) + mu_ex_lutsko(eta_l, A_l, B_l) + a * rl / kT
        
        f_v = rv * (jnp.log(rv * sigma**3) - 1) + rv * f_ex_lutsko(eta_v, A_v, B_v) + 0.5 * a * rv**2 / kT
        f_l = rl * (jnp.log(rl * sigma**3) - 1) + rl * f_ex_lutsko(eta_l, A_l, B_l) + 0.5 * a * rl**2 / kT
        P_v = rv * mu_v - f_v
        P_l = rl * mu_l - f_l
        
        return (mu_v - mu_l)**2 + ((P_v - P_l)/(jnp.abs(P_v) + 0.01))**2
    
    @eqx.filter_value_and_grad
    def total_loss(net):
        return sum(loss_fn(net, T, rv, rl) for T, rv, rl in zip(T_train, rv_train, rl_train)) / len(T_train)
    
    losses = []
    for epoch in range(n_epochs):
        lv, gr = total_loss(network)
        up, opt_state = optimizer.update(eqx.filter(gr, eqx.is_array), opt_state)
        network = eqx.apply_updates(network, up)
        losses.append(float(lv))
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: loss = {losses[-1]:.2e}")
    
    print(f"    Final: {losses[-1]:.2e}")
    return network, losses


# =============================================================================
# PLOTTING
# =============================================================================

def plot_lutsko(potential, coex):
    """4-panel traditional Lutsko plot."""
    print("\nPlotting Lutsko phase diagram...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (a) Coexistence
    ax = axes[0, 0]
    ax.plot(coex['rho_v'], coex['T'], 'bo-', ms=4, lw=2, label='Vapor')
    ax.plot(coex['rho_l'], coex['T'], 'ro-', ms=4, lw=2, label='Liquid')
    rho_avg = (coex['rho_v'] + coex['rho_l']) / 2
    ax.plot(rho_avg, coex['T'], 'g--', lw=1.5, alpha=0.7, label='Diameter')
    
    if len(coex['T']) > 0:
        T_c = coex['T'][-1]
        rho_c = (coex['rho_v'][-1] + coex['rho_l'][-1]) / 2
        ax.scatter([rho_c], [T_c], s=100, c='purple', marker='*', zorder=10)
        ax.axhline(T_c, color='purple', ls=':', alpha=0.7, label=f'T*_c ≈ {T_c:.2f}')
    
    ax.axhline(1.28, color='gray', ls='--', alpha=0.5, label='Lutsko T*_c ~ 1.28')
    ax.set_xlabel('ρσ³', fontsize=14)
    ax.set_ylabel('T* = kT/ε', fontsize=14)
    ax.set_title('(a) Vapor-Liquid Coexistence', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim([0, 0.9])
    ax.set_ylim([0.6, 1.5])
    ax.grid(True, alpha=0.3)
    
    # (b) Pressure isotherms
    ax = axes[0, 1]
    T_list = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(T_list)))
    rho_range = np.linspace(0.001, 0.75, 300)
    
    for T, c in zip(T_list, colors):
        fluid = UniformLJFluid(potential, T)
        P = [fluid.P_total(r) for r in rho_range]
        ax.plot(rho_range, P, color=c, lw=1.5, label=f'T*={T}')
    
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('ρσ³', fontsize=14)
    ax.set_ylabel('βPσ³', fontsize=14)
    ax.set_title('(b) Pressure Isotherms', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.set_xlim([0, 0.75])
    ax.set_ylim([-0.3, 2])
    ax.grid(True, alpha=0.3)
    
    # (c) Chemical potential
    ax = axes[1, 0]
    for T, c in zip(T_list, colors):
        fluid = UniformLJFluid(potential, T)
        rho_range2 = np.linspace(0.01, 0.75, 300)
        mu = [fluid.mu_total(r) for r in rho_range2]
        ax.plot(rho_range2, mu, color=c, lw=1.5, label=f'T*={T}')
    
    ax.set_xlabel('ρσ³', fontsize=14)
    ax.set_ylabel('βμ', fontsize=14)
    ax.set_title('(c) Chemical Potential Isotherms', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.set_xlim([0, 0.75])
    ax.grid(True, alpha=0.3)
    
    # (d) Free energy at T*=0.9
    ax = axes[1, 1]
    T = 0.9
    fluid = UniformLJFluid(potential, T)
    rho_plot = np.linspace(0.01, 0.7, 300)
    f = [fluid.f_total(r) for r in rho_plot]
    ax.plot(rho_plot, f, 'b-', lw=2, label=f'f(ρ) at T*={T}')
    
    # Common tangent
    pd = LJPhaseDiagram()
    rv, rl = pd.find_coexistence(T)
    if rv is not None:
        mu_coex = fluid.mu_total(rv)
        fv, fl = fluid.f_total(rv), fluid.f_total(rl)
        rho_tan = np.linspace(rv, rl, 100)
        f_tan = fv + mu_coex * (rho_tan - rv)
        ax.plot(rho_tan, f_tan, 'r--', lw=2, label='Common tangent')
        ax.scatter([rv, rl], [fv, fl], c='red', s=80, zorder=10)
        ax.axvline(rv, color='red', ls=':', alpha=0.5)
        ax.axvline(rl, color='red', ls=':', alpha=0.5)
    
    ax.set_xlabel('ρσ³', fontsize=14)
    ax.set_ylabel('βfσ³', fontsize=14)
    ax.set_title(f'(d) Free Energy Density at T*={T}', fontsize=14)
    ax.legend(fontsize=9)
    ax.set_xlim([0, 0.7])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Lennard-Jones Fluid Phase Behavior (r_c = 3σ)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lj_phase_diagram_lutsko.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: lj_phase_diagram_lutsko.png")
    plt.close()


def plot_nn(network, losses, potential, ref_coex):
    """6-panel NN results."""
    print("\nPlotting NN results...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # (a) Training
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(losses, 'b-', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Training', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # (b) Phase diagram
    ax = fig.add_subplot(gs[0, 1])
    
    print("  Computing NN coexistence...")
    pd_nn = LJPhaseDiagram(nn=network)
    coex_nn = pd_nn.compute_coexistence_curve(T_min=0.7, T_max=1.35, n_points=50)
    print(f"  Found {len(coex_nn['T'])} NN points")
    
    if len(coex_nn['T']) > 0:
        ax.plot(coex_nn['rho_v'], coex_nn['T'], 'b-', lw=2.5, label='NN vapor')
        ax.plot(coex_nn['rho_l'], coex_nn['T'], 'b--', lw=2.5, label='NN liquid')
        T_c = coex_nn['T'][-1]
        ax.axhline(T_c, color='g', ls='--', lw=1.5, label=f'T*_c ~ {T_c:.2f}')
    
    ax.plot(ref_coex['rho_v'], ref_coex['T'], 'r:', lw=2, label='Lutsko')
    ax.plot(ref_coex['rho_l'], ref_coex['T'], 'r:', lw=2)
    
    ax.set_xlabel('ρσ³')
    ax.set_ylabel('T*')
    ax.set_title('(b) Phase Diagram', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 0.9])
    ax.set_ylim([0.6, 1.5])
    ax.grid(True, alpha=0.3)
    
    # (c-f) Learned parameters
    eta_range = np.linspace(0.01, 0.5, 100)
    
    ax = fig.add_subplot(gs[0, 2])
    A_vals = [float(network.from_eta(e)[0]) for e in eta_range]
    ax.plot(eta_range, A_vals, color='blue', lw=2, label='A(η)')
    ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='Lutsko A=1')
    ax.set_xlabel('η')
    ax.set_ylabel('A(η)')
    ax.set_title('(c) Learned A', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    B_vals = [float(network.from_eta(e)[1]) for e in eta_range]
    ax.plot(eta_range, B_vals, color='blue', lw=2, label='B(η)')
    ax.axhline(0.0, color='gray', ls='--', lw=1.5, label='Lutsko B=0')
    ax.set_xlabel('η')
    ax.set_ylabel('B(η)')
    ax.set_title('(d) Learned B', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    D = [8*float(network.from_eta(e)[0]) + 2*float(network.from_eta(e)[1]) - 9 for e in eta_range]
    ax.plot(eta_range, D, color='blue', lw=2, label='Δ(η)')
    ax.axhline(0, color='gray', ls='--', lw=1.5, label='PY (Δ=0)')
    ax.axhline(-1, color='orange', ls=':', lw=1.5, label='Lutsko (Δ=-1)')
    ax.set_xlabel('η')
    ax.set_ylabel('Δ = 8A + 2B - 9')
    ax.set_title('(e) EOS Correction', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    eta_g = np.linspace(0.01, 0.5, 50)
    A_g = np.array([float(network.from_eta(e)[0]) for e in eta_g])
    ax.plot(eta_g, A_g, 'b-', lw=2)
    ax.set_xlabel('η')
    ax.set_ylabel('A(η)')
    ax.set_title('(f) A(η) Profile', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Neural Network A(η, T*), B(η, T*)', fontsize=16, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'lj_phase_diagram_nn.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: lj_phase_diagram_nn.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("LJ PHASE DIAGRAM (Lutsko Method)")
    print("="*70)
    
    # Create potential
    potential = LJPotential(sigma=1.0, epsilon=1.0, r_cut=3.0)
    
    print(f"\nPotential parameters:")
    print(f"  r_min = {potential.r_min:.4f}σ")
    print(f"  v(r_min) = {potential.v_at_rmin:.4f}ε")
    print(f"  van der Waals a = {potential.vdw_parameter():.4f} ε σ³")
    print(f"  BH diameter at T*=1: d_T = {potential.barker_henderson_diameter(1.0):.4f}σ")
    
    # Traditional Lutsko (A=1, B=0)
    print("\n--- Traditional Lutsko (A=1, B=0) ---")
    pd = LJPhaseDiagram()
    coex = pd.compute_coexistence_curve(T_min=0.7, T_max=1.35, n_points=50)
    print(f"Found {len(coex['T'])} coexistence points")
    if len(coex['T']) > 0:
        print(f"T*_c ≈ {coex['T'][-1]:.3f}")
    
    plot_lutsko(potential, coex)
    
    # NN training
    print("\n--- Neural Network ---")
    network = ConditionalNetwork(jax.random.PRNGKey(42))
    network, losses = train_nn(network, potential, coex, n_epochs=300, lr=0.01)
    plot_nn(network, losses, potential, coex)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
