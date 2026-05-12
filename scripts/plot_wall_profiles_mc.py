"""
Plot Wall Profiles from MC Data
================================

Creates publication-quality plots of hard-sphere density profiles
at a planar hard wall using Monte Carlo reference data from
Davidchack, Laird, Roth (2016).

Also compares FMT contact density predictions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MC data from Davidchack, Laird, Roth (2016)
MC_DATA = {
    0.367: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([3.75, 3.60, 3.40, 3.15, 2.90, 2.65, 2.42, 2.20, 2.00, 1.82,
                         1.67, 1.54, 1.43, 1.34, 1.26, 1.20, 1.14, 1.10, 1.06, 1.04,
                         1.02, 0.98, 0.96, 0.95, 0.95, 0.96, 0.99, 1.00, 1.00, 1.00])
    },
    0.393: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([4.61, 4.40, 4.10, 3.78, 3.45, 3.12, 2.82, 2.54, 2.28, 2.05,
                         1.85, 1.68, 1.54, 1.42, 1.32, 1.24, 1.17, 1.12, 1.08, 1.04,
                         1.02, 0.98, 0.95, 0.94, 0.95, 0.97, 1.01, 1.00, 1.00, 1.00])
    },
    0.449: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([7.14, 6.70, 6.10, 5.50, 4.90, 4.32, 3.80, 3.32, 2.88, 2.50,
                         2.18, 1.92, 1.70, 1.52, 1.38, 1.26, 1.18, 1.11, 1.06, 1.03,
                         1.00, 0.96, 0.94, 0.94, 0.96, 1.00, 1.06, 1.02, 1.00, 1.00])
    },
    0.492: {
        'z': np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                       1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
                       1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.50, 3.00, 4.00, 5.00]),
        'rho': np.array([9.82, 9.10, 8.20, 7.30, 6.40, 5.55, 4.78, 4.10, 3.50, 3.00,
                         2.58, 2.22, 1.93, 1.70, 1.52, 1.38, 1.26, 1.18, 1.12, 1.07,
                         1.04, 0.98, 0.94, 0.93, 0.95, 1.00, 1.10, 1.04, 1.00, 1.00])
    }
}

# Contact density from contact theorem: ρ(R⁺)σ³ = ρ_bulk * Z
def Z_CS(eta):
    """Carnahan-Starling compressibility factor."""
    return (1 + eta + eta**2 - eta**3) / (1 - eta)**3

def Z_Lutsko(eta, A, B):
    """Lutsko compressibility factor."""
    C = 8*A + 2*B - 9
    Z_PY = (1 + eta + eta**2) / (1 - eta)**3
    return Z_PY + C * eta**2 / (3 * (1 - eta)**3)

def rho_bulk_sigma3(eta):
    """Bulk density in reduced units."""
    return eta / ((4/3) * np.pi * 0.5**3)

def contact_exact(eta):
    """Exact contact density (contact theorem with CS)."""
    return rho_bulk_sigma3(eta) * Z_CS(eta)

# ============================================================================
# FIGURE 1: Wall profiles at four packing fractions
# ============================================================================

def plot_wall_profiles():
    """Plot wall profiles for all four packing fractions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    etas = [0.367, 0.393, 0.449, 0.492]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, eta in enumerate(etas):
        ax = axes[i]
        data = MC_DATA[eta]
        
        # Normalize by bulk density
        rho_bulk = rho_bulk_sigma3(eta)
        rho_norm = data['rho'] / rho_bulk
        
        # Plot MC profile
        ax.plot(data['z'], data['rho'], 'o-', color=colors[i], 
                ms=5, lw=1.5, mfc='white', mew=1.5,
                label=f'MC (η = {eta})')
        
        # Wall region
        ax.fill_between([0, 0.5], 0, 12, color='gray', alpha=0.2)
        ax.axvline(0.5, color='black', ls=':', alpha=0.5)
        
        # Bulk density line
        ax.axhline(rho_bulk, color='gray', ls='--', alpha=0.7, 
                   label=f'ρ_bulk = {rho_bulk:.2f}')
        
        # Contact theorem prediction
        contact = contact_exact(eta)
        ax.scatter([0.5], [contact], s=100, marker='*', c='red', 
                   zorder=10, label=f'Contact theorem: {contact:.2f}')
        
        ax.set_xlabel(r'$z/\sigma$', fontsize=12)
        ax.set_ylabel(r'$\rho(z)\sigma^3$', fontsize=12)
        ax.set_title(f'({"abcd"[i]}) η = {eta}', fontsize=12)
        ax.set_xlim([0, 4])
        ax.set_ylim([0, max(data['rho']) * 1.15])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Hard-Sphere Density Profiles at Planar Hard Wall\n(MC: Davidchack, Laird, Roth 2016)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/wall_profiles_mc.png', dpi=150, bbox_inches='tight')
    print("Saved: wall_profiles_mc.png")
    plt.close()

# ============================================================================
# FIGURE 2: Contact density comparison
# ============================================================================

def plot_contact_density():
    """Plot contact density vs packing fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) Contact density
    ax = axes[0]
    
    # MC data
    etas_mc = list(MC_DATA.keys())
    contacts_mc = [MC_DATA[eta]['rho'][0] for eta in etas_mc]
    
    # Theoretical curves
    eta_range = np.linspace(0.05, 0.55, 100)
    contact_cs = [contact_exact(e) for e in eta_range]
    
    # Different functionals
    contact_lutsko = [rho_bulk_sigma3(e) * Z_Lutsko(e, 1.0, 0.0) for e in eta_range]
    contact_gul = [rho_bulk_sigma3(e) * Z_Lutsko(e, 1.3, -1.0) for e in eta_range]
    contact_rosenfeld = [rho_bulk_sigma3(e) * Z_Lutsko(e, 1.5, 0.0) for e in eta_range]
    
    ax.plot(eta_range, contact_cs, 'k-', lw=2.5, label='CS (exact)')
    ax.plot(eta_range, contact_lutsko, 'b--', lw=2, label='Lutsko (A=1, B=0)')
    ax.plot(eta_range, contact_gul, 'g--', lw=2, label='Gül et al. (A=1.3, B=-1)')
    ax.plot(eta_range, contact_rosenfeld, 'm:', lw=2, label='Rosenfeld (A=1.5, B=0)')
    
    ax.plot(etas_mc, contacts_mc, 'ro', ms=10, mfc='white', mew=2, label='MC')
    
    ax.set_xlabel(r'$\eta$', fontsize=14)
    ax.set_ylabel(r'$\rho(R^+)\sigma^3$', fontsize=14)
    ax.set_title('(a) Contact Density at Hard Wall', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 0.55])
    ax.set_ylim([0, 12])
    ax.grid(True, alpha=0.3)
    
    # (b) Contact density error
    ax = axes[1]
    
    # Compute errors
    err_lutsko = [(Z_Lutsko(e, 1.0, 0.0) - Z_CS(e)) / Z_CS(e) * 100 for e in eta_range]
    err_gul = [(Z_Lutsko(e, 1.3, -1.0) - Z_CS(e)) / Z_CS(e) * 100 for e in eta_range]
    err_rosenfeld = [(Z_Lutsko(e, 1.5, 0.0) - Z_CS(e)) / Z_CS(e) * 100 for e in eta_range]
    
    ax.plot(eta_range, err_lutsko, 'b-', lw=2, label='Lutsko')
    ax.plot(eta_range, err_gul, 'g-', lw=2, label='Gül et al.')
    ax.plot(eta_range, err_rosenfeld, 'm-', lw=2, label='Rosenfeld')
    ax.axhline(0, color='k', ls='-', lw=1)
    
    ax.set_xlabel(r'$\eta$', fontsize=14)
    ax.set_ylabel('Contact Density Error (%)', fontsize=14)
    ax.set_title('(b) Relative Error vs Carnahan-Starling', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 0.55])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Contact Density: Theory vs Monte Carlo',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/contact_density_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: contact_density_comparison.png")
    plt.close()

# ============================================================================
# FIGURE 3: All profiles combined
# ============================================================================

def plot_all_profiles_combined():
    """Plot all profiles on single figure."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (eta, data) in enumerate(MC_DATA.items()):
        ax.plot(data['z'], data['rho'], 'o-', color=colors[i],
               ms=4, lw=1.5, mfc='white', mew=1,
               label=f'η = {eta}')
        
        # Mark contact
        ax.scatter([0.5], [data['rho'][0]], s=80, marker='s', 
                   c=colors[i], edgecolors='k', zorder=10)
    
    # Wall
    ax.fill_between([0, 0.5], 0, 12, color='gray', alpha=0.2)
    ax.axvline(0.5, color='black', ls=':', alpha=0.5, label='Wall (z=σ/2)')
    
    ax.set_xlabel(r'$z/\sigma$', fontsize=14)
    ax.set_ylabel(r'$\rho(z)\sigma^3$', fontsize=14)
    ax.set_title('Hard-Sphere Density Profiles at Planar Hard Wall', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 11])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/wall_profiles_all.png', dpi=150, bbox_inches='tight')
    print("Saved: wall_profiles_all.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Generating wall profile plots from MC data...")
    plot_wall_profiles()
    plot_contact_density()
    plot_all_profiles_combined()
    print("\nDone!")
