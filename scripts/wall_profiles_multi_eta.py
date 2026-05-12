"""Four-panel wall profiles at η = 0.367, 0.393, 0.449, 0.492."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/mnt/user-data/uploads')
from fmt_1d_wbii_tensor import WallSolver, RosenfeldFMT, WhiteBearIIFMT, ModifiedRSLT

PI = np.pi

# MC data
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
    ]),
    0.393: np.array([
        [0.510, 4.6143129], [0.530, 3.9234880], [0.550, 3.3460584],
        [0.570, 2.8629162], [0.590, 2.4580002], [0.610, 2.1180064],
        [0.630, 1.8322161], [0.650, 1.5915280], [0.670, 1.3885278],
        [0.690, 1.2169994], [0.710, 1.0718423], [0.730, 0.9487341],
        [0.750, 0.8442476], [0.770, 0.7554538], [0.790, 0.6798796],
        [0.810, 0.6155189], [0.830, 0.5607089], [0.850, 0.5140389],
        [0.870, 0.4743454], [0.890, 0.4407258], [0.910, 0.4123178],
        [0.930, 0.3885200], [0.950, 0.3688011], [0.970, 0.3527117],
        [0.990, 0.3398599], [1.010, 0.3300786], [1.030, 0.3231189],
        [1.050, 0.3188159], [1.070, 0.3171822], [1.090, 0.3182020],
        [1.110, 0.3218368], [1.130, 0.3283230], [1.150, 0.3377242],
    ]),
    0.449: np.array([
        [0.510, 7.1434255], [0.530, 5.6966596], [0.550, 4.5630358],
        [0.570, 3.6720352], [0.590, 2.9702559], [0.610, 2.4154181],
        [0.630, 1.9761309], [0.650, 1.6268439], [0.670, 1.3483351],
        [0.690, 1.1256610], [0.710, 0.9469973], [0.730, 0.8031472],
        [0.750, 0.6869939], [0.770, 0.5928965], [0.790, 0.5164968],
        [0.810, 0.4542666], [0.830, 0.4035183], [0.850, 0.3621957],
        [0.870, 0.3284787], [0.890, 0.3011020], [0.910, 0.2790407],
        [0.930, 0.2615448], [0.950, 0.2478760], [0.970, 0.2376521],
        [0.990, 0.2305936], [1.010, 0.2263595], [1.030, 0.2249940],
        [1.050, 0.2263999], [1.070, 0.2307339], [1.090, 0.2381395],
        [1.110, 0.2489811], [1.130, 0.2637171], [1.150, 0.2830402],
    ]),
    0.492: np.array([
        [0.510, 9.9922671], [0.530, 7.4945040], [0.550, 5.6419169],
        [0.570, 4.2651678], [0.590, 3.2397498], [0.610, 2.4745935],
        [0.630, 1.9022182], [0.650, 1.4731428], [0.670, 1.1505633],
        [0.690, 0.9072574], [0.710, 0.7228926], [0.730, 0.5827276],
        [0.750, 0.4755202], [0.770, 0.3932636], [0.790, 0.3296989],
        [0.810, 0.2803417], [0.830, 0.2419543], [0.850, 0.2119487],
        [0.870, 0.1883925], [0.890, 0.1700398], [0.910, 0.1557248],
        [0.930, 0.1448624], [0.950, 0.1368231], [0.970, 0.1312500],
        [0.990, 0.1279621], [1.010, 0.1269121], [1.030, 0.1280122],
        [1.050, 0.1315832], [1.070, 0.1377009], [1.090, 0.1469531],
        [1.110, 0.1598221], [1.130, 0.1773741], [1.150, 0.2010002],
    ]),
}

print("="*60)
print("FOUR-PANEL WALL PROFILES")
print("="*60)

eta_values = [0.367, 0.393, 0.449, 0.492]
solver = WallSolver(nz=512, Lz=6.0, R=0.5)

functionals = [
    ('Rosenfeld', RosenfeldFMT()),
    ('White Bear II', WhiteBearIIFMT()),
    ('Modified RSLT', ModifiedRSLT()),
]

colors = {'Rosenfeld': 'C0', 'White Bear II': 'C1', 'Modified RSLT': 'C2'}

all_results = {}

for eta in eta_values:
    print(f"\nη = {eta}:")
    all_results[eta] = {}
    for name, func in functionals:
        result = solver.solve(eta, func, max_iter=1500, tol=1e-7, verbose=False)
        all_results[eta][name] = result
        print(f"  {name}: contact = {result['contact']:.3f}")

# Create figure
print("\nCreating figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, eta in enumerate(eta_values):
    ax = axes[idx]
    
    # MC data
    mc_data = MC_PROFILES[eta]
    mc_rho_bulk = 6 * eta / PI
    mc_z = mc_data[:, 0]
    mc_rho_norm = mc_data[:, 1] / mc_rho_bulk
    
    ax.plot(mc_z, mc_rho_norm, 'ko', ms=4, mfc='white', mew=1.5, 
            alpha=0.8, label='MC', zorder=10)
    
    # FMT results
    for name, result in all_results[eta].items():
        ax.plot(result['z'], result['rho_norm'], '-', color=colors[name], 
                lw=1.5, label=f"{name}: {result['contact']:.2f}")
    
    # Reference lines
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.axvline(0.5, color='gray', ls='--', alpha=0.3)
    
    # CS contact
    CS_Z = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    ax.axhline(CS_Z, color='green', ls=':', alpha=0.5, lw=1.5)
    
    ax.set_xlabel(r'$z/\sigma$', fontsize=11)
    ax.set_ylabel(r'$\rho(z)/\rho_{\mathrm{bulk}}$', fontsize=11)
    ax.set_title(f'η = {eta}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim([0.4, 2.0])
    ax.grid(True, alpha=0.3)
    
    # Adjust y-limits
    if eta < 0.4:
        ax.set_ylim([0, 8])
    elif eta < 0.46:
        ax.set_ylim([0, 10])
    else:
        ax.set_ylim([0, 15])

plt.suptitle('Hard Sphere Density Profiles at Planar Hard Wall\n(FMT vs Monte Carlo)', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('outputs/wall_profiles.png', dpi=150, bbox_inches='tight')
print("Saved: wall_profiles.png")
plt.close()

# Summary table
print("\n" + "="*60)
print("CONTACT DENSITY SUMMARY")
print("="*60)
print(f"\n{'η':>6} {'MC':>8} {'CS':>8} {'Rosenfeld':>10} {'WBII':>10} {'mRSLT':>10}")
print("-"*60)

for eta in eta_values:
    mc_data = MC_PROFILES[eta]
    mc_rho_bulk = 6 * eta / PI
    mc_contact = mc_data[0, 1] / mc_rho_bulk
    CS_Z = (1 + eta + eta**2 - eta**3) / (1 - eta)**3
    
    ros = all_results[eta]['Rosenfeld']['contact']
    wbii = all_results[eta]['White Bear II']['contact']
    mrslt = all_results[eta]['Modified RSLT']['contact']
    
    print(f"{eta:6.3f} {mc_contact:8.2f} {CS_Z:8.2f} {ros:10.2f} {wbii:10.2f} {mrslt:10.2f}")

print("\nDone!")
