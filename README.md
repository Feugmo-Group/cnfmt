# CNFMT: Conditional Neural Fundamental Measure Theory

[![JAX](https://img.shields.io/badge/JAX-Accelerated-blue)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-Neural_Networks-green)](https://github.com/patrick-kidger/equinox)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A JAX/Equinox implementation of **Fundamental Measure Theory (FMT)** for classical Density Functional Theory (cDFT) of hard-sphere fluids, with neural network enhancements for learning optimal functional parameters.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Package Structure](#package-structure)
- [Theory](#theory)
- [Implemented Methods](#implemented-methods)
- [Usage Examples](#usage-examples)
- [Scripts and Figures](#scripts-and-figures)
- [Key Results](#key-results)
- [References](#references)

---

## Overview

**CNFMT** implements the Lutsko framework for Fundamental Measure Theory, which provides a unified description of hard-sphere fluids through density-dependent parameters A(η) and B(η). The package combines:

1. **Classical FMT functionals** (Rosenfeld, White Bear II, Modified RSLT)
2. **Neural network parameterization** for learning optimal A(η), B(η)
3. **Lennard-Jones extension** via WCA decomposition and mean-field attraction
4. **Validated solvers** for density profiles at hard walls

The key innovation is the **esFMT (extended scaled FMT)** framework where the third contribution to the free energy density is parameterized as:

```
Φ₃ = [A·(n₂³ - 3n₂n⃗₂² + ...) + B·(n₂³ - 3n₂·Tr(T²) + ...)] / (24π(1-η)²)
```

Different choices of (A, B) recover different classical functionals and equations of state.

---

## Features

- ✅ **Multiple FMT Functionals**: Rosenfeld, White Bear II, Modified RSLT, esFMT
- ✅ **Neural Network A(η), B(η)**: Learn optimal parameters from thermodynamic constraints
- ✅ **Validated 1D/3D Solvers**: Accurate wall profiles matching Monte Carlo data
- ✅ **Lennard-Jones Phase Diagram**: Vapor-liquid coexistence via Lutsko method
- ✅ **JAX Acceleration**: GPU/TPU compatible, automatic differentiation
- ✅ **Comprehensive Visualization**: Publication-ready figures

---

## Installation

### Requirements

```bash
pip install jax jaxlib equinox optax numpy scipy matplotlib
```

### Install Package

```bash
git clone https://github.com/your-repo/cnfmt.git
cd cnfmt
pip install -e .
```

---

## Package Structure

```
cnfmt/
│
├── core/                    # Core computational modules
│   ├── grid.py              # 3D computational grid with FFT support
│   ├── weights.py           # FMT weight functions (scalar, vector, tensor)
│   ├── densities.py         # Weighted density calculations
│   └── thermodynamics.py    # EOS, chemical potential, compressibility
│
├── functionals/             # Free energy functionals
│   ├── lutsko.py            # Lutsko esFMT with (A, B) parameters
│   └── potentials.py        # LJ potential with WCA decomposition
│
├── neural/                  # Neural network components
│   ├── network.py           # MLP for A(η), B(η) or A(η, T*)
│   └── features.py          # Feature engineering (η, η², log(1-η), ...)
│
├── solvers/                 # DFT solvers
│   ├── fmt_1d_wbii_tensor.py   # ★ Validated 1D FMT (RECOMMENDED)
│   ├── fmt_3d_tensor.py        # Full 3D FMT with tensor terms
│   ├── wall_profile.py         # Wall profile calculator
│   ├── minimizer.py            # Density minimization (Adam, L-BFGS)
│   └── test_particle.py        # Test particle insertion
│
├── training/                # Training infrastructure
│   ├── losses.py            # Loss functions (EOS, contact, δμ, δχ)
│   ├── optimizers.py        # Optimizer configurations
│   ├── config.py            # Training hyperparameters
│   └── checkpoints.py       # Model saving/loading
│
├── lj/                      # Lennard-Jones module
│   └── phase_diagram.py     # VLE coexistence, NN training
│
├── scripts/                 # Runnable scripts for figures
│   ├── fast_four_approaches.py      # Compare training approaches
│   ├── wall_profiles_multi_eta.py   # Wall profiles at multiple η
│   ├── fmt_comprehensive_all_methods.py  # FMT comparison
│   └── ...
│
└── utils/                   # Utilities
    ├── plotting.py          # Visualization functions
    └── analysis.py          # Data analysis tools
```

---

## Theory

### Fundamental Measure Theory (FMT)

FMT expresses the excess free energy of hard spheres as a functional of **weighted densities**:

```
F_ex[ρ] = ∫ Φ(n_α(r)) dr
```

where the weighted densities are convolutions of the density with geometric weight functions:

| Density | Symbol | Definition | Physical Meaning |
|---------|--------|------------|------------------|
| Packing fraction | n₃ = η | ∫ρ(r')w₃(r-r')dr' | Local volume fraction |
| Surface | n₂ | ∫ρ(r')w₂(r-r')dr' | Surface area density |
| Mean curvature | n₁ | n₂/(4πR) | Curvature contribution |
| Gaussian curvature | n₀ | n₂/(4πR²) | Topological term |
| Vector | n⃗₂ | ∫ρ(r')w⃗₂(r-r')dr' | Orientational density |
| **Tensor** | **T** | ∫ρ(r')w_T(r-r')dr' | Anisotropy (traceless) |

### Weight Functions

For hard spheres of radius R = σ/2:

| Weight | Real Space | Fourier Space |
|--------|------------|---------------|
| w₃(r) | Θ(R - \|r\|) | (4π/k³)[sin(kR) - kR·cos(kR)] |
| w₂(r) | δ(\|r\| - R) | 4πR²·sin(kR)/(kR) |
| w⃗₂(r) | (r/\|r\|)δ(\|r\| - R) | -4πiR·j₁(kR)·k̂ |
| w_T(r) | (rr/r² - I/3)δ(\|r\| - R) | 4πR²·3j₂(kR)·(k̂k̂ - I/3) |

### Free Energy Density

The free energy density has three contributions:

```
Φ = Φ₁ + Φ₂ + Φ₃
```

**Φ₁ — Ideal Cavity Term:**
```
Φ₁ = -n₀ ln(1 - η)
```

**Φ₂ — Two-Body Correlations:**
```
Φ₂ = (n₁n₂ - n⃗₁·n⃗₂) / (1 - η)
```

**Φ₃ — Three-Body Correlations (various formulations):**

| Formulation | Φ₃ Expression |
|-------------|---------------|
| Rosenfeld | (n₂³ - 3n₂n⃗₂²) / (24π(1-η)²) |
| White Bear II | φ₃(η)·(n₂³ - 3n₂n⃗₂²) / (24π(1-η)²) |
| esFMT (Lutsko) | [A·term_A + B·term_B] / (24π(1-η)²) |

### Lutsko (A, B) Parameterization

The esFMT framework parameterizes Φ₃ with two parameters:

```
Φ₃ = [A·term_A + B·term_B] / (24π(1-η)²)
```

where:
- **term_A** = n₂³ - 3n₂(n⃗₂)² + 3(n⃗₂·T·n⃗₂) - Tr(T³)
- **term_B** = n₂³ - 3n₂·Tr(T²) + 2·Tr(T³)

The **constraint parameter** C = 8A + 2B - 9 determines the bulk equation of state:

| C Value | EOS | Functional |
|---------|-----|------------|
| C = +3 | Percus-Yevick | Rosenfeld (A=1.5, B=0) |
| C = 0 | Near PY | Transition point |
| C = -1 | Lutsko baseline | (A=1, B=0) |
| C = -3 | Carnahan-Starling | Exact bulk thermodynamics |

### White Bear II Corrections

For accurate bulk thermodynamics, WBII introduces correction functions:

```
φ₂(η) = 1 - [2η - 3η² + 2η³ + 2(1-η)²ln(1-η)] / (3η²)

φ₃(η) = 1 - [2η - η² + 2(1-η)ln(1-η)] / (3η²)
```

These ensure the **Carnahan-Starling equation of state** in bulk:

```
Z_CS = (1 + η + η² - η³) / (1 - η)³
```

### Direct Correlation Function

The one-body direct correlation function is computed via the chain rule:

```
c⁽¹⁾(r) = -δF_ex/δρ(r) = -Σ_α (∂Φ/∂n_α ★ w_α)
```

For **Rosenfeld FMT**, c(r) equals the **Percus-Yevick** result exactly:

```
c_PY(r) = -α + β(r/σ) - γ(r/σ)³   for r < σ
        = 0                        for r ≥ σ
```

where:
- α = (1 + 2η)² / (1-η)⁴
- β = 6η(1 + η/2)² / (1-η)⁴
- γ = η(1 + 2η)² / (2(1-η)⁴)

### Lennard-Jones Extension

For Lennard-Jones fluids, we use WCA decomposition:

```
v_LJ(r) = v_rep(r) + w_att(r)
```

**Barker-Henderson diameter** (temperature-dependent):
```
d(T) = ∫₀^r_min [1 - exp(-βv_rep(r))] dr
```

**Mean-field attraction:**
```
a = ∫ 4πr² w_att(r) dr ≈ -14.56 εσ³  (for r_c = 3σ)
```

**Total free energy:**
```
f = f_id + f_HS(η_eff) + (a/2)ρ²/kT
```

---

## Implemented Methods

### FMT Functionals

| Class | A | B | C | Description |
|-------|---|---|---|-------------|
| `RosenfeldFMT` | 1.5 | 0.0 | +3 | Original FMT (1989), PY EOS |
| `WhiteBearIIFMT` | - | - | -3 | φ₂, φ₃ corrections, CS EOS |
| `ModifiedRSLT` | - | - | - | (1-ξ²)³ factor, positive definite |
| `esFMT_Tensor` | A | B | 8A+2B-9 | General (A,B) with tensor terms |

### Neural Network Architectures

**Hard-Sphere Network: A(η), B(η)**
```python
class SimpleNetwork:
    input_features: [η, η², η³, log(1-η), 1/(1-η)]
    hidden_layers: [32, 32] with SiLU activation
    output: [A, B] with sigmoid constraints
        A ∈ [0.8, 1.5]
        B ∈ [-1.5, 0.0]
```

**Lennard-Jones Network: A(η, T*), B(η, T*)**
```python
class ABNetwork:
    input_features: [η, T*, η², T*², η·T*, log(1-η), 1/(1-η)]
    hidden_layers: [32, 32] with SiLU activation
    output: [A, B] with constraints
```

### Training Approaches

| # | Name | Loss Function | Target |
|---|------|--------------|--------|
| 1 | CS EOS | \|Z - Z_CS\|² + λ\|μ - μ_CS\|² | Bulk thermodynamics |
| 2 | δμ, δχ | (δμ/μ)² + (δχ/χ)² | DFT-bulk consistency |
| 3 | Contact | \|Z_DFT - Z_CS\|² / Z_CS² | Wall contact density |
| 4 | Combined | L_EOS + λ·L_contact | Multi-objective |

### Solvers

| Solver | Geometry | Method | Key Features |
|--------|----------|--------|--------------|
| `WallSolver` | 1D planar | Picard iteration | Real-space convolution, tensor weights |
| `DFTSolver3D` | Full 3D | Picard iteration | FFT convolution, periodic BC |
| `DensityMinimizer` | General | Adam/L-BFGS | Gradient-based optimization |

---

## Usage Examples

### 1. Compute Wall Density Profile

```python
from cnfmt.solvers import WallSolver, RosenfeldFMT, WhiteBearIIFMT

# Create solver (1024 grid points, box length 6σ)
solver = WallSolver(nz=1024, Lz=6.0, R=0.5)

# Solve for packing fraction η = 0.367
result = solver.solve(
    eta=0.367, 
    functional=RosenfeldFMT(), 
    max_iter=3000, 
    tol=1e-8
)

print(f"Contact density: {result['contact']:.3f}")
# Output: Contact density: 5.835

# Access full profile
z = result['z']           # Position array
rho_norm = result['rho_norm']  # ρ(z)/ρ_bulk
```

### 2. Compare Multiple Functionals

```python
from cnfmt.solvers import (
    WallSolver, RosenfeldFMT, WhiteBearIIFMT, 
    ModifiedRSLT, esFMT_Tensor
)

solver = WallSolver(nz=1024, Lz=6.0)
eta = 0.367

functionals = {
    'Rosenfeld': RosenfeldFMT(),
    'White Bear II': WhiteBearIIFMT(),
    'Modified RSLT': ModifiedRSLT(),
    'esFMT(1,-1)': esFMT_Tensor(A=1.0, B=-1.0),
    'Gül et al.': esFMT_Tensor(A=1.3, B=-1.0),
}

for name, func in functionals.items():
    result = solver.solve(eta, func, max_iter=3000)
    print(f"{name}: contact = {result['contact']:.3f}")
```

### 3. Train Neural Network A(η), B(η)

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Define network
class SimpleNetwork(eqx.Module):
    layers: list
    
    def __init__(self, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(5, 32, key=keys[0]),
            eqx.nn.Linear(32, 32, key=keys[1]),
            eqx.nn.Linear(32, 2, key=keys[2]),
        ]
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)
    
    def from_eta(self, eta):
        features = jnp.array([
            eta, eta**2, eta**3, 
            jnp.log(1-eta+1e-10), 
            1/(1-eta+1e-10)
        ])
        out = self(features)
        A = 0.8 + 0.7 * jax.nn.sigmoid(out[0])
        B = -1.5 + 1.5 * jax.nn.sigmoid(out[1])
        return A, B

# Train
network = SimpleNetwork(jax.random.PRNGKey(42))
optimizer = optax.adamw(1e-3)
# ... training loop
```

### 4. Lennard-Jones Phase Diagram

```python
from cnfmt.lj.phase_diagram import LJPhaseDiagram, LJPotential

# Create LJ potential with cutoff
potential = LJPotential(sigma=1.0, epsilon=1.0, r_cut=3.0)

print(f"BH diameter at T*=1: {potential.barker_henderson_diameter(1.0):.4f}")
print(f"vdW parameter: {potential.vdw_parameter():.4f}")

# Compute coexistence curve
pd = LJPhaseDiagram(A=1.0, B=0.0)  # Lutsko parameters
coex = pd.compute_coexistence_curve(T_min=0.7, T_max=1.35, n_points=50)

print(f"Critical temperature: T*_c ≈ {coex['T'][-1]:.2f}")
# Output: Critical temperature: T*_c ≈ 1.28
```

### 5. Direct Correlation Function

```python
import numpy as np

def c_PY(r, eta, sigma=1.0):
    """Percus-Yevick direct correlation function."""
    alpha = (1 + 2*eta)**2 / (1 - eta)**4
    beta = 6*eta * (1 + eta/2)**2 / (1 - eta)**4
    gamma = eta * (1 + 2*eta)**2 / (2*(1 - eta)**4)
    x = r / sigma
    return np.where(r < sigma, -alpha + beta*x - gamma*x**3, 0.0)

# Compute c(r) at η = 0.3
r = np.linspace(0.01, 1.5, 200)
c_r = c_PY(r, eta=0.3)

# Fourier transform to get ĉ(k)
def c_fourier(r, c_r, k_max=30.0):
    k = np.linspace(0.01, k_max, 256)
    dr = r[1] - r[0]
    c_k = np.zeros_like(k)
    for i, ki in enumerate(k):
        c_k[i] = 4*np.pi * np.sum(r**2 * c_r * np.sin(ki*r)/(ki*r)) * dr
    return k, c_k
```

---

## Scripts and Figures

### Quick Start — Generate All Key Figures

```bash
cd cnfmt

# 1. LJ Phase Diagram (generates 2 figures)
python -m lj.phase_diagram

# 2. Four Training Approaches (generates 1 figure)
python scripts/fast_four_approaches.py

# 3. Wall Profiles at Multiple η (generates 1 figure)
python scripts/wall_profiles_multi_eta.py

# 4. FMT Comparison with c(r) (generates 1 figure)
python scripts/fmt_comprehensive_all_methods.py

# 5. Validated 1D FMT (generates 1 figure)
python solvers/fmt_1d_wbii_tensor.py
```

### Figure Summary

| Script | Output File | Description |
|--------|-------------|-------------|
| `lj/phase_diagram.py` | `lj_phase_diagram_lutsko.png` | 4-panel: VLE, P(ρ), μ(ρ), f(ρ) |
| `lj/phase_diagram.py` | `lj_phase_diagram_nn.png` | 6-panel: NN training, learned A,B |
| `scripts/fast_four_approaches.py` | `four_approaches_comparison.png` | 9-panel training comparison |
| `scripts/wall_profiles_multi_eta.py` | `wall_profiles.png` | 4-panel: η = 0.367, 0.393, 0.449, 0.492 |
| `scripts/fmt_comprehensive_all_methods.py` | `fmt_comprehensive_all_methods.png` | 4-panel: profiles, c(r), ĉ(k) |
| `solvers/fmt_1d_wbii_tensor.py` | `fmt_1d_wbii_tensor.png` | Tensor FMT vs Monte Carlo |

See **[SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)** for complete documentation.

---

## Key Results

### Contact Density at Hard Wall (η = 0.367)

| Method | Contact ρ(R⁺)/ρ_bulk | % of MC | % of CS |
|--------|---------------------|---------|---------|
| Monte Carlo | 5.36 | 100% | 94% |
| Carnahan-Starling | 5.73 | 107% | 100% |
| **Rosenfeld FMT** | **5.84** | **109%** | 102% |
| White Bear II | 5.07 | 95% | 89% |
| Modified RSLT | 5.52 | 103% | 96% |
| esFMT(1,-1) | 4.29 | 80% | 75% |
| Gül et al. | 4.76 | 89% | 83% |

### LJ Critical Point Comparison

| Method | T*_c | ρ*_c σ³ |
|--------|------|---------|
| Lutsko (A=1, B=0) | 1.28 | 0.31 |
| Simulation (Johnson) | 1.31 | 0.31 |
| Mean-field estimate | 1.35 | 0.33 |

### Learned Parameters (CS EOS Training)

At η = 0.3:
- A ≈ 1.27
- B ≈ -0.63
- C ≈ -0.05 (near PY line)

---

## References

1. **Rosenfeld, Y.** (1989). Free-energy model for the inhomogeneous hard-sphere fluid mixture and density-functional theory of freezing. *Phys. Rev. Lett.* 63, 980.

2. **Roth, R., Evans, R., Lang, A., & Kahl, G.** (2002). Fundamental measure theory for hard-sphere mixtures revisited: the White Bear version. *J. Phys.: Condens. Matter* 14, 12063.

3. **Hansen-Goos, H. & Roth, R.** (2006). Density functional theory for hard-sphere mixtures: the White Bear version mark II. *J. Phys.: Condens. Matter* 18, 8413.

4. **Lutsko, J. F.** (2007). Density functional theory of inhomogeneous liquids. I. The liquid-vapor interface in Lennard-Jones fluids. *J. Chem. Phys.* 127, 054701.

5. **Tarazona, P.** (2000). Density functional for hard sphere crystals: A fundamental measure approach. *Phys. Rev. Lett.* 84, 694.

6. **Davidchack, R. L., Laird, B. B., & Roth, R.** (2016). Hard spheres at a planar hard wall: Simulations and density functional theory. *Condens. Matter Phys.* 19, 23001.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{cnfmt2024,
  title={CNFMT: Conditional Neural Fundamental Measure Theory},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/cnfmt}
}
```

---

*For questions, issues, or contributions, please open an issue or pull request on GitHub.*
