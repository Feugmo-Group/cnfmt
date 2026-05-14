# CNFMT: Conditional Neural Fundamental Measure Theory

[![JAX](https://img.shields.io/badge/JAX-Accelerated-blue)](https://github.com/google/jax)
[![Equinox](https://img.shields.io/badge/Equinox-Neural_Networks-green)](https://github.com/patrick-kidger/equinox)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A JAX/Equinox implementation of **Fundamental Measure Theory (FMT)** for classical Density Functional Theory (cDFT) of hard-sphere fluids, enhanced with neural networks that learn optimal functional parameters.

The package trains neural networks to predict density-dependent Lutsko parameters A(eta) and B(eta), combining classical FMT functionals (Rosenfeld, White Bear II, Modified RSLT) with data-driven optimization against Carnahan-Starling thermodynamics and Monte Carlo wall profiles.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Using the Package in Your Code](#using-the-package-in-your-code)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Installation

### 1. Create a Conda Environment (Recommended)

```bash
conda create -n cnfmt python=3.10
conda activate cnfmt
```

### 2. Install JAX

JAX installation depends on your hardware. Pick the one that matches your setup:

**CPU only (macOS / Linux):**
```bash
pip install jax jaxlib
```

**GPU (CUDA 12, Linux):**
```bash
pip install jax[cuda12]
```

**Apple Silicon (macOS, Metal):**
```bash
pip install jax-metal
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other configurations.

### 3. Install CNFMT

Clone the repository and install in editable mode:

```bash
git clone https://github.com/username/cnfmt.git
cd cnfmt
pip install -e .
```

This installs all required dependencies automatically:

| Package | Version | Purpose |
|---------|---------|---------|
| `jax` | >= 0.4.0 | Array computation, automatic differentiation |
| `jaxlib` | >= 0.4.0 | JAX backend |
| `equinox` | >= 0.11.0 | Neural network modules |
| `optax` | >= 0.1.0 | Optimizers (Adam, L-BFGS) |
| `numpy` | >= 1.21.0 | Numerical utilities |
| `matplotlib` | >= 3.5.0 | Plotting and figure generation |

**Optional dev tools** (for formatting and testing):

```bash
pip install -e ".[dev]"
```

This adds `pytest`, `black`, and `isort`.

### 4. Verify Installation

```bash
python -c "import cnfmt; print(f'CNFMT v{cnfmt.__version__} ready')"
```

You should see: `CNFMT v1.0.0 ready`

---

## Quick Start

Run the fast training comparison to verify everything works (takes about 2 minutes):

```bash
python -m cnfmt.scripts.fast_four_approaches --fast
```

This trains four different approaches for learning A(eta), B(eta) and saves a 9-panel comparison figure to `outputs/`.

---

## Running Experiments

All scripts are run from the **repository root** using `python -m cnfmt.scripts.<name>`. Figures are saved to the `outputs/` directory.

### Main Experiments

#### 1. Compare Training Approaches

Trains neural networks with four different loss functions and compares learned parameters, thermodynamics, and wall contact densities.

```bash
# Quick version (~2 min, 100 iterations)
python -m cnfmt.scripts.fast_four_approaches --fast

# Full version (~10 min, 500 iterations)
python -m cnfmt.scripts.fast_four_approaches
```

**Output:** `outputs/four_approaches_comparison.png` (9-panel figure)

#### 2. Wall Density Profiles

Computes hard-sphere density profiles at a planar hard wall for multiple packing fractions, compared against Monte Carlo data from Davidchack et al. (2016).

```bash
python -m cnfmt.scripts.wall_profiles_multi_eta
```

**Output:** `outputs/wall_profiles.png` (4-panel figure at eta = 0.367, 0.393, 0.449, 0.492)

#### 3. FMT Method Comparison

Compares all implemented FMT functionals: Rosenfeld, White Bear II, Modified RSLT, esFMT, and Gul et al. Includes density profiles, contact densities, and direct correlation functions.

```bash
python -m cnfmt.scripts.fmt_comprehensive_all_methods
```

**Output:** `outputs/fmt_comprehensive_all_methods.png` (4-panel figure)

#### 4. Lennard-Jones Phase Diagram

Computes vapor-liquid coexistence curves using the Lutsko FMT framework with WCA decomposition. Also trains a neural network for temperature-dependent A(eta, T*), B(eta, T*).

```bash
python -m cnfmt.lj.phase_diagram
```

**Output:**
- `outputs/lj_phase_diagram_lutsko.png` (4-panel: coexistence, P, mu, free energy)
- `outputs/lj_phase_diagram_nn.png` (6-panel: NN training and learned parameters)

### Additional Scripts

| Script | Command | Description |
|--------|---------|-------------|
| NN wall analysis | `python -m cnfmt.scripts.nn_wall_analysis` | Wall profiles with trained neural network |
| NN wall profiles | `python -m cnfmt.scripts.nn_wall_profiles` | Neural network density profiles |
| Three-phase training | `python -m cnfmt.scripts.train_three_phase` | Multi-phase training (bulk, test-particle, wall) |
| Test particle training | `python -m cnfmt.scripts.train_test_particle` | Train with test-particle insertion loss |
| Bulk training | `python -m cnfmt.scripts.train_bulk` | Train on bulk thermodynamics only |
| Feature ablation | `python -m cnfmt.scripts.feature_ablation` | Test different input feature sets |
| Regenerate paper figures | `python -m cnfmt.scripts.regenerate_paper_figures` | Regenerate all publication figures |

### Run All Experiments

```bash
python -m cnfmt.scripts.run_all
```

---

## Using the Package in Your Code

### Compute a Wall Density Profile

```python
from cnfmt.solvers import WallSolver, RosenfeldFMT, WhiteBearIIFMT

# Create solver (1024 grid points, box length 6 sigma)
solver = WallSolver(nz=1024, Lz=6.0, R=0.5)

# Solve for packing fraction eta = 0.367
result = solver.solve(
    eta=0.367,
    functional=RosenfeldFMT(),
    max_iter=3000,
    tol=1e-8
)

print(f"Contact density: {result['contact']:.3f}")

# Access full profile
z = result['z']               # Position array
rho_norm = result['rho_norm']  # rho(z) / rho_bulk
```

### Compare Multiple Functionals

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
}

for name, func in functionals.items():
    result = solver.solve(eta, func, max_iter=3000)
    print(f"{name}: contact = {result['contact']:.3f}")
```

### Train a Neural Network for A(eta), B(eta)

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from cnfmt import ConditionalNetwork, TrainingConfig
from cnfmt.training import train_bulk_phase

# Create network and train
key = jax.random.PRNGKey(42)
network = ConditionalNetwork(key, n_features=5)

config = TrainingConfig(n_epochs=500, lr=1e-3)
network, losses = train_bulk_phase(network, config)

# Predict parameters at a given packing fraction
A, B = network.from_eta(0.3)
print(f"A = {A:.3f}, B = {B:.3f}")
```

### Compute the LJ Phase Diagram

```python
from cnfmt.lj.phase_diagram import LJPhaseDiagram, LJPotential

potential = LJPotential(sigma=1.0, epsilon=1.0, r_cut=3.0)
print(f"BH diameter at T*=1: {potential.barker_henderson_diameter(1.0):.4f}")

pd = LJPhaseDiagram(A=1.0, B=0.0)
coex = pd.compute_coexistence_curve(T_min=0.7, T_max=1.35, n_points=50)
print(f"Critical temperature: T*_c = {coex['T'][-1]:.2f}")
```

---

## Project Structure

```
cnfmt/
├── core/                          # Core computational modules
│   ├── grid.py                    # 3D FFT-based computational grid
│   ├── weights.py                 # FMT weight kernels in Fourier space
│   ├── densities.py               # Weighted density calculator (FFT convolution)
│   ├── thermodynamics.py          # EOS: Percus-Yevick, Carnahan-Starling, Lutsko
│   └── constants.py               # Physical constants
│
├── functionals/                   # Free energy functionals
│   ├── lutsko.py                  # Lutsko esFMT with (A, B) parameterization
│   └── potentials.py              # Grand potential, LJ with WCA decomposition
│
├── neural/                        # Neural network components
│   ├── network.py                 # ConditionalNetwork: eta -> (A, B)
│   └── features.py                # Feature extraction from density fields
│
├── solvers/                       # DFT equation solvers
│   ├── fmt_1d_wbii_tensor.py      # Validated 1D solver (recommended)
│   ├── fmt_3d_tensor.py           # Full 3D solver with tensor terms
│   ├── wall_profile.py            # Wall profile calculator
│   ├── minimizer.py               # Adam / L-BFGS density optimization
│   └── test_particle.py           # Test particle insertion
│
├── training/                      # Training infrastructure
│   ├── losses.py                  # Loss functions (EOS, contact, consistency)
│   ├── optimizers.py              # Optax optimizer wrappers
│   ├── config.py                  # TrainingConfig dataclass
│   └── checkpoints.py             # Model save / load
│
├── lj/                            # Lennard-Jones extension
│   └── phase_diagram.py           # VLE coexistence, NN training for A(eta,T*)
│
├── scripts/                       # Runnable experiment scripts
│   ├── fast_four_approaches.py    # Training approach comparison
│   ├── wall_profiles_multi_eta.py # Wall profiles at multiple eta
│   ├── fmt_comprehensive_all_methods.py  # FMT functional comparison
│   ├── run_all.py                 # Run all experiments
│   └── ...                        # Additional analysis scripts
│
├── utils/                         # Utilities
│   ├── plotting.py                # Visualization helpers
│   └── analysis.py                # Data analysis tools
│
├── data/                          # Reference data
│   └── hswall/                    # Monte Carlo wall profiles (Davidchack 2016)
│
├── outputs/                       # Generated figures and logs
├── figures/                       # Additional figure outputs
├── setup.py                       # Package installation
└── README.md                      # This file
```

---

## Output Files

All experiment scripts save their results to `outputs/`. Here are the key outputs:

| File | Generated By | Description |
|------|-------------|-------------|
| `four_approaches_comparison.png` | `fast_four_approaches` | 9-panel training comparison |
| `wall_profiles.png` | `wall_profiles_multi_eta` | Density profiles at 4 packing fractions |
| `fmt_comprehensive_all_methods.png` | `fmt_comprehensive_all_methods` | FMT functional comparison |
| `lj_phase_diagram_lutsko.png` | `lj.phase_diagram` | LJ vapor-liquid coexistence |
| `lj_phase_diagram_nn.png` | `lj.phase_diagram` | NN-learned LJ parameters |
| `nn_wall_profiles.png` | `nn_wall_profiles` | Neural network wall profiles |
| `three_phase_comparison.png` | `train_three_phase` | Multi-phase training results |
| `feature_ablation.png` | `feature_ablation` | Input feature comparison |

---

## Troubleshooting

### "No module named cnfmt"

Make sure you installed the package in editable mode from the repository root:

```bash
cd cnfmt
pip install -e .
```

### JAX not using GPU

Check your JAX backend:

```python
import jax
print(jax.devices())
```

If it shows only CPU devices, reinstall JAX with GPU support (see [Installation](#installation)).

### "float64 not supported" or precision errors

CNFMT requires 64-bit floating point. This is enabled automatically when you import the package. If you see precision-related errors, make sure to import `cnfmt` before other JAX operations:

```python
import cnfmt  # This sets jax_enable_x64 = True
```

### Slow first run

JAX compiles functions on first use (JIT compilation). The first run of any script will be slower than subsequent runs. This is normal.

### matplotlib display issues (headless server)

If running on a server without a display:

```bash
export MPLBACKEND=Agg
python -m cnfmt.scripts.fast_four_approaches
```

Figures will still be saved to `outputs/` even without a display.

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

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{cnfmt2024,
  title={CNFMT: Conditional Neural Fundamental Measure Theory},
  author={Tetsas, C.},
  year={2024},
  url={https://github.com/username/cnfmt}
}
```
