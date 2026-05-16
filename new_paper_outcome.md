# Nonlocal Neural Functional — New Paper Outcomes

## Novelty Summary

### vs. Classical FMT functionals (Rosenfeld, White Bear II, Gül et al.)

Classical functionals fix A and B as **constants** (e.g., Lutsko: A=1, B=0; Gül: A=1.3,
B=−1). This work makes **A and B learnable functions** — first of η alone, then of (η, T*)
for Lennard-Jones, and now of nonlocal density features via the kernel extension. The
functional form stays interpretable: you can always read off where you are in the A–B plane
and map to known limits (Rosenfeld, White Bear, CS). The parameters adapt to the training
objective while remaining physically anchored.

### vs. Sammüller et al. (neural functional theory)

Sammüller et al. learn the full mapping ρ → c⁽¹⁾ — formally exact but a black box
requiring extensive simulation data. This approach:

- Stays **inside the esFMT framework** (only 2 scalar parameters), so every learned state
  has a physical interpretation via C = 8A + 2B − 9
- Requires **zero simulation data** — trained purely through physics constraints:
  bulk EOS matching, SPT exact relations, sine-wave free-energy matching, and Noether
  translational invariance
- Is **end-to-end differentiable** via JAX through the DFT solver itself, enabling
  gradient-based optimization of all network and kernel parameters simultaneously

### The nonlocal extension — the main novel contribution

This goes beyond all prior work by adding a **learnable convolution kernel K̂(k)** in
Fourier space that preprocesses the density field before the network sees it. Specifically:

- A and B now depend on **nonlocal density information**: not just the local packing
  fraction η(r), but a spatially averaged, k-filtered version of ρ(r)
- This is the first FMT functional where the **spatial range of correlations encoded in
  the functional is itself learned**, rather than fixed by the FMT geometry (which only
  encodes the hard-sphere radius R via the weight functions)
- The kernel K̂(k) is unconstrained in Fourier space, allowing the functional to learn
  which length scales matter for a given thermodynamic or structural objective
- Still **simulation-free**: the kernel is trained entirely through bulk constraints and
  free-energy matching on sinusoidal test fields that activate the kernel at specific
  wavenumbers k = 2πm/L

In one sentence: the novelty is a **physics-constrained, interpretable, nonlocal neural
functional** that learns both *what weight to give* the FMT tensor invariants (A, B) and
*over what spatial range* to average the density field (K̂(k)), without any MD or MC data.

---

## Potential Application Systems

The approach is general: any system where cDFT applies and the optimal functional parameters
are unknown or state-dependent. Below are the most natural targets, ordered by proximity to
the current implementation.

### Tier 1 — Direct extensions (minimal new code)

| System | Why it fits | Key change needed |
|--------|------------|-------------------|
| **Hard-sphere mixtures** (binary, polydisperse) | FMT is exact for additive HS mixtures; A, B become functions of partial packing fractions η₁, η₂ | Generalize network input; multi-component weight functions |
| **Lennard-Jones at interfaces** (liquid–vapor, liquid–solid) | BH perturbation theory already implemented; wall training signal would drive C toward −3 | Add interfacial free energy or surface tension loss |
| **Soft repulsive spheres** (WCA, Yukawa) | Map to effective HS via BH or similar diameter; same architecture applies | Generalize barker_henderson_diameter for softer cores |
| **Hard rods / spherocylinders** (liquid crystals) | Tensor FMT already encodes orientational correlations; K̂(k) could learn anisotropic filtering | Extend weight functions to non-spherical shapes |

### Tier 2 — Moderate extensions (new physics terms)

| System | Why it fits | Key change needed |
|--------|------------|-------------------|
| **Charged hard spheres** (primitive model electrolytes) | FMT handles HS repulsion; add mean-field electrostatics (MSA or PB) as perturbation | Add electrostatic free energy term; screen length as extra input |
| **Colloid–polymer mixtures** (Asakura–Oosawa) | Effective depletion attraction mapped to HS + perturbation | Add polymer reservoir chemical potential as input feature |
| **Patchy particles** (directional bonding) | Reference fluid is HS; patch interactions enter as perturbation (Wertheim TPT) | Add bonding free energy contribution |
| **Confined fluids in nanopores** | Same FMT but 1D/2D geometry; kernel learns confinement length scale | Change grid geometry; add pore wall potential |

### Tier 3 — Ambitious extensions (new formalism)

| System | Why it fits | Key change needed |
|--------|------------|-------------------|
| **Polymer melts / block copolymers** | PRISM + FMT hybrid; chain connectivity as nonlocal constraint | Add chain propagator; K̂(k) learns chain correlation hole |
| **Crystal nucleation / solid phases** | cDFT captures solid–liquid transition; learned kernel encodes lattice periodicity | Train on crystalline density waves; K̂(k) peaks at reciprocal lattice vectors |
| **Active matter** (self-propelled particles) | Motility-induced phase separation has HS-like excluded volume; activity as extra parameter | Add swim pressure term; activity Péclet number as network input |
| **Liquid metals** (pseudopotential + HS ref.) | Empty core pseudopotential gives effective HS + perturbation; K̂(k) learns electronic screening | Add electron gas free energy contribution |

### Most Promising Single Target for a Follow-up Paper

**Binary hard-sphere mixtures** are the strongest immediate candidate:
- FMT is exact → ground truth exists for validation
- Depletion-driven phase separation (Asakura–Oosawa) is a well-studied benchmark
- The nonlocal kernel K̂(k) should learn to encode the depletion length scale (set by
  the small-sphere radius), which is genuinely nonlocal information not captured by
  local η alone
- Experimental data abundant (colloidal suspensions); MD/MC data for comparison

---

## Prior ML Approaches in the Literature and in the Previous Paper

### Related work: Sammüller et al. (2023, 2024)

**ML approach:** Neural functional theory — a neural network directly learns the full
one-body direct correlation functional mapping ρ(r) → c⁽¹⁾[ρ](r).

**Training data:** Grand-canonical Monte Carlo (GCMC) simulations of hard spheres and
Lennard-Jones fluids at a planar hard wall. Many density profiles ρ(r) paired with their
corresponding c⁽¹⁾(r) (computed via the DFT self-consistency relation) are used as
supervised training pairs.

**Key properties:**
- Formally exact — no approximation to the functional form is made
- Black box — the network has no interpretable parameters
- Data-hungry — requires a large library of GCMC profiles covering the full state space
- Generalizes to arbitrary interparticle interactions if retrained

---

### Related work: Gül et al. (2024)

**ML approach:** Not machine learning — numerical optimization of the two scalar constants
A and B (fixed, not state-dependent) against test-particle sum-rule residuals computed on
a 3D DFT grid. Result: A=1.3, B=−1.0 (C=−0.6).

**Training data:** No simulation data. The loss is the self-consistency residual of the
test-particle route — DFT is run with a point-source external field and the chemical
potential and compressibility computed via two routes must agree.

**Key limitation:** A and B are optimized as global constants for a single objective
(test-particle sum rules); they cannot adapt to different thermodynamic targets or
state points.

---

### This paper's approaches and data sources

Four training strategies were compared, all operating within the esFMT framework (learning
A(η) and B(η) as neural network outputs):

| Approach | Training signal | Data source | Simulation needed? |
|----------|----------------|-------------|-------------------|
| **Approach 1**: EOS matching | Match Z(η) and μ_ex(η) to Carnahan–Starling | Analytical CS formula | No |
| **Approach 2**: Thermodynamic self-consistency | Minimize deviations in chemical potential and isothermal compressibility χ_T | Analytical CS formula | No |
| **Approach 3**: Wall contact density | Match ρ_contact(η) at a hard wall | **MD data** — Davidchack, Laird & Roth (2016) | Yes (external) |
| **Approach 4**: Combined multi-objective | Weighted sum of Approaches 1–3 | CS formula + MD data | Yes (external) |

**MD benchmark data details (Davidchack et al. 2016):**
- Hard spheres at a planar hard wall; two walls separated by 65σ, cross-section ~50σ×50σ
- 17 reduced densities ρ*=ρσ³ from 0.052 to 0.938; 50 independent runs per density
- Particle counts 8,000–150,000; bin width 0.02σ
- Used here at four packing fractions: η = 0.367, 0.393, 0.449, 0.492

**Three-phase training strategy (the key result):**
1. **Phase 1** (simulation-free): Bulk EOS → A≈1.32, B≈−1.31, C≈−1.1; contact densities undershoot MD by 21–34%
2. **Phase 2** (simulation-free): Test-particle sum-rule residuals on 3D DFT grid (no MC/MD) → C shifts toward 0; provides 5.5× better convergence for Phase 3
3. **Phase 3** (uses external MD): Wall contact density fine-tuning via numerical finite-difference gradients through the Picard solver → contact densities within 1–2% of MD

**LJ extension data source:** The Lutsko reference binodal (A=1, B=0), computed analytically via Barker–Henderson perturbation theory + van der Waals mean field. No simulation data used. The network is trained to reproduce the coexistence curve of the analytical reference, not experimental or MD data.

---

### Summary comparison table

| Method | Architecture | Training data | Interpretable? | Simulation-free? |
|--------|-------------|---------------|----------------|-----------------|
| Sammüller et al. | Deep NN, ρ→c⁽¹⁾ | GCMC profiles (many) | No | No |
| Gül et al. | No ML — constant A,B | DFT self-consistency | Yes | Yes |
| This paper (bulk) | Small NN, η→(A,B) | Analytical CS EOS | Yes | Yes |
| This paper (3-phase) | Small NN, η→(A,B) | CS + MD wall profiles | Yes | Partially |
| **Nonlocal extension** | NN + K̂(k) kernel | Analytical CS + sine-wave DFT | Yes | **Yes** |

The nonlocal extension is the only approach that is simultaneously interpretable,
fully simulation-free, and nonlocal in the density field.

---

## Training Architecture (four-phase curriculum)

| Phase | Loss components | LR | Purpose |
|-------|----------------|----|---------|
| 1 | Bulk EOS (Z, μ vs CS) + SPT exact relations | 3e-3 | Constrain A, B to physical bulk EOS |
| 2 | Sine-wave free-energy matching vs WB-II reference | 1e-3 | Activate K̂(k) at wavenumbers m=1,2,4,8 |
| 3 | Same as Phase 2, lower LR | 5e-4 | Stabilize kernel at more Fourier modes |
| 4 | Phase 2 loss + Noether translational invariance | 2e-4 → 0 (cosine) | Fine-tune, enforce symmetry |

Note: OZ consistency and c₂ reference losses are excluded from Phases 3–4 because
`compute_c2_bulk` (Hessian-vector product) is not correctly normalized for the nonlocal
functional's LearnableKernel contribution. This is left for future work.

---

## Current Results (run3, 32³ grid, L=10σ)

| Phase | Best loss | A(η=0.3) | B(η=0.3) | C = 8A+2B−9 |
|-------|-----------|----------|----------|-------------|
| 1 | 6.65e-3 | 1.035 | −0.270 | −1.26 |
| 2 | 6.71e-3 | 1.280 | −1.277 | −1.32 |
| 3 | 6.70e-3 | 1.280 | −1.277 | −1.32 |
| 4 | 6.70e-3 | ~1.28 | ~−1.27 | ~−1.27 |

C ≈ −1.3 sits between PY (C=0) and CS (C=−3), consistent with the Gül et al. optimized
functional (A≈1.3, B≈−1). This is expected: the sine-wave reference uses White Bear II
which lives near the Gül region. Reaching CS (C=−3) would require wall or interfacial
training data.

---

## Key Bug Fixes Applied (feature/nonlocal-extension)

1. **`low_density_limit_loss`** — was computing `(μ_ex/η)²`; but μ_ex/η → 8 is an exact
   physics constant (second virial coefficient), giving ~64 constant loss with near-zero
   gradient. Fixed to `μ_ex²` which correctly vanishes as η → 0.

2. **`solve_oz_fourier` / `compute_structure_factor`** — denominator clamped from 1e-12
   to 0.05, capping S(k) ≤ 20 and preventing blowup when noisy c₂ pushes the denominator
   negative during early training.

3. **Phase 2 gradient signal** — `float(eta)` broke autodiff; replaced with `jnp.asarray`.
   Contact loss replaced with sine-wave free-energy matching to give genuine gradient
   signal to K̂(k) (kernel has zero gradient for uniform density).

4. **Phase 3 instability** — `oz_consistency_loss` and `c2_reference_loss` both require
   `compute_c2_bulk` (Hessian-vector product), which returns values ~1e6 (should be ~10)
   for the nonlocal functional. Both excluded; Phase 3 delegates to Phase 2 loss with
   lower LR.

---

## Future Work

- **Multi-objective loss**: balance bulk EOS and interfacial accuracy simultaneously
- **Wall fine-tuning via numerical gradients**: extend the three-phase protocol from the
  hard-sphere work to the nonlocal functional (requires re-deriving kernel Hessian
  normalization for `compute_c2_bulk`)
- **Larger grids**: 64³ for better Fourier mode resolution of K̂(k)
- **Simulation data integration**: use MC/MD wall profiles or interfacial tensions as
  additional training signal to drive C toward −3 (CS) for inhomogeneous targets
- **Mixture extension**: generalize K̂(k) to binary hard-sphere mixtures (most promising
  near-term target — see Tier 1 above)
- **OZ/c₂ consistency**: re-derive the correct Hessian normalization for the nonlocal
  functional to re-enable `c2_reference_loss` in Phase 3
