"""
Curriculum Trainer
==================

Four-phase curriculum training for the nonlocal neural functional.

The key idea: train exclusively through physics constraints (no simulation
data) by introducing constraints in order of computational cost and
difficulty.

Phases
------
Phase 1 (bulk): Match Z, μ, χ to Carnahan-Starling at expanding η range.
Phase 2 (contact): Add contact sum rule via 1D wall profiles (Picard).
Phase 3 (OZ): Add Ornstein-Zernike consistency (c₂ vs PY).
Phase 4 (all): Add Noether invariance + fine-tune all losses.

Each phase builds on the previous, and checkpoints are saved between
phases so training can be resumed.

Reference
---------
Training protocol described in Next_step/NEXT_STEPS.md §4.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from core.grid import Grid
from core.weights import FMTKernels
from core.densities import WeightedDensityCalculator
from core.thermodynamics import BulkThermodynamics
from nonlocal_ext.functional import NonlocalLutskoFunctional
from nonlocal_ext.kernels import LearnableKernel
from neural.network import NonlocalConditionalNetwork
from solvers.fmt_1d_wbii_tensor import WallSolver, esFMT_Tensor
from constraints.sum_rules import contact_sum_rule_loss
from constraints.noether import translational_invariance_loss
from constraints.scaled_particle import (
    low_density_limit_loss, close_packing_limit_loss,
    spt_exact_relations_loss, positivity_loss,
)
from constraints.consistency import oz_consistency_loss, c2_reference_loss
from training.losses import compute_nonlocal_bulk_loss
from training.config import TrainingConfig


# ═════════════════════════════════════════════════════════════════════
# CURRICULUM CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumConfig:
    """Configuration for four-phase curriculum training.

    Parameters
    ----------
    n_epochs_phase1 : int
        Bulk thermodynamics epochs.
    n_epochs_phase2 : int
        Contact sum rule epochs.
    n_epochs_phase3 : int
        OZ consistency epochs.
    n_epochs_phase4 : int
        Noether + fine-tune epochs.
    eta_range_initial : tuple
        (low, high) packing fraction for Phase 1 start.
    eta_range_final : tuple
        (low, high) packing fraction expanded to by end of Phase 1.
    n_eta_train : int
        Number of training η values sampled each epoch.
    eta_wall_profiles : list
        Packing fractions for wall profile calculations.
    lr_phase1 : float
        Learning rate for Phase 1.
    lr_phase2 : float
        Learning rate for Phase 2.
    lr_phase3 : float
        Learning rate for Phase 3.
    lr_phase4 : float
        Learning rate for Phase 4 (cosine annealing from this value).
    grad_clip : float
        Gradient clipping norm.
    solver_nz : int
        1D solver grid points.
    solver_Lz : float
        1D solver box length (in σ).
    solver_max_iter : int
        Picard iteration limit.
    checkpoint_dir : str
        Directory for saving checkpoints.
    log_every : int
        Print frequency (epochs).
    """
    # Phase durations
    n_epochs_phase1: int = 200
    n_epochs_phase2: int = 300
    n_epochs_phase3: int = 300
    n_epochs_phase4: int = 200

    # Packing fraction schedule
    eta_range_initial: Tuple[float, float] = (0.01, 0.30)
    eta_range_final: Tuple[float, float] = (0.01, 0.52)
    n_eta_train: int = 30
    eta_wall_profiles: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.367, 0.393, 0.449]
    )

    # Learning rates per phase
    lr_phase1: float = 3e-3
    lr_phase2: float = 1e-3
    lr_phase3: float = 5e-4
    lr_phase4: float = 2e-4

    # Optimiser
    grad_clip: float = 1.0
    weight_decay: float = 1e-4

    # 1D solver
    solver_nz: int = 2048
    solver_Lz: float = 6.0
    solver_max_iter: int = 5000

    # OZ settings
    oz_eta_values: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4]
    )

    # Loss weights (overrides for each phase)
    w_contact: float = 1.0
    w_oz: float = 0.5
    w_c2_ref: float = 0.3
    w_noether: float = 0.1
    w_spt: float = 0.5
    w_positivity: float = 1.0

    # Infrastructure
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10
    save_every: int = 50


# ═════════════════════════════════════════════════════════════════════
# HELPER: BULK PARAMETER WRAPPER
# ═════════════════════════════════════════════════════════════════════

class _BulkParamWrapper:
    """Adapter so SPT losses can call .from_eta(eta) on a functional."""

    def __init__(self, functional):
        self._func = functional

    def from_eta(self, eta_val):
        return self._func.bulk_parameters(eta_val)


# ═════════════════════════════════════════════════════════════════════
# HELPER: SOLVE WALL PROFILES
# ═════════════════════════════════════════════════════════════════════

def solve_wall_profiles(
    functional: NonlocalLutskoFunctional,
    eta_values: List[float],
    solver_nz: int = 2048,
    solver_Lz: float = 6.0,
    solver_max_iter: int = 5000,
    verbose: bool = False,
) -> List[Dict]:
    """Solve 1D wall density profiles at given packing fractions.

    Uses the esFMT_Tensor solver with A, B from the functional's bulk
    parameters at each η.  This is a non-differentiable step — profiles
    are computed as fixed targets, not differentiated through.

    Returns list of dicts with keys: z, rho, eta_bulk, converged.
    """
    solver = WallSolver(nz=solver_nz, Lz=solver_Lz)
    results = []

    for eta in eta_values:
        A, B = functional.bulk_parameters(eta)
        fmt = esFMT_Tensor(A=float(A), B=float(B))
        result = solver.solve(eta, fmt, max_iter=solver_max_iter,
                              verbose=verbose)
        results.append(result)

    return results


# ═════════════════════════════════════════════════════════════════════
# CURRICULUM TRAINER
# ═════════════════════════════════════════════════════════════════════

class CurriculumTrainer:
    """Four-phase curriculum training for the nonlocal functional.

    Phase 1 (epochs 0 — n1): Bulk thermodynamics only
        Match Z, μ, χ to Carnahan-Starling.
        η range gradually expands from initial to final.

    Phase 2 (epochs n1 — n1+n2): Add contact sum rule
        Solve wall profiles with Picard iteration.
        Add contact density loss at wall profiles η values.

    Phase 3 (epochs n1+n2 — n1+n2+n3): Add OZ consistency
        Compute c₂ via autodiff.
        Compare g(r) constraints from OZ equation.
        Optionally compare c₂ to PY reference.

    Phase 4 (epochs n1+n2+n3 — total): Add Noether + fine-tune
        Translational invariance.
        All losses active with cosine annealing LR.

    Parameters
    ----------
    functional : NonlocalLutskoFunctional
        The nonlocal functional to train.
    grid : Grid
        3D computational grid.
    config : CurriculumConfig
        Curriculum training configuration.
    training_config : TrainingConfig
        Base training config (for loss weights).
    """

    def __init__(
        self,
        functional: NonlocalLutskoFunctional,
        grid: Grid,
        config: CurriculumConfig,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.functional = functional
        self.grid = grid
        self.config = config
        self.tc = training_config or TrainingConfig()

        # History
        self.loss_history: List[Dict] = []
        self.phase_boundaries: List[int] = []

    # ─────────────────────────────────────────────────────────────
    # OPTIMISER CREATION
    # ─────────────────────────────────────────────────────────────

    def _create_optimizer(self, lr: float) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(lr, weight_decay=self.config.weight_decay),
        )

    def _create_cosine_optimizer(
        self, lr_init: float, n_epochs: int
    ) -> optax.GradientTransformation:
        schedule = optax.cosine_decay_schedule(
            init_value=lr_init, decay_steps=max(n_epochs, 1), alpha=0.01
        )
        return optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(schedule, weight_decay=self.config.weight_decay),
        )

    # ─────────────────────────────────────────────────────────────
    # ETA SCHEDULE
    # ─────────────────────────────────────────────────────────────

    def _sample_eta(self, epoch: int, n_phase1: int, key) -> jnp.ndarray:
        """Sample η values with expanding range during Phase 1."""
        lo_i, hi_i = self.config.eta_range_initial
        lo_f, hi_f = self.config.eta_range_final
        progress = jnp.clip(epoch / max(n_phase1 - 1, 1), 0.0, 1.0)
        lo = lo_i + (lo_f - lo_i) * progress
        hi = hi_i + (hi_f - hi_i) * progress
        return jax.random.uniform(
            key, (self.config.n_eta_train,), minval=lo, maxval=hi
        )

    # ─────────────────────────────────────────────────────────────
    # PHASE 1: BULK THERMODYNAMICS
    # ─────────────────────────────────────────────────────────────

    def _phase1_loss(self, functional, eta_values):
        """Bulk EOS + SPT losses."""
        bulk = compute_nonlocal_bulk_loss(
            functional, eta_values, self.grid, self.tc
        )
        wrapper = _BulkParamWrapper(functional)
        spt = (low_density_limit_loss(wrapper)
               + close_packing_limit_loss(wrapper)
               + spt_exact_relations_loss(wrapper))
        return bulk + self.config.w_spt * spt

    def train_phase1(self, verbose: bool = True) -> None:
        """Phase 1: Bulk thermodynamics matching."""
        cfg = self.config
        n_epochs = cfg.n_epochs_phase1
        if n_epochs <= 0:
            self.phase_boundaries.append(len(self.loss_history))
            return

        if verbose:
            print("\n" + "=" * 65)
            print("  PHASE 1: Bulk Thermodynamics")
            print(f"  Epochs: {n_epochs}, LR: {cfg.lr_phase1}")
            print(f"  η range: {cfg.eta_range_initial} → {cfg.eta_range_final}")
            print("=" * 65)

        optimizer = self._create_optimizer(cfg.lr_phase1)
        opt_state = optimizer.init(eqx.filter(self.functional, eqx.is_array))
        best_loss = float("inf")
        best_func = self.functional

        for epoch in range(n_epochs):
            key = jax.random.PRNGKey(epoch)
            eta_vals = self._sample_eta(epoch, n_epochs, key)

            @eqx.filter_value_and_grad
            def loss_fn(func):
                return self._phase1_loss(func, eta_vals)

            loss, grads = loss_fn(self.functional)
            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array), opt_state,
                eqx.filter(self.functional, eqx.is_array),
            )
            self.functional = eqx.apply_updates(self.functional, updates)

            loss_val = float(loss)
            self.loss_history.append({"epoch": epoch, "phase": 1,
                                      "loss": loss_val})

            if loss_val < best_loss:
                best_loss = loss_val
                best_func = self.functional

            if verbose and (epoch % cfg.log_every == 0
                            or epoch == n_epochs - 1):
                A, B = self.functional.bulk_parameters(0.3)
                C = 8 * float(A) + 2 * float(B) - 9
                print(f"  [{epoch:4d}/{n_epochs}] loss={loss_val:10.4e} "
                      f"A(0.3)={float(A):.4f} B(0.3)={float(B):.4f} "
                      f"C={C:.4f}")

        self.functional = best_func
        self.phase_boundaries.append(len(self.loss_history))
        self._save_checkpoint("phase1")
        if verbose:
            print(f"  Phase 1 complete — best loss: {best_loss:.4e}")

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: CONTACT SUM RULE
    # ─────────────────────────────────────────────────────────────

    def _phase2_loss(self, functional, eta_values, wall_results, key):
        """Bulk + contact sum rule + SPT + positivity."""
        bulk = compute_nonlocal_bulk_loss(
            functional, eta_values, self.grid, self.tc
        )

        # Contact sum rule from wall profiles
        contact_loss = 0.0
        for res in wall_results:
            eta = res["eta_bulk"]
            z = jnp.array(res["z"])
            rho = jnp.array(res["rho"])
            contact_loss += contact_sum_rule_loss(rho, z, eta)
        contact_loss /= max(len(wall_results), 1)

        # SPT
        wrapper = _BulkParamWrapper(functional)
        spt = (low_density_limit_loss(wrapper)
               + close_packing_limit_loss(wrapper)
               + spt_exact_relations_loss(wrapper))

        # Positivity at a mid-range η
        rho_bulk = 6.0 * 0.3 / jnp.pi
        rho_uniform = jnp.ones(
            (self.grid.nx, self.grid.ny, self.grid.nz)
        ) * rho_bulk
        pos = positivity_loss(functional, rho_uniform, self.grid)

        return (bulk
                + self.config.w_contact * contact_loss
                + self.config.w_spt * spt
                + self.config.w_positivity * pos)

    def train_phase2(self, verbose: bool = True) -> None:
        """Phase 2: Add contact sum rule."""
        cfg = self.config
        n_epochs = cfg.n_epochs_phase2
        if n_epochs <= 0:
            self.phase_boundaries.append(len(self.loss_history))
            return

        if verbose:
            print("\n" + "=" * 65)
            print("  PHASE 2: Contact Sum Rule")
            print(f"  Epochs: {n_epochs}, LR: {cfg.lr_phase2}")
            print(f"  Wall η: {cfg.eta_wall_profiles}")
            print("=" * 65)

        # Solve initial wall profiles
        if verbose:
            print("  Solving wall profiles...")
        wall_results = solve_wall_profiles(
            self.functional, cfg.eta_wall_profiles,
            cfg.solver_nz, cfg.solver_Lz, cfg.solver_max_iter,
        )
        n_converged = sum(1 for r in wall_results if r["converged"])
        if verbose:
            print(f"  Converged: {n_converged}/{len(wall_results)}")

        optimizer = self._create_optimizer(cfg.lr_phase2)
        opt_state = optimizer.init(eqx.filter(self.functional, eqx.is_array))
        best_loss = float("inf")
        best_func = self.functional

        # Full η range for bulk loss
        eta_full = jnp.linspace(
            cfg.eta_range_final[0], cfg.eta_range_final[1],
            cfg.n_eta_train,
        )

        for epoch in range(n_epochs):
            key = jax.random.PRNGKey(cfg.n_epochs_phase1 + epoch)

            @eqx.filter_value_and_grad
            def loss_fn(func):
                return self._phase2_loss(func, eta_full, wall_results, key)

            loss, grads = loss_fn(self.functional)
            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array), opt_state,
                eqx.filter(self.functional, eqx.is_array),
            )
            self.functional = eqx.apply_updates(self.functional, updates)

            loss_val = float(loss)
            self.loss_history.append({"epoch": epoch, "phase": 2,
                                      "loss": loss_val})

            if loss_val < best_loss:
                best_loss = loss_val
                best_func = self.functional

            # Re-solve profiles periodically
            if (epoch + 1) % cfg.save_every == 0 and epoch < n_epochs - 1:
                if verbose:
                    print("  Re-solving wall profiles...")
                wall_results = solve_wall_profiles(
                    self.functional, cfg.eta_wall_profiles,
                    cfg.solver_nz, cfg.solver_Lz, cfg.solver_max_iter,
                )

            if verbose and (epoch % cfg.log_every == 0
                            or epoch == n_epochs - 1):
                # Report contact density accuracy
                contact_errs = []
                for res in wall_results:
                    if res["converged"]:
                        contact_errs.append(
                            abs(res["contact"] / res["contact_CS"] - 1.0)
                        )
                mean_err = sum(contact_errs) / max(len(contact_errs), 1)
                print(f"  [{epoch:4d}/{n_epochs}] loss={loss_val:10.4e} "
                      f"contact_err={mean_err:.4e}")

        self.functional = best_func
        self.phase_boundaries.append(len(self.loss_history))
        self._save_checkpoint("phase2")
        if verbose:
            print(f"  Phase 2 complete — best loss: {best_loss:.4e}")

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: OZ CONSISTENCY
    # ─────────────────────────────────────────────────────────────

    def _phase3_loss(self, functional, eta_values, wall_results, key):
        """Bulk + contact + OZ consistency + c₂ reference."""
        # Reuse Phase 2 loss components
        base = self._phase2_loss(functional, eta_values, wall_results, key)

        # OZ consistency at selected η values
        oz_loss = 0.0
        for eta in self.config.oz_eta_values:
            rho_bulk = 6.0 * eta / jnp.pi
            oz_loss += oz_consistency_loss(
                functional, rho_bulk, self.grid
            )
        oz_loss /= max(len(self.config.oz_eta_values), 1)

        # c₂ vs PY reference
        c2_loss = 0.0
        for eta in self.config.oz_eta_values:
            rho_bulk = 6.0 * eta / jnp.pi
            c2_loss += c2_reference_loss(
                functional, rho_bulk, self.grid, eta
            )
        c2_loss /= max(len(self.config.oz_eta_values), 1)

        return (base
                + self.config.w_oz * oz_loss
                + self.config.w_c2_ref * c2_loss)

    def train_phase3(self, verbose: bool = True) -> None:
        """Phase 3: Add OZ consistency."""
        cfg = self.config
        n_epochs = cfg.n_epochs_phase3
        if n_epochs <= 0:
            self.phase_boundaries.append(len(self.loss_history))
            return

        if verbose:
            print("\n" + "=" * 65)
            print("  PHASE 3: OZ Consistency")
            print(f"  Epochs: {n_epochs}, LR: {cfg.lr_phase3}")
            print(f"  OZ η: {cfg.oz_eta_values}")
            print("=" * 65)

        # Get wall profiles from current functional
        if verbose:
            print("  Solving wall profiles...")
        wall_results = solve_wall_profiles(
            self.functional, cfg.eta_wall_profiles,
            cfg.solver_nz, cfg.solver_Lz, cfg.solver_max_iter,
        )

        optimizer = self._create_optimizer(cfg.lr_phase3)
        opt_state = optimizer.init(eqx.filter(self.functional, eqx.is_array))
        best_loss = float("inf")
        best_func = self.functional

        eta_full = jnp.linspace(
            cfg.eta_range_final[0], cfg.eta_range_final[1],
            cfg.n_eta_train,
        )

        for epoch in range(n_epochs):
            key = jax.random.PRNGKey(
                cfg.n_epochs_phase1 + cfg.n_epochs_phase2 + epoch
            )

            @eqx.filter_value_and_grad
            def loss_fn(func):
                return self._phase3_loss(func, eta_full, wall_results, key)

            loss, grads = loss_fn(self.functional)
            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array), opt_state,
                eqx.filter(self.functional, eqx.is_array),
            )
            self.functional = eqx.apply_updates(self.functional, updates)

            loss_val = float(loss)
            self.loss_history.append({"epoch": epoch, "phase": 3,
                                      "loss": loss_val})

            if loss_val < best_loss:
                best_loss = loss_val
                best_func = self.functional

            # Re-solve profiles periodically
            if (epoch + 1) % cfg.save_every == 0 and epoch < n_epochs - 1:
                if verbose:
                    print("  Re-solving wall profiles...")
                wall_results = solve_wall_profiles(
                    self.functional, cfg.eta_wall_profiles,
                    cfg.solver_nz, cfg.solver_Lz, cfg.solver_max_iter,
                )

            if verbose and (epoch % cfg.log_every == 0
                            or epoch == n_epochs - 1):
                print(f"  [{epoch:4d}/{n_epochs}] loss={loss_val:10.4e}")

        self.functional = best_func
        self.phase_boundaries.append(len(self.loss_history))
        self._save_checkpoint("phase3")
        if verbose:
            print(f"  Phase 3 complete — best loss: {best_loss:.4e}")

    # ─────────────────────────────────────────────────────────────
    # PHASE 4: NOETHER + FINE-TUNE
    # ─────────────────────────────────────────────────────────────

    def _phase4_loss(self, functional, eta_values, wall_results, key):
        """All losses: bulk + contact + OZ + Noether."""
        base = self._phase3_loss(functional, eta_values, wall_results, key)

        # Noether translational invariance
        rho_bulk = 6.0 * 0.3 / jnp.pi
        rho_uniform = jnp.ones(
            (self.grid.nx, self.grid.ny, self.grid.nz)
        ) * rho_bulk
        noether = translational_invariance_loss(
            functional, rho_uniform, self.grid, n_shifts=5, key=key,
        )

        return base + self.config.w_noether * noether

    def train_phase4(self, verbose: bool = True) -> None:
        """Phase 4: Add Noether + fine-tune all losses with cosine LR."""
        cfg = self.config
        n_epochs = cfg.n_epochs_phase4
        if n_epochs <= 0:
            self.phase_boundaries.append(len(self.loss_history))
            return

        if verbose:
            print("\n" + "=" * 65)
            print("  PHASE 4: Noether + Fine-Tune (Cosine Annealing)")
            print(f"  Epochs: {n_epochs}, LR: {cfg.lr_phase4} → 0")
            print("=" * 65)

        # Solve wall profiles
        if verbose:
            print("  Solving wall profiles...")
        wall_results = solve_wall_profiles(
            self.functional, cfg.eta_wall_profiles,
            cfg.solver_nz, cfg.solver_Lz, cfg.solver_max_iter,
        )

        optimizer = self._create_cosine_optimizer(cfg.lr_phase4, n_epochs)
        opt_state = optimizer.init(eqx.filter(self.functional, eqx.is_array))
        best_loss = float("inf")
        best_func = self.functional

        eta_full = jnp.linspace(
            cfg.eta_range_final[0], cfg.eta_range_final[1],
            cfg.n_eta_train,
        )

        for epoch in range(n_epochs):
            offset = (cfg.n_epochs_phase1 + cfg.n_epochs_phase2
                      + cfg.n_epochs_phase3 + epoch)
            key = jax.random.PRNGKey(offset)

            @eqx.filter_value_and_grad
            def loss_fn(func):
                return self._phase4_loss(func, eta_full, wall_results, key)

            loss, grads = loss_fn(self.functional)
            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array), opt_state,
                eqx.filter(self.functional, eqx.is_array),
            )
            self.functional = eqx.apply_updates(self.functional, updates)

            loss_val = float(loss)
            self.loss_history.append({"epoch": epoch, "phase": 4,
                                      "loss": loss_val})

            if loss_val < best_loss:
                best_loss = loss_val
                best_func = self.functional

            if verbose and (epoch % cfg.log_every == 0
                            or epoch == n_epochs - 1):
                A, B = self.functional.bulk_parameters(0.4)
                C = 8 * float(A) + 2 * float(B) - 9
                print(f"  [{epoch:4d}/{n_epochs}] loss={loss_val:10.4e} "
                      f"C(0.4)={C:.4f}")

        self.functional = best_func
        self.phase_boundaries.append(len(self.loss_history))
        self._save_checkpoint("final")
        if verbose:
            print(f"  Phase 4 complete — best loss: {best_loss:.4e}")

    # ─────────────────────────────────────────────────────────────
    # CHECKPOINT
    # ─────────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str) -> Path:
        """Save functional checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(exist_ok=True)
        filepath = ckpt_dir / f"nonlocal_{name}.eqx"
        eqx.tree_serialise_leaves(filepath, self.functional)
        print(f"  Saved checkpoint: {filepath}")
        return filepath

    def load_checkpoint(self, name: str) -> None:
        """Load functional from checkpoint."""
        filepath = Path(self.config.checkpoint_dir) / f"nonlocal_{name}.eqx"
        self.functional = eqx.tree_deserialise_leaves(
            filepath, self.functional
        )
        print(f"  Loaded checkpoint: {filepath}")

    # ─────────────────────────────────────────────────────────────
    # FULL TRAINING
    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        start_phase: int = 1,
        verbose: bool = True,
    ) -> NonlocalLutskoFunctional:
        """Run full curriculum training.

        Parameters
        ----------
        start_phase : int
            Phase to start from (1-4). Use >1 to resume after loading
            a checkpoint.
        verbose : bool
            Print progress.

        Returns
        -------
        functional : NonlocalLutskoFunctional
            Trained functional.
        """
        t0 = time.time()
        if verbose:
            total = (self.config.n_epochs_phase1 + self.config.n_epochs_phase2
                     + self.config.n_epochs_phase3 + self.config.n_epochs_phase4)
            print("\n" + "═" * 65)
            print("  CURRICULUM TRAINING: Nonlocal Neural Functional")
            print(f"  Total epochs: {total}")
            print(f"  Phases: {self.config.n_epochs_phase1} + "
                  f"{self.config.n_epochs_phase2} + "
                  f"{self.config.n_epochs_phase3} + "
                  f"{self.config.n_epochs_phase4}")
            print("═" * 65)

        if start_phase <= 1:
            self.train_phase1(verbose)
        if start_phase <= 2:
            self.train_phase2(verbose)
        if start_phase <= 3:
            self.train_phase3(verbose)
        if start_phase <= 4:
            self.train_phase4(verbose)

        elapsed = time.time() - t0
        if verbose:
            print("\n" + "═" * 65)
            print(f"  Training complete in {elapsed:.1f}s")
            print(f"  Final best loss: {self.loss_history[-1]['loss']:.4e}")
            print("═" * 65)

        return self.functional

    # ─────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────

    def get_loss_by_phase(self) -> Dict[int, List[float]]:
        """Return loss history grouped by phase."""
        result = {}
        for entry in self.loss_history:
            phase = entry["phase"]
            if phase not in result:
                result[phase] = []
            result[phase].append(entry["loss"])
        return result

    def report(self) -> None:
        """Print summary of training results."""
        by_phase = self.get_loss_by_phase()
        print("\n  Training Summary")
        print("  " + "-" * 40)
        for phase, losses in sorted(by_phase.items()):
            print(f"  Phase {phase}: {len(losses)} epochs, "
                  f"final={losses[-1]:.4e}, best={min(losses):.4e}")

        # Print final bulk parameters
        print("\n  Final bulk parameters:")
        for eta in [0.1, 0.2, 0.3, 0.4, 0.45]:
            A, B = self.functional.bulk_parameters(eta)
            C = 8 * float(A) + 2 * float(B) - 9
            Z_lut = BulkThermodynamics.Z_lutsko(eta, A, B)
            Z_cs = BulkThermodynamics.Z_CS(eta)
            err = abs(float(Z_lut / Z_cs - 1))
            print(f"  η={eta:.2f}: A={float(A):.4f} B={float(B):.4f} "
                  f"C={C:+.4f} Z_err={err:.2e}")
