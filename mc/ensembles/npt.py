"""
NPT (Isobaric-Isothermal) Monte Carlo
======================================

Implements the Metropolis NPT ensemble for both hard-sphere and
soft-potential (LJ, WCA) systems.

Move protocol
-------------
Two types of moves are attempted in a fixed ratio (one volume move per
``vol_freq`` displacement moves):

1. **Displacement move** (same as NVT):
   - Select particle i uniformly at random.
   - Propose r_i' = r_i + Δ·(U − 0.5),  U ~ Uniform(0,1)³.
   - For hard spheres: accept if no overlap.
   - For soft potentials: accept with prob min(1, exp(−β ΔU)).

2. **Volume move** (NPT-specific):
   - Propose V' = V · exp(δ_V · (rand − 0.5))  (log-volume step).
   - Scale all positions: r_i' = r_i · (V'/V)^(1/3).
   - Apply PBC to scaled positions.
   - Acceptance probability:
       acc = min(1, exp(−β [ΔU + P ΔV − (N+1) kT ln(V'/V)]))
   - For hard spheres: ΔU = ∞ if any overlap, else 0.

Acceptance-rate tuning
----------------------
Every ``tune_freq`` displacement (or volume) moves, ``max_disp`` and
``max_vol_step`` are independently adjusted to target 30–40 % acceptance:
    rate > 0.40  →  parameter *= 1.05
    rate < 0.30  →  parameter *= 0.95
Clamped to sane ranges.

Hard-sphere NPT
---------------
Pass ``potential=None`` to use the hard-sphere code path.  No energy is
computed; only overlap checks (O(N²)) are performed.  This gives Z(η) from
the ensemble-averaged density ⟨ρ⟩ = ⟨N/V⟩ at a prescribed pressure P.

    Z_MC = β P / ⟨ρ⟩

References
----------
Frenkel & Smit, *Understanding Molecular Simulation*, 2nd ed., Ch. 5.
Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., Ch. 4.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from mc.core.state import MCState


# ── NumPy overlap helpers (pure NumPy, no JAX) ───────────────────────


def _hs_pair_overlap(positions: np.ndarray, box_length: float, sigma2: float) -> bool:
    """O(N²) check: True if any pair of particles overlaps."""
    N = len(positions)
    for i in range(N - 1):
        dr = positions[i + 1:] - positions[i]           # (N-i-1, 3)
        dr -= box_length * np.round(dr / box_length)
        r2 = np.sum(dr ** 2, axis=1)
        if np.any(r2 < sigma2):
            return True
    return False


def _hs_single_overlap(
    pos_i: np.ndarray,
    positions: np.ndarray,
    skip: int,
    box_length: float,
    sigma2: float,
) -> bool:
    """Check if *pos_i* overlaps any particle except index *skip*."""
    if len(positions) == 0:
        return False
    dr = positions - pos_i                               # (N, 3)
    dr -= box_length * np.round(dr / box_length)
    r2 = np.sum(dr ** 2, axis=1)
    r2[skip] = sigma2 + 1.0                              # exclude self
    return bool(np.any(r2 < sigma2))


# ── NPTSampler ────────────────────────────────────────────────────────


class NPTSampler:
    """Isothermal-isobaric (NPT) Monte Carlo sampler.

    Handles both hard-sphere systems (``potential=None``) and soft-potential
    systems (``potential`` is a ``LennardJones`` or ``WCA`` instance).

    Parameters
    ----------
    mc_state : MCState
        Initial configuration.  Must be a cubic box.
    potential : LennardJones | WCA | None
        Pair potential.  Pass ``None`` for hard spheres.
    beta : float
        Inverse temperature 1/(kT).  For hard spheres kT = 1 → beta = 1.
    pressure : float
        Target pressure P* (reduced units).
    sigma : float
        Hard-sphere diameter (only used when ``potential=None``).
    seed : int
        NumPy random seed.
    """

    def __init__(
        self,
        mc_state: MCState,
        potential=None,
        beta: float = 1.0,
        pressure: float = 1.0,
        sigma: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.positions: np.ndarray = np.array(mc_state.positions, dtype=float)
        self.box_length: float = float(mc_state.box_length)
        self.N: int = mc_state.n_particles
        self.potential = potential
        self.beta = float(beta)
        self.pressure = float(pressure)
        self.sigma = float(sigma)
        self.sigma2 = sigma ** 2
        self.rng = np.random.default_rng(seed)

        # Move counters
        self._disp_acc = 0
        self._disp_att = 0
        self._vol_acc = 0
        self._vol_att = 0

    # ── Energy helpers ────────────────────────────────────────────────

    def _total_energy(self) -> float:
        """Total potential energy (soft systems only)."""
        if self.potential is None:
            return 0.0
        return self.potential.pair_energy(self.positions, self.box_length)

    def _single_energy(self, pos_i: np.ndarray, i: int) -> float:
        """Energy of particle *i* at *pos_i* with all others (soft systems)."""
        if self.potential is None:
            return 0.0
        return self.potential.single_particle_energy(
            pos_i, self.positions, i, self.box_length
        )

    # ── Displacement move ─────────────────────────────────────────────

    def _try_displace(self, max_disp: float) -> None:
        """Attempt one displacement move."""
        self._disp_att += 1
        i = int(self.rng.integers(0, self.N))
        old_pos = self.positions[i].copy()

        delta = max_disp * (self.rng.random(3) - 0.5)
        new_pos = (old_pos + delta) % self.box_length

        if self.potential is None:
            # Hard-sphere path: overlap check only
            if _hs_single_overlap(new_pos, self.positions, i, self.box_length, self.sigma2):
                return
            self.positions[i] = new_pos
            self._disp_acc += 1
        else:
            # Soft-potential path: Metropolis energy criterion
            dU = (
                self.potential.single_particle_energy(new_pos, self.positions, i, self.box_length)
                - self.potential.single_particle_energy(old_pos, self.positions, i, self.box_length)
            )
            if dU <= 0.0 or self.rng.random() < math.exp(-self.beta * dU):
                self.positions[i] = new_pos
                self._disp_acc += 1

    # ── Volume move ───────────────────────────────────────────────────

    def _try_volume(self, max_vol_step: float, U_old: float) -> float:
        """Attempt one volume-scaling move.

        Uses a log-volume step so the volume ratio is tried symmetrically:
            ln(V') = ln(V) + max_vol_step * (rand - 0.5)

        Acceptance criterion (NPT Boltzmann factor):
            acc = min(1, exp(−β [ΔU + P ΔV − (N+1) kT ln(V'/V)]))

        Parameters
        ----------
        max_vol_step : float
            Maximum step size in ln(V).
        U_old : float
            Potential energy before the move (may be 0 for hard spheres).

        Returns
        -------
        float
            Updated total energy after the accepted/rejected move.
        """
        self._vol_att += 1

        V_old = self.box_length ** 3
        lnV_old = math.log(V_old)

        # Propose new log-volume
        lnV_new = lnV_old + max_vol_step * (self.rng.random() - 0.5)
        V_new = math.exp(lnV_new)
        L_new = V_new ** (1.0 / 3.0)

        # Scale all positions uniformly
        scale = L_new / self.box_length
        new_positions = self.positions * scale

        if self.potential is None:
            # Hard-sphere NPT: check all pairs in new box
            if _hs_pair_overlap(new_positions, L_new, self.sigma2):
                return U_old  # reject

            # Acceptance: exp(−β P ΔV + (N+1) ln(V'/V))
            dV = V_new - V_old
            log_acc = (
                -self.beta * self.pressure * dV
                + (self.N + 1) * math.log(V_new / V_old)
            )
        else:
            # Soft-potential NPT
            U_new = self.potential.pair_energy(new_positions, L_new)
            dU = U_new - U_old
            dV = V_new - V_old
            log_acc = (
                -self.beta * (dU + self.pressure * dV)
                + (self.N + 1) * math.log(V_new / V_old)
            )

        # Metropolis decision
        if log_acc >= 0.0 or self.rng.random() < math.exp(log_acc):
            self.positions = new_positions
            self.box_length = L_new
            self._vol_acc += 1
            return 0.0 if self.potential is None else float(U_new)

        return U_old

    # ── Public run method ─────────────────────────────────────────────

    def run(
        self,
        n_equil: int,
        n_prod: int,
        max_disp: float = 0.1,
        max_vol_step: float = 0.05,
        n_sample: int = 10,
        vol_freq: int = 10,
        tune_freq: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """Run NPT Monte Carlo.

        Parameters
        ----------
        n_equil : int
            Number of equilibration sweeps (positions updated but data
            discarded).
        n_prod : int
            Number of production sweeps.
        max_disp : float
            Initial maximum displacement (tuned automatically).
        max_vol_step : float
            Initial maximum ln(V) step (tuned automatically).
        n_sample : int
            Sample observables every *n_sample* production sweeps.
        vol_freq : int
            Attempt one volume move per *vol_freq* displacement moves.
        tune_freq : int
            Tune displacement and volume step sizes every *tune_freq* sweeps.
        verbose : bool
            Print progress every 500 sweeps.

        Returns
        -------
        dict with keys:
            'positions'     : (N, 3) array — final particle positions
            'box_lengths'   : list of sampled box lengths
            'energies'      : list of sampled total energies (0 for HS)
            'densities'     : list of sampled number densities N/V
            'mean_density'  : float — mean density ⟨N/V⟩
            'mean_energy'   : float — mean total energy ⟨U⟩ (0 for HS)
            'acc_disp'      : displacement acceptance rate
            'acc_vol'       : volume acceptance rate
        """
        disp = float(max_disp)
        vol_step = float(max_vol_step)

        # Current energy (0 for HS)
        U = self._total_energy()

        def _one_sweep() -> None:
            nonlocal U
            for _ in range(self.N):
                self._try_displace(disp)
            self._try_volume(vol_step, U)
            # Re-sync energy for soft potentials (avoids drift from float errors)
            U = self._total_energy()

        def _tune() -> None:
            nonlocal disp, vol_step
            # Displacement tuning
            if self._disp_att > 0:
                rate_d = self._disp_acc / self._disp_att
                if rate_d > 0.40:
                    disp *= 1.05
                elif rate_d < 0.30:
                    disp *= 0.95
                disp = float(np.clip(disp, 0.001, self.box_length / 2.0))

            # Volume tuning
            if self._vol_att > 0:
                rate_v = self._vol_acc / self._vol_att
                if rate_v > 0.40:
                    vol_step *= 1.05
                elif rate_v < 0.30:
                    vol_step *= 0.95
                vol_step = float(np.clip(vol_step, 1e-5, 2.0))

            # Reset counters
            self._disp_acc = self._disp_att = 0
            self._vol_acc = self._vol_att = 0

        # ── Equilibration ─────────────────────────────────────────────
        for sweep in range(n_equil):
            _one_sweep()
            if (sweep + 1) % tune_freq == 0:
                _tune()
            if verbose and (sweep + 1) % 500 == 0:
                rho = self.N / (self.box_length ** 3)
                print(
                    f"  [equil] sweep {sweep+1:6d}/{n_equil}  "
                    f"ρ={rho:.4f}  L={self.box_length:.4f}  "
                    f"disp={disp:.4f}  vol={vol_step:.4f}"
                )

        # Reset counters for production
        self._disp_acc = self._disp_att = 0
        self._vol_acc = self._vol_att = 0

        # ── Production ────────────────────────────────────────────────
        box_lengths: List[float] = []
        energies: List[float] = []
        densities: List[float] = []

        for sweep in range(n_prod):
            _one_sweep()
            if (sweep + 1) % tune_freq == 0:
                _tune()

            if (sweep + 1) % n_sample == 0:
                rho = self.N / (self.box_length ** 3)
                box_lengths.append(float(self.box_length))
                energies.append(float(U))
                densities.append(float(rho))

            if verbose and (sweep + 1) % 500 == 0:
                rho = self.N / (self.box_length ** 3)
                print(
                    f"  [prod]  sweep {sweep+1:6d}/{n_prod}  "
                    f"ρ={rho:.4f}  L={self.box_length:.4f}  "
                    f"acc_d={self._disp_acc/max(self._disp_att,1):.3f}  "
                    f"acc_v={self._vol_acc/max(self._vol_att,1):.3f}"
                )

        densities_arr = np.array(densities)
        energies_arr = np.array(energies)

        return {
            "positions": self.positions.copy(),
            "box_lengths": box_lengths,
            "energies": energies_arr,
            "densities": densities_arr,
            "mean_density": float(np.mean(densities_arr)) if len(densities_arr) > 0 else 0.0,
            "mean_energy": float(np.mean(energies_arr)) if len(energies_arr) > 0 else 0.0,
            "acc_disp": self._disp_acc / max(self._disp_att, 1),
            "acc_vol": self._vol_acc / max(self._vol_att, 1),
        }

    # ── Current state accessor ────────────────────────────────────────

    @property
    def state(self) -> MCState:
        """Return current simulation state as an MCState."""
        import jax.numpy as jnp
        return MCState(
            positions=jnp.array(self.positions),
            box_length=self.box_length,
            n_particles=self.N,
        )
