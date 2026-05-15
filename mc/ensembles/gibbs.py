"""
Gibbs Ensemble Monte Carlo (Panagiotopoulos 1987).

Two simulation boxes at fixed total N and V, same T and μ.
At equilibrium: P₁ = P₂, μ₁ = μ₂, T₁ = T₂.

Moves:
  1. Displacement in box 1 or 2 (NVT-style)
  2. Volume exchange: ΔV transferred from box 1 to box 2
     Accept with: min(1, (V1_new/V1_old)^N1 * (V2_new/V2_old)^N2 * exp(-β*ΔU))
  3. Particle transfer: move particle from box 1 to box 2 (or vice versa)
     Accept with: min(1, (N1*V2/(N2+1)/V1) * exp(-β*ΔU_insert - β*ΔU_delete))

At coexistence: one box → liquid (high ρ), other → vapor (low ρ).

References
----------
Panagiotopoulos, A. Z. (1987). Direct determination of phase coexistence
properties of fluids by Monte Carlo simulation in a new ensemble.
Molecular Physics, 61(4), 813–826.

Frenkel & Smit, Understanding Molecular Simulation, 2nd ed., Ch. 8.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from mc.core.state import MCState


# ── NumPy helpers (pure NumPy, no JAX) ───────────────────────────────


def _single_energy_np(
    pot,
    pos_i: np.ndarray,
    positions: np.ndarray,
    skip: int,
    box_length: float,
) -> float:
    """LJ energy of pos_i with all particles in *positions* except index *skip*."""
    if len(positions) == 0:
        return 0.0
    rc2 = pot.r_cut_abs ** 2
    sigma2 = pot.sigma ** 2
    dr = positions - pos_i                               # (N, 3)
    dr -= box_length * np.round(dr / box_length)
    r2 = np.sum(dr ** 2, axis=1)                        # (N,)
    mask = (r2 < rc2)
    mask[skip] = False
    if not np.any(mask):
        return 0.0
    inv_r2 = sigma2 / r2[mask]
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return float(np.sum(4.0 * pot.epsilon * (inv_r12 - inv_r6) - pot._u_shift))


def _insertion_energy_np(
    pot,
    pos_i: np.ndarray,
    positions: np.ndarray,
    box_length: float,
) -> float:
    """LJ energy of inserting pos_i into a box with *positions* (no exclusion)."""
    if len(positions) == 0:
        return 0.0
    rc2 = pot.r_cut_abs ** 2
    sigma2 = pot.sigma ** 2
    dr = positions - pos_i                               # (N, 3)
    dr -= box_length * np.round(dr / box_length)
    r2 = np.sum(dr ** 2, axis=1)                        # (N,)
    mask = r2 < rc2
    if not np.any(mask):
        return 0.0
    inv_r2 = sigma2 / r2[mask]
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return float(np.sum(4.0 * pot.epsilon * (inv_r12 - inv_r6) - pot._u_shift))


def _total_energy_np(pot, positions: np.ndarray, box_length: float) -> float:
    """Total LJ pair energy (O(N²))."""
    return pot.pair_energy(positions, box_length)


# ── GibbsSampler ──────────────────────────────────────────────────────


class GibbsSampler:
    """Gibbs Ensemble Monte Carlo for liquid-gas coexistence.

    Two simulation boxes at fixed total particle number N_total = N1 + N2
    and fixed total volume V_total = V1 + V2, at the same temperature T.
    At equilibrium the two boxes coexist as liquid and vapor phases.

    Three types of Monte Carlo moves are attempted in a fixed ratio:
        - 0.50  displacement moves (NVT-like, within each box)
        - 0.10  volume exchange moves (V₁↔V₂, conserving V_total)
        - 0.40  particle transfer moves (N₁↔N₂, conserving N_total)

    Parameters
    ----------
    state1 : MCState
        Initial configuration of box 1 (typically the liquid guess).
    state2 : MCState
        Initial configuration of box 2 (typically the vapor guess).
    potential : LennardJones
        Pair potential (must have ``single_particle_energy`` and
        ``pair_energy`` methods).
    beta : float
        Inverse temperature β = 1/(kT).  For reduced LJ units with ε=1,
        β = 1/T*.
    seed : int
        NumPy random seed.
    """

    def __init__(
        self,
        state1: MCState,
        state2: MCState,
        potential,
        beta: float = 1.0,
        seed: int = 0,
    ) -> None:
        # Box 1
        self.pos1: np.ndarray = np.array(state1.positions, dtype=float)
        self.L1: float = float(state1.box_length)
        self.N1: int = state1.n_particles

        # Box 2
        self.pos2: np.ndarray = np.array(state2.positions, dtype=float)
        self.L2: float = float(state2.box_length)
        self.N2: int = state2.n_particles

        self.potential = potential
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)

        # Conserved totals
        self._N_total = self.N1 + self.N2
        self._V_total = self.L1 ** 3 + self.L2 ** 3

        # Move counters (displacement, volume, transfer)
        self._disp_acc = [0, 0]
        self._disp_att = [0, 0]
        self._vol_acc = 0
        self._vol_att = 0
        self._xfer_acc = 0
        self._xfer_att = 0

    # ── Properties ───────────────────────────────────────────────────

    @property
    def rho1(self) -> float:
        return self.N1 / self.L1 ** 3

    @property
    def rho2(self) -> float:
        return self.N2 / self.L2 ** 3

    # ── Energy helpers ────────────────────────────────────────────────

    def _U1(self) -> float:
        return _total_energy_np(self.potential, self.pos1, self.L1)

    def _U2(self) -> float:
        return _total_energy_np(self.potential, self.pos2, self.L2)

    # ── Displacement move ─────────────────────────────────────────────

    def _try_displace(self, box: int, max_disp: float) -> None:
        """Attempt one Metropolis displacement in box *box* (0 or 1)."""
        if box == 0:
            if self.N1 == 0:
                return
            pos = self.pos1
            L = self.L1
            N = self.N1
            box_idx = 0
        else:
            if self.N2 == 0:
                return
            pos = self.pos2
            L = self.L2
            N = self.N2
            box_idx = 1

        self._disp_att[box_idx] += 1
        i = int(self.rng.integers(0, N))
        old_pos = pos[i].copy()
        delta = max_disp * (self.rng.random(3) - 0.5)
        new_pos = (old_pos + delta) % L

        u_old = _single_energy_np(self.potential, old_pos, pos, i, L)
        u_new = _single_energy_np(self.potential, new_pos, pos, i, L)
        dU = u_new - u_old

        if dU <= 0.0 or self.rng.random() < math.exp(-self.beta * dU):
            pos[i] = new_pos
            self._disp_acc[box_idx] += 1

    # ── Volume exchange move ──────────────────────────────────────────

    def _try_volume(self, max_vol_step: float, U1: float, U2: float) -> Tuple[float, float]:
        """Attempt a volume exchange move.

        Propose transfer of δV from box 1 to box 2 (or vice versa) by
        drawing a log-volume step for box 1:
            ln(V1') = ln(V1) + max_vol_step * (rand - 0.5)
        then setting V2' = V_total - V1'.

        Acceptance criterion (Panagiotopoulos 1987):
            acc = min(1, (V1'/V1)^N1 * (V2'/V2)^N2 * exp(-β ΔU))

        Returns
        -------
        (U1_new, U2_new) : updated energies after accepted/rejected move.
        """
        self._vol_att += 1

        V1_old = self.L1 ** 3
        V2_old = self.L2 ** 3

        # Propose new V1 via log-volume step, keep V_total fixed
        lnV1_old = math.log(V1_old)
        lnV1_new = lnV1_old + max_vol_step * (self.rng.random() - 0.5)
        V1_new = math.exp(lnV1_new)
        V2_new = self._V_total - V1_new

        if V2_new <= 0.0:
            return U1, U2

        L1_new = V1_new ** (1.0 / 3.0)
        L2_new = V2_new ** (1.0 / 3.0)

        # Scale positions
        scale1 = L1_new / self.L1
        scale2 = L2_new / self.L2
        pos1_new = self.pos1 * scale1
        pos2_new = self.pos2 * scale2

        # Compute energies after scaling
        U1_new = _total_energy_np(self.potential, pos1_new, L1_new)
        U2_new = _total_energy_np(self.potential, pos2_new, L2_new)
        dU = (U1_new + U2_new) - (U1 + U2)

        # Acceptance (Eq. 9 in Panagiotopoulos 1987)
        log_acc = (
            self.N1 * math.log(V1_new / V1_old)
            + self.N2 * math.log(V2_new / V2_old)
            - self.beta * dU
        )

        if log_acc >= 0.0 or self.rng.random() < math.exp(min(log_acc, 0.0)):
            self.pos1 = pos1_new
            self.pos2 = pos2_new
            self.L1 = L1_new
            self.L2 = L2_new
            self._vol_acc += 1
            return float(U1_new), float(U2_new)

        return U1, U2

    # ── Particle transfer move ────────────────────────────────────────

    def _try_transfer(self) -> None:
        """Attempt a particle transfer between the two boxes.

        With equal probability, try to move a particle from box 1 → 2
        or from box 2 → 1.

        Acceptance criterion (Gibbs ensemble):
            acc = min(1, (N_donor / (N_acceptor + 1)) * (V_acceptor / V_donor)
                        * exp(-β [ΔU_insert + ΔU_delete]))

        This satisfies detailed balance for the semi-grand canonical
        partition function of the Gibbs ensemble.
        """
        self._xfer_att += 1

        # Choose direction: 0 = box1→box2, 1 = box2→box1
        direction = int(self.rng.integers(0, 2))

        if direction == 0:
            # Donor: box1, acceptor: box2
            if self.N1 == 0:
                return
            N_donor = self.N1
            N_acceptor = self.N2
            pos_donor = self.pos1
            pos_acceptor = self.pos2
            L_donor = self.L1
            L_acceptor = self.L2
        else:
            # Donor: box2, acceptor: box1
            if self.N2 == 0:
                return
            N_donor = self.N2
            N_acceptor = self.N1
            pos_donor = self.pos2
            pos_acceptor = self.pos1
            L_donor = self.L2
            L_acceptor = self.L1

        # Pick a random particle from the donor box to remove
        idx_remove = int(self.rng.integers(0, N_donor))
        pos_remove = pos_donor[idx_remove].copy()

        # Energy cost of removal (deletion energy = negative of pair sum)
        u_delete = _single_energy_np(
            self.potential, pos_remove, pos_donor, idx_remove, L_donor
        )

        # Random insertion position in acceptor box
        pos_insert = self.rng.random(3) * L_acceptor

        # Energy gain of insertion
        u_insert = _insertion_energy_np(
            self.potential, pos_insert, pos_acceptor, L_acceptor
        )

        dU = u_insert - u_delete

        # Acceptance criterion
        log_acc = (
            math.log(N_donor * L_acceptor ** 3)
            - math.log((N_acceptor + 1) * L_donor ** 3)
            - self.beta * dU
        )

        if log_acc >= 0.0 or self.rng.random() < math.exp(min(log_acc, 0.0)):
            # Execute transfer: remove from donor, insert into acceptor
            if direction == 0:
                self.pos1 = np.delete(self.pos1, idx_remove, axis=0)
                self.N1 -= 1
                self.pos2 = np.vstack([self.pos2, pos_insert]) if self.N2 > 0 else pos_insert.reshape(1, 3)
                self.N2 += 1
            else:
                self.pos2 = np.delete(self.pos2, idx_remove, axis=0)
                self.N2 -= 1
                self.pos1 = np.vstack([self.pos1, pos_insert]) if self.N1 > 0 else pos_insert.reshape(1, 3)
                self.N1 += 1
            self._xfer_acc += 1

    # ── Tuning ───────────────────────────────────────────────────────

    def _tune_disp(self, max_disp: float, box: int) -> float:
        """Tune displacement step for box *box*."""
        att = self._disp_att[box]
        if att == 0:
            return max_disp
        rate = self._disp_acc[box] / att
        if rate > 0.40:
            max_disp *= 1.05
        elif rate < 0.30:
            max_disp *= 0.95
        L = self.L1 if box == 0 else self.L2
        return float(np.clip(max_disp, 0.001, L / 2.0))

    def _tune_vol(self, max_vol_step: float) -> float:
        """Tune volume step size."""
        if self._vol_att == 0:
            return max_vol_step
        rate = self._vol_acc / self._vol_att
        if rate > 0.40:
            max_vol_step *= 1.05
        elif rate < 0.30:
            max_vol_step *= 0.95
        return float(np.clip(max_vol_step, 1e-5, 2.0))

    def _reset_counters(self) -> None:
        self._disp_acc = [0, 0]
        self._disp_att = [0, 0]
        self._vol_acc = 0
        self._vol_att = 0
        self._xfer_acc = 0
        self._xfer_att = 0

    # ── Public run method ─────────────────────────────────────────────

    def run(
        self,
        n_equil: int,
        n_prod: int,
        max_disp: float = 0.1,
        max_vol_step: float = 0.05,
        n_sample: int = 10,
        verbose: bool = False,
    ) -> Dict:
        """Run Gibbs Ensemble Monte Carlo.

        Each sweep consists of N_total attempted moves, distributed as:
            - 50% displacement moves (alternating between boxes)
            - 10% volume exchange moves
            - 40% particle transfer moves

        Displacement and volume step sizes are auto-tuned every 200 sweeps
        to target 30–40% acceptance rates.

        Parameters
        ----------
        n_equil : int
            Number of equilibration sweeps.
        n_prod : int
            Number of production sweeps.
        max_disp : float
            Initial maximum displacement step (tuned automatically).
        max_vol_step : float
            Initial maximum ln(V) step for volume moves (tuned automatically).
        n_sample : int
            Record observables every *n_sample* production sweeps.
        verbose : bool
            Print progress every 500 sweeps.

        Returns
        -------
        dict with keys:
            'rho1'        : list of sampled densities from box 1
            'rho2'        : list of sampled densities from box 2
            'N1'          : list of sampled particle counts in box 1
            'N2'          : list of sampled particle counts in box 2
            'mean_rho1'   : float — mean density of box 1
            'mean_rho2'   : float — mean density of box 2
            'mean_N1'     : float — mean N1
            'mean_N2'     : float — mean N2
            'rho_liquid'  : float — density of the denser (liquid) box
            'rho_vapor'   : float — density of the sparser (vapor) box
            'acc_disp1'   : displacement acceptance rate, box 1
            'acc_disp2'   : displacement acceptance rate, box 2
            'acc_vol'     : volume-move acceptance rate
            'acc_xfer'    : particle-transfer acceptance rate
        """
        disp1 = float(max_disp)
        disp2 = float(max_disp)
        vol_step = float(max_vol_step)
        tune_freq = 200

        # Move mix fractions
        FRAC_DISP = 0.50
        FRAC_VOL = 0.10
        # remainder (0.40) is particle transfer

        def _one_sweep(U1: float, U2: float) -> Tuple[float, float]:
            """Attempt N_total moves in the Gibbs move mix."""
            N_total = self.N1 + self.N2
            N_moves = max(N_total, 1)

            n_disp = int(FRAC_DISP * N_moves)
            n_vol = max(int(FRAC_VOL * N_moves), 1)
            n_xfer = N_moves - n_disp - n_vol

            # Displacement moves (alternate between boxes)
            for k in range(n_disp):
                box = k % 2
                self._try_displace(box, disp1 if box == 0 else disp2)

            # Re-sync energies after displacements (avoid drift)
            U1 = self._U1()
            U2 = self._U2()

            # Volume exchange moves
            for _ in range(n_vol):
                U1, U2 = self._try_volume(vol_step, U1, U2)

            # Particle transfer moves
            for _ in range(n_xfer):
                self._try_transfer()

            # Re-sync energies after transfers
            U1 = self._U1()
            U2 = self._U2()
            return U1, U2

        # ── Equilibration ─────────────────────────────────────────────
        U1 = self._U1()
        U2 = self._U2()

        for sweep in range(n_equil):
            U1, U2 = _one_sweep(U1, U2)

            if (sweep + 1) % tune_freq == 0:
                disp1 = self._tune_disp(disp1, 0)
                disp2 = self._tune_disp(disp2, 1)
                vol_step = self._tune_vol(vol_step)
                self._reset_counters()

            if verbose and (sweep + 1) % 500 == 0:
                print(
                    f"  [equil] sweep {sweep+1:6d}/{n_equil}  "
                    f"N1={self.N1:4d}  N2={self.N2:4d}  "
                    f"ρ1={self.rho1:.4f}  ρ2={self.rho2:.4f}  "
                    f"disp1={disp1:.4f}  disp2={disp2:.4f}  vol={vol_step:.4f}"
                )

        # Reset counters for production statistics
        self._reset_counters()

        # ── Production ────────────────────────────────────────────────
        rho1_samples: List[float] = []
        rho2_samples: List[float] = []
        N1_samples: List[int] = []
        N2_samples: List[int] = []

        for sweep in range(n_prod):
            U1, U2 = _one_sweep(U1, U2)

            if (sweep + 1) % tune_freq == 0:
                disp1 = self._tune_disp(disp1, 0)
                disp2 = self._tune_disp(disp2, 1)
                vol_step = self._tune_vol(vol_step)
                self._reset_counters()

            if (sweep + 1) % n_sample == 0:
                rho1_samples.append(float(self.rho1))
                rho2_samples.append(float(self.rho2))
                N1_samples.append(int(self.N1))
                N2_samples.append(int(self.N2))

            if verbose and (sweep + 1) % 500 == 0:
                print(
                    f"  [prod]  sweep {sweep+1:6d}/{n_prod}  "
                    f"N1={self.N1:4d}  N2={self.N2:4d}  "
                    f"ρ1={self.rho1:.4f}  ρ2={self.rho2:.4f}  "
                    f"acc_xfer={self._xfer_acc/max(self._xfer_att,1):.3f}"
                )

        rho1_arr = np.array(rho1_samples)
        rho2_arr = np.array(rho2_samples)
        N1_arr = np.array(N1_samples)
        N2_arr = np.array(N2_samples)

        mean_rho1 = float(np.mean(rho1_arr)) if len(rho1_arr) > 0 else 0.0
        mean_rho2 = float(np.mean(rho2_arr)) if len(rho2_arr) > 0 else 0.0

        # Identify liquid (denser) and vapor (sparser) boxes
        if mean_rho1 >= mean_rho2:
            rho_liquid = mean_rho1
            rho_vapor = mean_rho2
        else:
            rho_liquid = mean_rho2
            rho_vapor = mean_rho1

        return {
            "rho1": rho1_arr.tolist(),
            "rho2": rho2_arr.tolist(),
            "N1": N1_arr.tolist(),
            "N2": N2_arr.tolist(),
            "mean_rho1": mean_rho1,
            "mean_rho2": mean_rho2,
            "mean_N1": float(np.mean(N1_arr)) if len(N1_arr) > 0 else 0.0,
            "mean_N2": float(np.mean(N2_arr)) if len(N2_arr) > 0 else 0.0,
            "rho_liquid": rho_liquid,
            "rho_vapor": rho_vapor,
            "acc_disp1": self._disp_acc[0] / max(self._disp_att[0], 1),
            "acc_disp2": self._disp_acc[1] / max(self._disp_att[1], 1),
            "acc_vol": self._vol_acc / max(self._vol_att, 1),
            "acc_xfer": self._xfer_acc / max(self._xfer_att, 1),
        }

    # ── State accessors ───────────────────────────────────────────────

    @property
    def state1(self) -> MCState:
        """Current MCState for box 1."""
        import jax.numpy as jnp
        return MCState(
            positions=jnp.array(self.pos1),
            box_length=self.L1,
            n_particles=self.N1,
        )

    @property
    def state2(self) -> MCState:
        """Current MCState for box 2."""
        import jax.numpy as jnp
        return MCState(
            positions=jnp.array(self.pos2),
            box_length=self.L2,
            n_particles=self.N2,
        )
