"""
NVT Sampler with Cell-List Neighbor Finding — O(N) per sweep
=============================================================

Replaces the O(N²) ``single_particle_overlap`` hot path with an O(N)
cell-list based neighbor check.  At each attempted move, only the ~27
cells surrounding the trial position are searched, giving O(1) overlap
checks per move and O(N) per sweep (N moves).

The cell list is maintained incrementally: on each accepted move, only
the moved particle's cell entry is updated (O(1)), avoiding the O(N)
full rebuild cost.

API is intentionally identical to :class:`mc.ensembles.nvt.NVTSampler`
so the two can be swapped transparently for benchmarking.

Physics / correctness
---------------------
- Minimum-image convention applied to all pair distances.
- All particles in the 27-cell stencil are checked; this is exact (no
  missed neighbors) when cell_size >= sigma = r_cut.
- Acceptance criterion: hard-sphere Metropolis (reject on any overlap).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple

from mc.core.state import MCState
from mc.core.cell_list import CellList


# ── Overlap helper (pure NumPy, no JAX) ──────────────────────────────


def _has_overlap_numpy(
    trial_pos: np.ndarray,
    positions: np.ndarray,
    i: int,
    box_length: float,
    sigma2: float,
    neighbor_indices: list,
) -> bool:
    """Check whether *trial_pos* overlaps any neighbor j ≠ i.

    Parameters
    ----------
    trial_pos : np.ndarray, shape (3,)
    positions : np.ndarray, shape (N, 3)
    i : int
        Index of the moving particle (excluded from check).
    box_length : float
    sigma2 : float
        σ², the squared hard-sphere diameter.
    neighbor_indices : list of int
        Candidates from cell list (may include i).

    Returns
    -------
    bool
        True if any overlap found.
    """
    L = box_length
    for j in neighbor_indices:
        if j == i:
            continue
        dr = trial_pos - positions[j]
        # Minimum image convention
        dr = dr - L * np.round(dr / L)
        if np.dot(dr, dr) < sigma2:
            return True
    return False


# ── CellListNVTSampler ────────────────────────────────────────────────


class CellListNVTSampler:
    """NVT Monte Carlo with O(N) cell-list neighbor finding.

    Identical interface to :class:`mc.ensembles.nvt.NVTSampler`.

    Parameters
    ----------
    mc_state : MCState
        Initial configuration.
    seed : int
        NumPy random seed.
    sigma : float
        Hard-sphere diameter (default 1.0).
    """

    def __init__(self, mc_state: MCState, seed: int = 0, sigma: float = 1.0):
        self.mc_state = mc_state
        self.sigma = float(sigma)
        self.sigma2 = self.sigma ** 2
        self._rng = np.random.default_rng(seed)

        # Build initial cell list; r_cut = sigma for hard spheres
        self._cell_list = CellList(mc_state.box_length, r_cut=self.sigma)
        self._positions = np.array(mc_state.positions, dtype=np.float64)  # copy, writable
        self._cell_list.build(self._positions)

        # Running counters
        self._n_accepted: int = 0
        self._n_total: int = 0

    # ── Single sweep (N attempted moves) ─────────────────────────────

    def _sweep(self, max_disp: float) -> int:
        """Perform one sweep (N attempted moves).

        Returns
        -------
        int
            Number of accepted moves this sweep.
        """
        N = self.mc_state.n_particles
        L = self.mc_state.box_length
        positions = self._positions
        rng = self._rng
        sigma2 = self.sigma2

        n_acc = 0
        for _ in range(N):
            # 1. Pick particle uniformly
            i = int(rng.integers(0, N))

            # 2. Propose displacement
            delta = max_disp * (rng.random(3) - 0.5)
            old_pos = positions[i]
            trial_pos = (old_pos + delta) % L

            # 3. Get neighbor candidates from cell list
            neighbors = self._cell_list.get_neighbors(trial_pos)

            # 4. Overlap check (excludes particle i)
            overlap = _has_overlap_numpy(
                trial_pos, positions, i, L, sigma2, neighbors
            )

            # 5. Accept / reject
            if not overlap:
                # Update cell list incrementally before updating positions
                self._cell_list.update_particle(i, old_pos, trial_pos)
                positions[i] = trial_pos
                n_acc += 1

        return n_acc

    # ── Displacement tuning ───────────────────────────────────────────

    def _adjust_disp(self, max_disp: float, rate: float) -> float:
        """Tune max_disp to keep acceptance rate in [0.30, 0.40]."""
        L = self.mc_state.box_length
        if rate > 0.40:
            max_disp *= 1.05
        elif rate < 0.30:
            max_disp *= 0.95
        return float(np.clip(max_disp, 0.01, L / 2.0))

    # ── Internal run loop ─────────────────────────────────────────────

    def _run_sweeps(
        self,
        n_sweeps: int,
        max_disp: float,
        adjust_disp: bool,
        verbose: bool,
        label: str,
    ) -> Tuple[float, float, Dict]:
        """Run *n_sweeps* sweeps, return (final_max_disp, acc_rate, history)."""
        ADJUST_EVERY = 100
        block_accepted = 0
        block_attempted = 0
        acc_per_block: list[float] = []
        disp_per_block: list[float] = []
        N = self.mc_state.n_particles

        for sweep_idx in range(n_sweeps):
            n_acc = self._sweep(max_disp)
            self._n_accepted += n_acc
            self._n_total += N
            block_accepted += n_acc
            block_attempted += N

            if adjust_disp and (sweep_idx + 1) % ADJUST_EVERY == 0:
                rate = block_accepted / max(block_attempted, 1)
                acc_per_block.append(float(rate))
                disp_per_block.append(float(max_disp))
                max_disp = self._adjust_disp(max_disp, rate)
                block_accepted = 0
                block_attempted = 0

                if verbose:
                    print(
                        f"  [{label}] sweep {sweep_idx+1:6d}/{n_sweeps}  "
                        f"acc={rate:.3f}  max_disp={max_disp:.4f}"
                    )

        overall_rate = self._n_accepted / max(self._n_total, 1)
        history = {
            "acceptance_per_block": np.array(acc_per_block),
            "max_disp_per_block": np.array(disp_per_block),
        }
        return max_disp, overall_rate, history

    # ── Public API ────────────────────────────────────────────────────

    def run(
        self,
        n_equil: int,
        n_prod: int,
        max_disp: float = 0.2,
        adjust_disp: bool = True,
        verbose: bool = False,
    ) -> Tuple[MCState, float, Dict]:
        """Run equilibration then production, matching NVTSampler.run API.

        Parameters
        ----------
        n_equil : int
            Equilibration sweeps (statistics not collected).
        n_prod : int
            Production sweeps.
        max_disp : float
            Initial maximum displacement per move, per dimension.
        adjust_disp : bool
            If True, tune max_disp every 100 sweeps to target 30–40% acceptance.
        verbose : bool
            Print progress every 100 sweeps.

        Returns
        -------
        final_mc_state : MCState
            Updated configuration after production.
        acceptance_rate : float
            Overall acceptance rate during *production* phase.
        history : dict
            Keys: ``'acceptance_per_block'``, ``'max_disp_per_block'``.
        """
        # ── Equilibration ──
        if verbose:
            print(f"Equilibrating ({n_equil} sweeps) [cell list]...")
        max_disp, _, _ = self._run_sweeps(
            n_equil, max_disp, adjust_disp, verbose, label="equil"
        )

        # Reset production counters
        self._n_accepted = 0
        self._n_total = 0

        # ── Production ──
        if verbose:
            print(f"Production ({n_prod} sweeps) [cell list]...")
        _, acc_rate, history = self._run_sweeps(
            n_prod, max_disp, adjust_disp, verbose, label="prod"
        )

        # Sync MCState with updated positions
        import jax.numpy as jnp
        self.mc_state = MCState(
            positions=jnp.array(self._positions),
            box_length=self.mc_state.box_length,
            n_particles=self.mc_state.n_particles,
        )
        return self.mc_state, float(acc_rate), history

    @property
    def state(self) -> MCState:
        """Current MCState (positions synced from internal NumPy array)."""
        import jax.numpy as jnp
        return MCState(
            positions=jnp.array(self._positions),
            box_length=self.mc_state.box_length,
            n_particles=self.mc_state.n_particles,
        )

    @property
    def cell_list(self) -> CellList:
        """Expose the CellList for inspection."""
        return self._cell_list
