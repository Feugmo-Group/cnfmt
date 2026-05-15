"""
NVT (Canonical) Monte Carlo for Hard Spheres
=============================================

Algorithm (Metropolis for hard spheres)
----------------------------------------
1.  Select particle i uniformly at random.
2.  Propose displacement:  r_i' = r_i + max_disp * (U - 0.5)  where U ~ Uniform(0,1)^3.
3.  Apply PBC:  r_i' = r_i' % L.
4.  If no overlap with any j ≠ i  →  accept (ΔU = 0, accept probability = 1).
5.  Else  →  reject (ΔU = ∞).

One **sweep** = N attempted moves.

Acceptance-rate tuning
-----------------------
Every 100 sweeps during a run, ``max_disp`` is adjusted to keep the
acceptance rate in [0.30, 0.40]:
    rate > 0.40  →  max_disp *= 1.05
    rate < 0.30  →  max_disp *= 0.95
clamped to [0.01, box_length / 2].

JAX patterns used
-----------------
- Single step is JIT-compiled.
- Sweep uses ``jax.lax.scan`` to avoid Python loop overhead.
- PRNG key threaded through every call (functional style).

Data classes
------------
NVTState wraps MCState with PRNG key and running counters.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from mc.core.state import MCState
from mc.core.potentials import single_particle_overlap

# Enable double precision
jax.config.update("jax_enable_x64", True)


# ── NVTState ──────────────────────────────────────────────────────────


@dataclass
class NVTState:
    """Combined NVT simulation state.

    Parameters
    ----------
    mc_state : MCState
    key : jnp.ndarray
        JAX PRNG key.
    n_accepted : int
        Cumulative accepted moves.
    n_total : int
        Cumulative attempted moves.
    """
    mc_state: MCState
    key: jnp.ndarray
    n_accepted: int
    n_total: int


# ── JIT-compiled single MC step ───────────────────────────────────────


@jax.jit
def _mc_step_jit(
    positions: jnp.ndarray,
    box_length: float,
    key: jnp.ndarray,
    max_disp: float,
    sigma: float,
) -> tuple[jnp.ndarray, bool, jnp.ndarray]:
    """Single displacement move (JIT-compiled).

    Parameters
    ----------
    positions : (N, 3)
    box_length : float
    key : PRNG key
    max_disp : float
        Maximum displacement in each direction.
    sigma : float
        Hard-sphere diameter.

    Returns
    -------
    new_positions : (N, 3)
    accepted : bool
    new_key : PRNG key
    """
    N = positions.shape[0]
    key, k1, k2 = jax.random.split(key, 3)

    # Select particle
    i = jax.random.randint(k1, shape=(), minval=0, maxval=N)

    # Propose displacement
    delta = max_disp * (jax.random.uniform(k2, shape=(3,)) - 0.5)
    new_pos_i = (positions[i] + delta) % box_length

    # Overlap check (minimum image convention inside)
    overlap = single_particle_overlap(new_pos_i, positions, i, box_length, sigma)

    # Accept or reject
    new_positions = jnp.where(
        overlap,
        positions,
        positions.at[i].set(new_pos_i),
    )
    accepted = ~overlap
    return new_positions, accepted, key


# ── Public API: step, sweep, run ──────────────────────────────────────


def mc_step(
    state: NVTState,
    max_disp: float = 0.2,
    sigma: float = 1.0,
) -> NVTState:
    """Attempt a single displacement move.

    Parameters
    ----------
    state : NVTState
    max_disp : float
    sigma : float

    Returns
    -------
    NVTState
        Updated state (positions updated if accepted).
    """
    new_positions, accepted, new_key = _mc_step_jit(
        state.mc_state.positions,
        state.mc_state.box_length,
        state.key,
        max_disp,
        sigma,
    )
    new_mc = MCState(
        positions=new_positions,
        box_length=state.mc_state.box_length,
        n_particles=state.mc_state.n_particles,
    )
    return NVTState(
        mc_state=new_mc,
        key=new_key,
        n_accepted=state.n_accepted + int(accepted),
        n_total=state.n_total + 1,
    )


def mc_sweep(
    state: NVTState,
    n_particles: int,
    max_disp: float = 0.2,
    sigma: float = 1.0,
) -> tuple[NVTState, int]:
    """Perform N attempted moves (one sweep) using ``jax.lax.scan``.

    The scan body is JIT-compiled automatically because ``_mc_step_jit``
    is already JIT-compiled and carries JAX arrays.

    Parameters
    ----------
    state : NVTState
    n_particles : int
        Number of moves = N (one sweep).
    max_disp : float
    sigma : float

    Returns
    -------
    (new_state, n_accepted_this_sweep)
    """
    box_length = state.mc_state.box_length

    def body(carry, _):
        positions, key, n_acc = carry
        new_positions, accepted, new_key = _mc_step_jit(
            positions, box_length, key, max_disp, sigma
        )
        return (new_positions, new_key, n_acc + jnp.int32(accepted)), None

    init = (state.mc_state.positions, state.key, jnp.int32(0))
    (new_positions, new_key, n_accepted_sweep), _ = jax.lax.scan(
        body, init, None, length=n_particles
    )

    new_mc = MCState(
        positions=new_positions,
        box_length=box_length,
        n_particles=n_particles,
    )
    new_state = NVTState(
        mc_state=new_mc,
        key=new_key,
        n_accepted=state.n_accepted + int(n_accepted_sweep),
        n_total=state.n_total + n_particles,
    )
    return new_state, int(n_accepted_sweep)


def run_nvt(
    state: NVTState,
    n_steps: int,
    max_disp: float = 0.2,
    sigma: float = 1.0,
    adjust_disp: bool = True,
    verbose: bool = False,
) -> tuple[NVTState, float, Dict]:
    """Run NVT Monte Carlo for *n_steps* sweeps.

    One sweep = N_particles attempted moves.

    Parameters
    ----------
    state : NVTState
    n_steps : int
        Number of sweeps.
    max_disp : float
        Initial max displacement (adjusted every 100 sweeps if adjust_disp).
    sigma : float
    adjust_disp : bool
        If True, tune max_disp every 100 sweeps to target 30–40% acceptance.
    verbose : bool
        Print progress every 1000 sweeps.

    Returns
    -------
    final_state : NVTState
    acceptance_rate : float
        Overall acceptance rate across all sweeps.
    history : dict
        Keys: 'acceptance_per_block' (every 100 sweeps), 'max_disp_per_block'.
    """
    N = state.mc_state.n_particles
    box_length = state.mc_state.box_length
    disp = float(max_disp)

    acc_per_block: list[float] = []
    disp_per_block: list[float] = []

    ADJUST_EVERY = 100
    block_accepted = 0
    block_attempted = 0

    for sweep_idx in range(n_steps):
        state, n_acc_sweep = mc_sweep(state, N, disp, sigma)
        block_accepted += n_acc_sweep
        block_attempted += N

        # Adjust displacement every ADJUST_EVERY sweeps
        if adjust_disp and (sweep_idx + 1) % ADJUST_EVERY == 0:
            rate = block_accepted / max(block_attempted, 1)
            acc_per_block.append(float(rate))
            disp_per_block.append(float(disp))

            if rate > 0.40:
                disp *= 1.05
            elif rate < 0.30:
                disp *= 0.95
            # Clamp
            disp = float(np.clip(disp, 0.01, box_length / 2.0))

            block_accepted = 0
            block_attempted = 0

            if verbose:
                print(
                    f"  sweep {sweep_idx+1:6d}/{n_steps}  "
                    f"acc={rate:.3f}  max_disp={disp:.4f}"
                )

    total_attempted = state.n_total
    total_accepted = state.n_accepted
    overall_rate = total_accepted / max(total_attempted, 1)

    history = {
        "acceptance_per_block": np.array(acc_per_block),
        "max_disp_per_block": np.array(disp_per_block),
    }
    return state, float(overall_rate), history


# ── NVTSampler convenience class ──────────────────────────────────────


class NVTSampler:
    """High-level NVT Monte Carlo sampler.

    Parameters
    ----------
    mc_state : MCState
        Initial configuration.
    seed : int
        Random seed for PRNG.
    sigma : float
    """

    def __init__(self, mc_state: MCState, seed: int = 0, sigma: float = 1.0):
        key = jax.random.PRNGKey(seed)
        self.nvt_state = NVTState(
            mc_state=mc_state,
            key=key,
            n_accepted=0,
            n_total=0,
        )
        self.sigma = float(sigma)

    def run(
        self,
        n_equil: int,
        n_prod: int,
        max_disp: float = 0.2,
        adjust_disp: bool = True,
        verbose: bool = False,
    ) -> tuple[NVTState, float, Dict]:
        """Run equilibration then production.

        Parameters
        ----------
        n_equil : int
            Number of equilibration sweeps (results discarded).
        n_prod : int
            Number of production sweeps.
        max_disp : float
        adjust_disp : bool
        verbose : bool

        Returns
        -------
        final_state, acceptance_rate, history
        """
        if verbose:
            print(f"Equilibrating ({n_equil} sweeps)...")
        self.nvt_state, _, _ = run_nvt(
            self.nvt_state,
            n_equil,
            max_disp=max_disp,
            sigma=self.sigma,
            adjust_disp=adjust_disp,
            verbose=verbose,
        )
        # Reset counters for production
        self.nvt_state = NVTState(
            mc_state=self.nvt_state.mc_state,
            key=self.nvt_state.key,
            n_accepted=0,
            n_total=0,
        )

        if verbose:
            print(f"Production ({n_prod} sweeps)...")
        self.nvt_state, acc_rate, history = run_nvt(
            self.nvt_state,
            n_prod,
            max_disp=max_disp,
            sigma=self.sigma,
            adjust_disp=adjust_disp,
            verbose=verbose,
        )
        return self.nvt_state, acc_rate, history

    @property
    def state(self) -> MCState:
        return self.nvt_state.mc_state
