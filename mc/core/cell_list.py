"""
Cell List for O(N) neighbor finding.

Divides the simulation box into cells of size >= r_cut.
Each cell stores the indices of particles within it.
For a displacement move, only check particles in the 27 neighboring cells.

This replaces the O(N²) all-pairs check in single_particle_overlap.

Implementation notes
--------------------
- Uses NumPy (not JAX): the dict-based cell→particle mapping doesn't JIT.
- Speedup comes from reducing neighbor candidates from N to ~27*ρ_cell.
- At η=0.3, σ=1, there are roughly 5–10 particles per cell, so each
  overlap check touches ~130–270 candidates instead of N.
- Incremental update on accepted moves keeps rebuild cost O(1) per step.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Tuple


# ── CellList ──────────────────────────────────────────────────────────


class CellList:
    """Spatial cell list for O(N) neighbor finding in a periodic cubic box.

    Divides [0, L)^3 into n_cells^3 cubic cells each of side >= r_cut.
    Neighbor queries return all particles in the 27 cells surrounding a
    given position (3×3×3 stencil with PBC wrapping).

    Parameters
    ----------
    box_length : float
        Side length L of the cubic simulation box.
    r_cut : float
        Cutoff distance.  Cell size is chosen so that cell_size >= r_cut,
        ensuring all potential neighbors lie within the 27-cell stencil.
        Defaults to 1.0 (one hard-sphere diameter).
    """

    def __init__(self, box_length: float, r_cut: float = 1.0):
        self.box_length = float(box_length)
        self.r_cut = float(r_cut)

        # Number of cells per dimension: floor(L / r_cut), minimum 1.
        # Cell side >= r_cut guarantees all neighbors within 27-cell stencil.
        self.n_cells: int = max(1, math.floor(self.box_length / self.r_cut))
        self.cell_size: float = self.box_length / self.n_cells

        # cell_index (ix, iy, iz) → list of particle indices
        self._cells: Dict[Tuple[int, int, int], List[int]] = {}

    # ── Internal helpers ──────────────────────────────────────────────

    def _cell_index(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Return (ix, iy, iz) cell index for position *pos*."""
        # Floor division; clamp to [0, n_cells-1] to handle floating-point
        # positions that land exactly on L due to PBC wrap.
        nc = self.n_cells
        ix = int(pos[0] / self.cell_size) % nc
        iy = int(pos[1] / self.cell_size) % nc
        iz = int(pos[2] / self.cell_size) % nc
        return (ix, iy, iz)

    def _neighbor_cells(self, cell_idx: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Return the 27 cells in the 3×3×3 stencil around *cell_idx* (PBC)."""
        nc = self.n_cells
        ix, iy, iz = cell_idx
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbors.append((
                        (ix + dx) % nc,
                        (iy + dy) % nc,
                        (iz + dz) % nc,
                    ))
        return neighbors

    # ── Public API ────────────────────────────────────────────────────

    def build(self, positions: np.ndarray) -> None:
        """Build the cell list from scratch.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
            Current particle positions in [0, L)^3.
        """
        positions = np.asarray(positions)
        self._cells = {}
        for i, pos in enumerate(positions):
            cidx = self._cell_index(pos)
            if cidx not in self._cells:
                self._cells[cidx] = []
            self._cells[cidx].append(i)

    def get_neighbors(self, pos_i: np.ndarray) -> List[int]:
        """Return indices of all particles in the 27-cell stencil around *pos_i*.

        Parameters
        ----------
        pos_i : array-like, shape (3,)
            Query position.

        Returns
        -------
        list of int
            Particle indices in the 27 neighboring cells (may include
            the queried particle itself; callers must exclude particle i
            during overlap checks).
        """
        pos_i = np.asarray(pos_i)
        cidx = self._cell_index(pos_i)
        neighbors: List[int] = []
        for nc in self._neighbor_cells(cidx):
            if nc in self._cells:
                neighbors.extend(self._cells[nc])
        return neighbors

    def update_particle(
        self,
        i: int,
        old_pos: np.ndarray,
        new_pos: np.ndarray,
    ) -> None:
        """Incrementally move particle *i* from its old cell to its new cell.

        O(particles-in-old-cell) — much faster than a full rebuild.

        Parameters
        ----------
        i : int
            Particle index.
        old_pos : array-like, shape (3,)
            Previous position (used to locate old cell).
        new_pos : array-like, shape (3,)
            New position (used to locate new cell).
        """
        old_pos = np.asarray(old_pos)
        new_pos = np.asarray(new_pos)

        old_cidx = self._cell_index(old_pos)
        new_cidx = self._cell_index(new_pos)

        if old_cidx == new_cidx:
            return  # Particle stays in the same cell — nothing to do.

        # Remove from old cell
        if old_cidx in self._cells:
            try:
                self._cells[old_cidx].remove(i)
            except ValueError:
                pass  # Defensive: particle not found, ignore.
            if not self._cells[old_cidx]:
                del self._cells[old_cidx]

        # Insert into new cell
        if new_cidx not in self._cells:
            self._cells[new_cidx] = []
        self._cells[new_cidx].append(i)

    # ── Diagnostics ───────────────────────────────────────────────────

    def occupancy_stats(self) -> Dict[str, float]:
        """Return mean and max particles per occupied cell (for diagnostics)."""
        if not self._cells:
            return {"mean": 0.0, "max": 0, "n_occupied": 0}
        counts = [len(v) for v in self._cells.values()]
        return {
            "mean": float(np.mean(counts)),
            "max": int(np.max(counts)),
            "n_occupied": len(counts),
        }

    def __repr__(self) -> str:
        return (
            f"CellList(L={self.box_length}, r_cut={self.r_cut}, "
            f"n_cells={self.n_cells}, cell_size={self.cell_size:.4f})"
        )
