"""Uniform grid for fixed-radius neighbour search.

Particles are bucketed into cells of size ``h`` with a counting sort. After
:meth:`UniformGrid.build`, ``destination[i]`` holds the sorted slot of particle
``i``; the solver uses it to physically reorder its arrays so that particles of
the same cell are contiguous in memory. Neighbour loops then read contiguous
ranges, which is what makes the GPU version fast.

Particles flagged as inactive are sent to an extra "parking" cell placed after
the last real cell, so they end up at the back of the arrays and are skipped.
"""

import math

import numpy as np
import taichi as ti

SCAN_BLOCK = 1024


@ti.data_oriented
class UniformGrid:
    def __init__(self, lower: np.ndarray, upper: np.ndarray, cell_size: float, capacity: int):
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        res = np.maximum(np.ceil((upper - lower) / cell_size).astype(int), 1)

        self.lower = ti.Vector([float(c) for c in lower])
        self.inv_cell = 1.0 / cell_size
        self.res = ti.Vector([int(r) for r in res])
        self.num_cells = int(np.prod(res))
        self.parking = self.num_cells  # index of the cell for inactive particles
        total = self.num_cells + 1
        self.num_blocks = math.ceil(total / SCAN_BLOCK)

        self.cell_count = ti.field(ti.i32, total)
        self.cell_start = ti.field(ti.i32, total)
        self.block_sum = ti.field(ti.i32, self.num_blocks)
        self.particle_cell = ti.field(ti.i32, capacity)
        self.rank = ti.field(ti.i32, capacity)
        self.destination = ti.field(ti.i32, capacity)
        self.num_active = ti.field(ti.i32, ())

    # --- queries used inside solver kernels -------------------------------------------------

    @ti.func
    def cell_of(self, p):
        c = ti.cast(ti.floor((p - self.lower) * self.inv_cell), ti.i32)
        return ti.max(ti.min(c, self.res - 1), 0)

    @ti.func
    def contains(self, c) -> bool:
        return (c >= 0).all() and (c < self.res).all()

    @ti.func
    def linear(self, c) -> ti.i32:
        return (c[0] * self.res[1] + c[1]) * self.res[2] + c[2]

    @ti.func
    def cell_range(self, c):
        """Half-open range [start, end) of sorted slots in cell ``c``."""
        idx = self.linear(c)
        start = self.cell_start[idx]
        return start, start + self.cell_count[idx]

    # --- construction ---------------------------------------------------------------------------

    def build(self, x: ti.template(), inactive_below: float, n: int) -> int:
        """Sort ``n`` particles into cells and return how many are active."""
        self._count(x, inactive_below, n)
        self._scan_blocks()
        self._scan_block_sums()
        self._add_block_offsets()
        self._assign_destinations(n)
        return int(self.num_active[None])

    @ti.kernel
    def _count(self, x: ti.template(), inactive_below: ti.f32, n: ti.i32):
        for c in self.cell_count:
            self.cell_count[c] = 0
        for i in range(n):
            cell = self.parking
            if x[i][1] >= inactive_below:
                cell = self.linear(self.cell_of(x[i]))
            self.particle_cell[i] = cell
            self.rank[i] = ti.atomic_add(self.cell_count[cell], 1)

    # Exclusive prefix sum of cell_count in three passes. Taichi's built-in scan
    # only runs on GPU backends; this one runs everywhere.
    @ti.kernel
    def _scan_blocks(self):
        total = self.num_cells + 1
        for b in range(self.num_blocks):
            acc = 0
            for k in range(b * SCAN_BLOCK, ti.min((b + 1) * SCAN_BLOCK, total)):
                self.cell_start[k] = acc
                acc += self.cell_count[k]
            self.block_sum[b] = acc

    @ti.kernel
    def _scan_block_sums(self):
        ti.loop_config(serialize=True)
        acc = 0
        for b in range(self.num_blocks):
            s = self.block_sum[b]
            self.block_sum[b] = acc
            acc += s

    @ti.kernel
    def _add_block_offsets(self):
        for k in self.cell_start:
            self.cell_start[k] += self.block_sum[k // SCAN_BLOCK]
        self.num_active[None] = self.cell_start[self.parking]

    @ti.kernel
    def _assign_destinations(self, n: ti.i32):
        for i in range(n):
            self.destination[i] = self.cell_start[self.particle_cell[i]] + self.rank[i]

    @ti.kernel
    def reorder(self, src: ti.template(), scratch: ti.template(), n: ti.i32):
        """Move ``src[i]`` to ``src[destination[i]]`` using ``scratch`` as buffer."""
        for i in range(n):
            scratch[self.destination[i]] = src[i]
        for i in range(n):
            src[i] = scratch[i]
