"""SPH smoothing kernels (Müller et al., 2003).

The NumPy versions are used for set-up (mass calibration) and as reference
implementations in the tests. The solver evaluates the same formulas inside
Taichi kernels using the coefficients computed here.
"""

from __future__ import annotations

import math

import numpy as np


def poly6_coefficient(h: float) -> float:
    """Normalisation constant of the Poly6 kernel with support radius ``h``."""
    return 315.0 / (64.0 * math.pi * h**9)


def spiky_gradient_coefficient(h: float) -> float:
    """Normalisation constant of the gradient of the Spiky kernel."""
    return -45.0 / (math.pi * h**6)


def poly6(r: np.ndarray | float, h: float) -> np.ndarray:
    """Poly6 kernel W(r, h). Zero outside the support radius."""
    r = np.asarray(r, dtype=np.float64)
    q = np.clip(h * h - r * r, 0.0, None)
    return poly6_coefficient(h) * q**3


def spiky_gradient(r_vec: np.ndarray, h: float) -> np.ndarray:
    """Gradient of the Spiky kernel for an array of offset vectors (N, 3)."""
    r_vec = np.atleast_2d(np.asarray(r_vec, dtype=np.float64))
    r = np.linalg.norm(r_vec, axis=1)
    inside = (r > 1e-12) & (r < h)
    grad = np.zeros_like(r_vec)
    scale = spiky_gradient_coefficient(h) * (h - r[inside]) ** 2 / r[inside]
    grad[inside] = scale[:, None] * r_vec[inside]
    return grad


def lattice_offsets(spacing: float, h: float) -> np.ndarray:
    """Offsets of all sites of a cubic lattice that lie inside the kernel support.

    Includes the origin. Used to calibrate particle mass and the constraint
    stiffness so that a particle inside a resting lattice has exactly the rest
    density.
    """
    n = int(math.ceil(h / spacing))
    axis = np.arange(-n, n + 1) * spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[np.linalg.norm(grid, axis=1) < h]


def boundary_volume_scale(spacing: float, h: float, boundary_spacing: float, layers: int) -> float:
    """Correction factor for Akinci boundary volumes.

    ``psi_b = rho0 / sum_k W_bk`` overestimates the contribution of thin,
    densely sampled walls. We sample a flat reference wall exactly like the
    real ones, place a fluid particle half a spacing above it and return the
    factor that makes the wall contribute what the missing half-space of fluid
    would have contributed. With it, a particle resting on a wall has the rest
    density.
    """
    n = int(math.ceil(h / boundary_spacing)) * 2 + 1
    axis = (np.arange(n) - n // 2) * boundary_spacing
    gx, gz = np.meshgrid(axis, axis, indexing="ij")
    wall = np.concatenate(
        [np.stack([gx.ravel(), np.full(gx.size, -k * boundary_spacing), gz.ravel()], 1) for k in range(layers)]
    )

    # Volume of each wall sample from its wall neighbours (as the solver does).
    diff = wall[:, None, :] - wall[None, :, :]
    sums = poly6(np.linalg.norm(diff, axis=-1), h).sum(axis=1)
    volume = 1.0 / sums

    probe = np.array([0.0, 0.5 * spacing, 0.0])
    actual = float((volume * poly6(np.linalg.norm(wall - probe, axis=1), h)).sum())

    sites = lattice_offsets(spacing, h * 2) + probe
    weights = poly6(np.linalg.norm(sites - probe, axis=1), h)
    below = weights[sites[:, 1] < 0].sum() / weights.sum()  # fraction of density from y < 0
    return below / actual
