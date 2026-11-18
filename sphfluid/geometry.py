"""Particle sampling of fluid volumes and solid boundaries.

Solids are represented by one or more layers of static boundary particles
(Akinci et al., 2012). Each layer is offset into the solid along the surface
normal, which keeps fast-moving fluid from tunnelling through thin walls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

BOUNDARY_SPACING_RATIO = 0.65  # boundary samples are denser than fluid particles


def lattice_in_box(lower, upper, spacing: float) -> np.ndarray:
    """Cell-centred cubic lattice filling the axis-aligned box [lower, upper]."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    counts = np.maximum(np.floor((upper - lower) / spacing).astype(int), 1)
    axes = [lower[d] + (np.arange(counts[d]) + 0.5) * spacing for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def _plane_samples(origin, u, v, extent_u, extent_v, spacing):
    nu = max(int(np.ceil(extent_u / spacing)), 1) + 1
    nv = max(int(np.ceil(extent_v / spacing)), 1) + 1
    su, sv = np.meshgrid(np.linspace(0, extent_u, nu), np.linspace(0, extent_v, nv), indexing="ij")
    return origin + su.reshape(-1, 1) * u + sv.reshape(-1, 1) * v


def box_walls(lower, upper, spacing: float, layers: int = 2, open_top: bool = True) -> np.ndarray:
    """Boundary particles on the inner faces of a container, layered outwards."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    e = np.eye(3)
    points = []
    for layer in range(layers):
        lo = lower - layer * spacing
        hi = upper + layer * spacing
        ext = hi - lo
        for axis in range(3):
            u_axis, v_axis = [a for a in range(3) if a != axis]
            for side, value in ((0, lo[axis]), (1, hi[axis])):
                if open_top and axis == 1 and side == 1:
                    continue
                origin = lo.copy()
                origin[axis] = value
                points.append(_plane_samples(origin, e[u_axis], e[v_axis], ext[u_axis], ext[v_axis], spacing))
    return _deduplicate(np.concatenate(points), spacing * 0.25)


def triangle_samples(vertices: np.ndarray, faces: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Points on every triangle on a barycentric grid no coarser than ``spacing``.

    Returns the points and the index of the face each one came from. Faces are
    grouped by subdivision level so the work stays vectorised.
    """
    tri = vertices[faces]
    edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(axis=1)
    levels = np.maximum(np.ceil(edges / spacing).astype(int), 1)
    points, owners = [], []
    for n in np.unique(levels):
        sel = np.nonzero(levels == n)[0]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        u, v = i[keep] / n, j[keep] / n
        w = 1.0 - u - v
        t = tri[sel]
        pts = w[None, :, None] * t[:, None, 0] + u[None, :, None] * t[:, None, 1] + v[None, :, None] * t[:, None, 2]
        points.append(pts.reshape(-1, 3))
        owners.append(np.repeat(sel, len(u)))
    return np.concatenate(points), np.concatenate(owners)


def mesh_surface(mesh: trimesh.Trimesh, spacing: float, layers: int = 2) -> np.ndarray:
    """Boundary particles covering a closed mesh, layered inwards along the normal."""
    samples, owner = triangle_samples(np.asarray(mesh.vertices), np.asarray(mesh.faces), spacing)
    normals = np.asarray(mesh.face_normals)[owner]
    layered = np.concatenate([samples - normals * spacing * k for k in range(layers)])
    return _deduplicate(layered, spacing * 0.5)


@dataclass(frozen=True)
class Heightfield:
    """Terrain described by heights on a regular grid in the XZ plane."""

    origin_x: float
    origin_z: float
    cell: float
    heights: np.ndarray  # shape (nx, nz)

    @property
    def extent(self) -> tuple[float, float]:
        nx, nz = self.heights.shape
        return (nx - 1) * self.cell, (nz - 1) * self.cell

    def height_at(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the terrain height."""
        nx, nz = self.heights.shape
        fx = np.clip((np.asarray(x) - self.origin_x) / self.cell, 0, nx - 1.0001)
        fz = np.clip((np.asarray(z) - self.origin_z) / self.cell, 0, nz - 1.0001)
        ix, iz = fx.astype(int), fz.astype(int)
        tx, tz = fx - ix, fz - iz
        h = self.heights
        return (
            h[ix, iz] * (1 - tx) * (1 - tz)
            + h[ix + 1, iz] * tx * (1 - tz)
            + h[ix, iz + 1] * (1 - tx) * tz
            + h[ix + 1, iz + 1] * tx * tz
        )

    def normal_at(self, x: np.ndarray, z: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        dhdx = (self.height_at(x + eps, z) - self.height_at(x - eps, z)) / (2 * eps)
        dhdz = (self.height_at(x, z + eps) - self.height_at(x, z - eps)) / (2 * eps)
        n = np.stack([-dhdx, np.ones_like(dhdx), -dhdz], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def surface_particles(self, spacing: float, layers: int = 2) -> np.ndarray:
        """Boundary particles following the terrain surface, layered downwards.

        Sampling the triangles (instead of a grid in plan) keeps the spacing
        constant on steep slopes, so there are no holes for fluid to leak through.
        """
        mesh = self.to_mesh()
        samples, owner = triangle_samples(np.asarray(mesh.vertices), np.asarray(mesh.faces), spacing)
        samples = _deduplicate(samples, spacing * 0.5)
        normals = self.normal_at(samples[:, 0], samples[:, 2])
        return np.concatenate([samples - normals * spacing * k for k in range(layers)])

    def skirt_mesh(self, bottom: float) -> trimesh.Trimesh:
        """Vertical sides closing the terrain down to ``bottom``: the cut edge of a diorama."""
        nx, nz = self.heights.shape
        xs = self.origin_x + np.arange(nx) * self.cell
        zs = self.origin_z + np.arange(nz) * self.cell
        ring = (
            [(xs[i], zs[0], self.heights[i, 0]) for i in range(nx)]
            + [(xs[-1], zs[j], self.heights[-1, j]) for j in range(1, nz)]
            + [(xs[i], zs[-1], self.heights[i, -1]) for i in range(nx - 2, -1, -1)]
            + [(xs[0], zs[j], self.heights[0, j]) for j in range(nz - 2, 0, -1)]
        )
        ring = np.array(ring)
        n = len(ring)
        top = np.c_[ring[:, 0], ring[:, 2], ring[:, 1]]
        low = np.c_[ring[:, 0], np.full(n, bottom), ring[:, 1]]
        vertices = np.concatenate([top, low, [[ring[:, 0].mean(), bottom, ring[:, 1].mean()]]])
        i = np.arange(n)
        j = (i + 1) % n
        sides = np.concatenate([np.stack([i, j, j + n], 1), np.stack([i, j + n, i + n], 1)])
        floor = np.stack([np.full(n, 2 * n), j + n, i + n], 1)
        mesh = trimesh.Trimesh(vertices, np.concatenate([sides, floor]), process=True)
        trimesh.repair.fix_normals(mesh)
        return mesh

    def to_mesh(self) -> trimesh.Trimesh:
        nx, nz = self.heights.shape
        xs = self.origin_x + np.arange(nx) * self.cell
        zs = self.origin_z + np.arange(nz) * self.cell
        gx, gz = np.meshgrid(xs, zs, indexing="ij")
        vertices = np.stack([gx, self.heights, gz], axis=-1).reshape(-1, 3)
        idx = np.arange(nx * nz).reshape(nx, nz)
        a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
        c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        faces = np.concatenate([np.stack([a, d, b], 1), np.stack([b, d, c], 1)])
        return trimesh.Trimesh(vertices, faces, process=False)


def remove_near(points: np.ndarray, obstacles: np.ndarray, distance: float) -> np.ndarray:
    """Drop ``points`` closer than ``distance`` to any obstacle particle."""
    if len(obstacles) == 0 or len(points) == 0:
        return points
    dist, _ = cKDTree(obstacles).query(points, k=1, distance_upper_bound=distance)
    return points[~np.isfinite(dist)]


def _deduplicate(points: np.ndarray, tolerance: float) -> np.ndarray:
    keys = np.round(points / tolerance).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique)]
