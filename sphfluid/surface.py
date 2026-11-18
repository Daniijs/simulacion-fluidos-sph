"""Water surface reconstruction with splashsurf (marching cubes on an SPH colour field).

For each stored particle frame this writes ``surface/frame_XXXXX.npz`` with
vertices, triangles, smoothed normals and the fluid velocity interpolated onto
the vertices (used for motion blur and to whiten fast, aerated water).

Isolated fast particles cannot form a surface; they are kept separately as
spray droplets, which the renderer draws as small white spheres.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pysplashsurf
from scipy.spatial import cKDTree

from .io import load_metadata, read_frame

log = logging.getLogger(__name__)

SPRAY_DENSITY = 0.25  # fraction of a full neighbourhood below which a particle is a droplet
SPRAY_SPEED = 0.8  # m/s


def reconstruct(
    positions: np.ndarray,
    velocities: np.ndarray,
    spacing: float,
    resolution: float = 0.5,
    smoothing: float = 3.0,
    threshold: float = 0.55,
) -> dict[str, np.ndarray]:
    """Triangle mesh of the free surface for one frame, plus the spray droplets.

    Isolated particles are left out of the mesh (each one would become a small
    bead) and returned as spray instead. ``smoothing`` and ``resolution`` are in
    particle radii: a support a little wider than the solver's keeps thin, fast
    sheets of water connected.
    """
    radius = 0.5 * spacing
    density = _relative_density(positions, spacing)
    speed = np.linalg.norm(velocities, axis=1)
    isolated = density < SPRAY_DENSITY
    bulk = ~isolated
    mesh_with_data, _ = pysplashsurf.reconstruction_pipeline(
        positions[bulk].astype(np.float32),
        attributes_to_interpolate={"velocity": velocities[bulk].astype(np.float32)},
        particle_radius=radius,
        smoothing_length=smoothing,
        cube_size=resolution,
        iso_surface_threshold=threshold,
        mesh_smoothing_iters=25,
        mesh_smoothing_weights=True,
        compute_normals=True,
        sph_normals=False,
        normals_smoothing_iters=10,
        mesh_cleanup=True,
        decimate_barnacles=True,
    )
    mesh = mesh_with_data.mesh
    attrs = mesh_with_data.point_attributes
    spray = isolated & (speed > SPRAY_SPEED)
    return {
        "spray_position": positions[spray].astype(np.float32),
        "spray_velocity": velocities[spray].astype(np.float16),
        "vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "triangles": np.asarray(mesh.triangles, dtype=np.int32),
        "normals": np.asarray(attrs["normals"], dtype=np.float16),
        "velocity": np.asarray(attrs["velocity"], dtype=np.float16),
    }


def _relative_density(positions: np.ndarray, spacing: float) -> np.ndarray:
    """Neighbour count within the solver's support radius, relative to a full neighbourhood."""
    tree = cKDTree(positions)
    counts = tree.query_ball_point(positions, r=2.0 * spacing, return_length=True, workers=-1)
    full = 4.0 / 3.0 * np.pi * 2.0**3  # lattice sites inside the support sphere
    return counts / full


def reconstruct_directory(
    directory: str | Path,
    resolution: float = 0.6,
    start: int = 0,
    end: int | None = None,
    on_frame: Callable[[int], None] | None = None,
) -> None:
    directory = Path(directory)
    meta = load_metadata(directory)
    spacing = meta["fluid"]["particle_spacing"]
    out = directory / "surface"
    out.mkdir(exist_ok=True)
    frames = sorted((directory / "particles").glob("frame_*.npz"))[start:end]
    for i, path in enumerate(frames):
        target = out / path.name
        if on_frame:
            on_frame(start + i)
        if target.exists():
            continue
        began = time.perf_counter()
        x, v, _ = read_frame(path, meta["domain_lower"], meta["domain_upper"])
        surface = reconstruct(x, v, spacing, resolution)
        np.savez(target, **surface)
        log.info("%s: %d triangles in %.1fs", path.name, len(surface["triangles"]), time.perf_counter() - began)
