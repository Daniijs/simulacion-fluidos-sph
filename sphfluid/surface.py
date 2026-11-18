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
from pathlib import Path

import numpy as np
import pysplashsurf

from .io import load_metadata, read_frame

log = logging.getLogger(__name__)

SPRAY_DENSITY = 0.35  # relative to the median density: isolated droplets
SPRAY_SPEED = 0.8  # m/s


def reconstruct(
    positions: np.ndarray, velocities: np.ndarray, spacing: float, resolution: float = 0.5
) -> dict[str, np.ndarray]:
    """Triangle mesh of the free surface for one frame."""
    radius = 0.5 * spacing
    mesh_with_data, reconstruction = pysplashsurf.reconstruction_pipeline(
        positions.astype(np.float32),
        attributes_to_interpolate={"velocity": velocities.astype(np.float32)},
        particle_radius=radius,
        smoothing_length=2.0,
        cube_size=resolution,
        iso_surface_threshold=0.6,
        mesh_smoothing_iters=15,
        mesh_smoothing_weights=True,
        compute_normals=True,
        sph_normals=False,
        normals_smoothing_iters=10,
        mesh_cleanup=True,
        decimate_barnacles=True,
    )
    mesh = mesh_with_data.mesh
    attrs = mesh_with_data.point_attributes
    density = np.asarray(reconstruction.particle_densities)
    speed = np.linalg.norm(velocities, axis=1)
    spray = (density < SPRAY_DENSITY * np.median(density)) & (speed > SPRAY_SPEED)
    return {
        "spray_position": positions[spray].astype(np.float32),
        "spray_velocity": velocities[spray].astype(np.float16),
        "vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "triangles": np.asarray(mesh.triangles, dtype=np.int32),
        "normals": np.asarray(attrs["normals"], dtype=np.float16),
        "velocity": np.asarray(attrs["velocity"], dtype=np.float16),
    }


def reconstruct_directory(
    directory: str | Path, resolution: float = 0.5, start: int = 0, end: int | None = None
) -> None:
    directory = Path(directory)
    meta = load_metadata(directory)
    spacing = meta["fluid"]["particle_spacing"]
    out = directory / "surface"
    out.mkdir(exist_ok=True)
    frames = sorted((directory / "particles").glob("frame_*.npz"))[start:end]
    for path in frames:
        target = out / path.name
        if target.exists():
            continue
        began = time.perf_counter()
        x, v, _ = read_frame(path, meta["domain_lower"], meta["domain_upper"])
        surface = reconstruct(x, v, spacing, resolution)
        np.savez(target, **surface)
        log.info("%s: %d triangles in %.1fs", path.name, len(surface["triangles"]), time.perf_counter() - began)
