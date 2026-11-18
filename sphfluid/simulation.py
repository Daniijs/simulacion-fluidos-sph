"""High-level driver: build a scene, run the solver and write frames."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import taichi as ti

from .config import SceneConfig
from .io import FrameInfo, FrameWriter
from .scene import SceneGeometry, build_scene

log = logging.getLogger(__name__)

FrameCallback = Callable[[FrameInfo, "PBFSolver"], None]  # noqa: F821


def init_backend(arch: str = "auto", debug: bool = False) -> None:
    """Initialise Taichi. ``auto`` picks a GPU when one is available."""
    archs = {"auto": ti.gpu, "gpu": ti.gpu, "cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
    ti.init(arch=archs[arch], debug=debug, log_level=ti.WARN, default_fp=ti.f32, offline_cache=True)


def export_meshes(scene: SceneGeometry, directory: Path) -> None:
    mesh_dir = directory / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for name, mesh in scene.meshes.items():
        mesh.export(mesh_dir / f"{name}.ply")


def run(config: SceneConfig, output: str | Path | None = None, on_frame: FrameCallback | None = None) -> Path:
    """Simulate ``config`` and return the output directory."""
    from .solver import PBFSolver

    out = Path(output or config.output.directory)
    scene = build_scene(config)
    log.info("Scene '%s': %d fluid / %d boundary particles", config.name, len(scene.fluid), len(scene.boundary))

    solver = PBFSolver(config.fluid, config.solver, scene)
    export_meshes(scene, out)
    writer = FrameWriter(
        out,
        scene.domain_lower,
        scene.domain_upper,
        {
            "scene": config.name,
            "fluid": asdict(config.fluid),
            "solver": asdict(config.solver),
            "fps": config.output.fps,
            "particle_mass": solver.mass,
            "num_particles": solver.num_particles,
        },
    )

    frame_dt = 1.0 / config.output.fps
    num_frames = int(round(config.output.duration * config.output.fps))
    started = time.perf_counter()
    for index in range(num_frames + 1):
        substeps = 0 if index == 0 else solver.advance(frame_dt)
        info = FrameInfo(index, solver.time, substeps, solver.num_active)
        if config.output.write_particles:
            writer.write(info, solver.positions(), solver.velocities())
        if on_frame:
            on_frame(info, solver)
        elapsed = time.perf_counter() - started
        log.info(
            "frame %4d/%d  t=%.3fs  substeps=%d  active=%d  wall=%.1fs",
            index,
            num_frames,
            solver.time,
            substeps,
            solver.num_active,
            elapsed,
        )
    writer.metadata["wall_time_s"] = time.perf_counter() - started
    writer.close()
    return out
