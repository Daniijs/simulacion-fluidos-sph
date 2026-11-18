"""Simulation configuration loaded from TOML scene files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class FluidParams:
    """Physical properties of the fluid and its discretisation."""

    rest_density: float = 1000.0
    particle_spacing: float = 0.01
    kernel_ratio: float = 2.0  # support radius h = kernel_ratio * particle_spacing

    @property
    def support_radius(self) -> float:
        return self.kernel_ratio * self.particle_spacing


@dataclass(frozen=True)
class SolverParams:
    """Position Based Fluids solver settings (Macklin & Müller, 2013)."""

    gravity: Vec3 = (0.0, -9.81, 0.0)
    iterations: int = 4
    cfl: float = 0.4
    max_dt: float = 1.0 / 240.0
    relaxation: float = 1e-3  # constraint softness relative to a full neighbourhood
    tensile_k: float = 1e-3  # artificial pressure strength (s_corr)
    tensile_n: int = 4
    tensile_dq: float = 0.2  # reference distance for s_corr, in units of h
    xsph: float = 0.02  # XSPH velocity smoothing
    vorticity: float = 0.0  # vorticity confinement strength
    max_correction: float = 0.5  # largest position correction per iteration, in particle spacings
    max_speed: float = 20.0  # m/s; a safety net, far above any physical speed in the scenes
    boundary_layers: int = 2


@dataclass(frozen=True)
class OutputParams:
    """What to write and how often."""

    fps: float = 60.0
    duration: float = 2.0
    directory: str = "output"
    write_particles: bool = True


@dataclass(frozen=True)
class SceneConfig:
    """Complete description of a simulation run."""

    name: str
    builder: str
    fluid: FluidParams = field(default_factory=FluidParams)
    solver: SolverParams = field(default_factory=SolverParams)
    output: OutputParams = field(default_factory=OutputParams)
    scene: dict[str, Any] = field(default_factory=dict)  # builder-specific options


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        ftype = known[key].type
        target = globals().get(ftype) if isinstance(ftype, str) else ftype
        if is_dataclass(target) and isinstance(value, dict):
            value = _from_dict(target, value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> SceneConfig:
    """Read a scene description from a TOML file."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    scene = data.pop("scene", {})
    config = _from_dict(SceneConfig, data)
    return SceneConfig(**{**config.__dict__, "scene": scene})
