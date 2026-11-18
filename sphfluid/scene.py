"""Scene description shared by the solver, the exporters and the renderer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .colliders import Collider
from .config import SceneConfig
from .geometry import Heightfield


@dataclass(frozen=True)
class Gate:
    """Solid that disappears at ``open_time`` (e.g. a dam breach): its boundary
    particles and its colliders (which must carry the gate's group id)."""

    particles: np.ndarray
    open_time: float
    name: str = "gate"
    colliders: tuple[Collider, ...] = ()


@dataclass
class SceneGeometry:
    """Everything a solver needs to start, plus the meshes the renderer needs."""

    fluid: np.ndarray
    boundary: np.ndarray
    domain_lower: np.ndarray
    domain_upper: np.ndarray
    fluid_velocity: np.ndarray | None = None
    gates: list[Gate] = field(default_factory=list)
    sink_height: float = -np.inf  # particles that fall below this are retired
    terrain: Heightfield | None = None
    meshes: dict[str, trimesh.Trimesh] = field(default_factory=dict)
    colliders: list[Collider] = field(default_factory=list)


SceneBuilder = Callable[[SceneConfig], SceneGeometry]
_BUILDERS: dict[str, SceneBuilder] = {}


def register(name: str) -> Callable[[SceneBuilder], SceneBuilder]:
    def decorator(fn: SceneBuilder) -> SceneBuilder:
        _BUILDERS[name] = fn
        return fn

    return decorator


def build_scene(config: SceneConfig) -> SceneGeometry:
    from . import scenes  # noqa: F401  (registers the built-in scenes)

    try:
        builder = _BUILDERS[config.builder]
    except KeyError:
        raise ValueError(f"Unknown scene builder '{config.builder}'. Available: {sorted(_BUILDERS)}") from None
    return builder(config)
