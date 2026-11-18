"""Analytic solid colliders.

Boundary particles give the fluid the right density next to walls, but under
a large hydrostatic head they cannot stop particles from being squeezed
through thin solids. Colliders are the hard guarantee: after every position
update, a particle found inside a collider is projected to its nearest face.

Two primitives cover the scenes: oriented boxes (houses, bridge, tower) and
sections of an arch dam. A collider may belong to a gate group, in which case
it is switched off when the gate opens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOX_PARAMS = 9  # cx, cy, cz, hx, hy, hz, yaw, group, unused
ARCH_PARAMS = 12  # cx, cz, radius, theta0, theta1, y0, y1, t_top, t_base, crest, group, unused


@dataclass(frozen=True)
class BoxCollider:
    center: tuple[float, float, float]
    half_extent: tuple[float, float, float]
    yaw: float = 0.0  # rotation around +y, radians
    group: int = 0

    def pack(self) -> np.ndarray:
        return np.array([*self.center, *self.half_extent, self.yaw, self.group, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class ArchCollider:
    """Arch dam section. In plan: x = cx - r cos(theta), z = cz + r sin(theta)."""

    center_x: float
    center_z: float
    radius: float
    theta0: float
    theta1: float
    y0: float
    y1: float
    thickness_top: float
    thickness_base: float
    crest: float
    group: int = 0

    def thickness(self, y: np.ndarray | float) -> np.ndarray:
        s = np.clip((self.crest - np.asarray(y)) / self.crest, 0.0, 1.0)
        return self.thickness_top + (self.thickness_base - self.thickness_top) * s

    def pack(self) -> np.ndarray:
        return np.array(
            [
                self.center_x,
                self.center_z,
                self.radius,
                self.theta0,
                self.theta1,
                self.y0,
                self.y1,
                self.thickness_top,
                self.thickness_base,
                self.crest,
                self.group,
                0.0,
            ],
            dtype=np.float32,
        )


Collider = BoxCollider | ArchCollider


def pack_colliders(colliders: list[Collider]) -> tuple[np.ndarray, np.ndarray]:
    boxes = [c.pack() for c in colliders if isinstance(c, BoxCollider)]
    arches = [c.pack() for c in colliders if isinstance(c, ArchCollider)]
    box_arr = np.array(boxes, dtype=np.float32).reshape(-1, BOX_PARAMS)
    arch_arr = np.array(arches, dtype=np.float32).reshape(-1, ARCH_PARAMS)
    return box_arr, arch_arr
