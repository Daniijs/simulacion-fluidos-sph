"""Classic 2D dam break of Martin & Moyce (1952), used for validation.

A water column of width ``a`` and height ``n² a`` collapses along a dry tank.
The tank is a thin slab, so the flow is effectively two-dimensional.
"""

from __future__ import annotations

import numpy as np

from ..config import SceneConfig
from ..geometry import BOUNDARY_SPACING_RATIO, box_walls, lattice_in_box
from ..scene import SceneGeometry, register


@register("dam_break")
def build(config: SceneConfig) -> SceneGeometry:
    opts = config.scene
    a = float(opts.get("column_width", 0.1))
    n2 = float(opts.get("height_ratio", 2.0))
    length = float(opts.get("tank_length", 8.0)) * a
    depth = float(opts.get("depth", 0.4)) * a
    height = float(opts.get("tank_height", 3.0)) * a
    dx = config.fluid.particle_spacing

    lower = np.zeros(3)
    upper = np.array([length, height, depth])
    fluid = lattice_in_box(lower, [a, n2 * a, depth], dx)
    walls = box_walls(lower, upper, dx * BOUNDARY_SPACING_RATIO, config.solver.boundary_layers, open_top=True)
    return SceneGeometry(fluid=fluid, boundary=walls, domain_lower=lower, domain_upper=upper)
