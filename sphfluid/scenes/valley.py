"""Miniature valley: an arch dam fails and floods a village downstream.

Everything is built procedurally from a seed, at tabletop scale (metres):

* a heightfield terrain with a reservoir basin, a narrow gorge and a winding
  river through a flood plain,
* an arch dam across the gorge whose central section is the breach,
* a village of gabled houses, a church tower and a bridge over the river.

The same meshes are exported for rendering, so the render matches the
simulation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..colliders import ArchCollider, BoxCollider, Collider
from ..config import SceneConfig
from ..geometry import BOUNDARY_SPACING_RATIO, Heightfield, lattice_in_box, mesh_surface, remove_near
from ..scene import Gate, SceneGeometry, register

# Layout (x downstream, y up, z across the valley)
LENGTH, HALF_WIDTH = 3.2, 0.8
DAM_X = 0.95
DAM_CREST = 0.44
WATER_LEVEL = 0.40
BASE_BOTTOM = -0.15
ARCH_RADIUS = 0.75
ARCH_SPAN = 0.85  # half angle of the arch in plan, radians
DAM_BASE = -0.1
THICKNESS_TOP, THICKNESS_BASE = 0.03, 0.08


@dataclass(frozen=True)
class ValleyLayout:
    seed: int = 3
    breach_half_angle: float = 0.16  # radians of arch removed by the breach
    breach_bottom: float = 0.03
    houses: int = 40


def _smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def river_center(x: np.ndarray) -> np.ndarray:
    return 0.13 * np.sin(2.3 * x - 1.2) * _smoothstep(x, 1.15, 1.7)


def valley_half_width(x: np.ndarray) -> np.ndarray:
    reservoir, gorge, plain = 0.55, 0.22, 0.42
    w = reservoir + (gorge - reservoir) * _smoothstep(x, 0.45, DAM_X)
    return w + (plain - gorge) * _smoothstep(x, DAM_X + 0.1, 1.6)


def floor_height(x: np.ndarray) -> np.ndarray:
    basin = 0.05
    plain = 0.03 - 0.06 * (x - DAM_X) / (LENGTH - DAM_X)
    return np.where(x < DAM_X, basin, basin + (plain - basin) * _smoothstep(x, DAM_X, DAM_X + 0.35))


def _relief(x, z, rng):
    out = np.zeros_like(x)
    for octave in range(5):
        freq = 2.5 * 2**octave
        amp = 0.02 / 1.8**octave
        kx, kz = rng.normal(size=2) * freq
        out += amp * np.sin(kx * x + kz * z + rng.uniform(0, 2 * np.pi))
    return out


def build_terrain(seed: int, cell: float = 0.01) -> Heightfield:
    rng = np.random.default_rng(seed)
    xs = np.arange(0.0, LENGTH + 1e-9, cell)
    zs = np.arange(-HALF_WIDTH, HALF_WIDTH + 1e-9, cell)
    x, z = np.meshgrid(xs, zs, indexing="ij")

    d = np.abs(z - river_center(x))
    w = valley_half_width(x)
    channel = -0.025 * (1 - _smoothstep(d, 0.03, 0.07)) * _smoothstep(x, DAM_X, DAM_X + 0.3)
    steepness = 0.28 - 0.1 * np.exp(-(((x - DAM_X - 0.1) / 0.3) ** 2))  # steep walls at the gorge
    slope = np.clip((d - w) / steepness, 0, None)
    height_scale = 0.75 - 0.33 * _smoothstep(x, 1.3, 2.0)  # gentler hills around the village
    mountains = height_scale * slope**1.5 / (1 + 0.6 * slope**1.5)
    back_wall = 0.7 * (1 - _smoothstep(x, 0.0, 0.25))  # closes the reservoir upstream
    rough = _relief(x, z, rng) * (0.25 + _smoothstep(d - w, 0.0, 0.2))
    heights = floor_height(x) + channel + mountains + back_wall + rough
    return Heightfield(origin_x=0.0, origin_z=-HALF_WIDTH, cell=cell, heights=heights)


# --- meshes -------------------------------------------------------------------------------------


def _loft(inner: np.ndarray, outer: np.ndarray) -> trimesh.Trimesh:
    """Closed mesh between two (n_u, n_v, 3) grids sharing their parametrisation."""
    nu, nv, _ = inner.shape
    verts = np.concatenate([inner.reshape(-1, 3), outer.reshape(-1, 3)])
    off = nu * nv
    idx = np.arange(off).reshape(nu, nv)
    faces = []

    def quad_strip(a, b, c, d, flip):
        tris = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
        tris = np.concatenate(tris)
        return tris[:, ::-1] if flip else tris

    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces.append(quad_strip(a, b, c, d, False))
    faces.append(quad_strip(a + off, b + off, c + off, d + off, True))
    for edge in (idx[0, :], idx[-1, :], idx[:, 0], idx[:, -1]):
        e0, e1 = edge[:-1], edge[1:]
        faces.append(quad_strip(e0, e1, e1 + off, e0 + off, False))
    mesh = trimesh.Trimesh(verts, np.concatenate(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def arch_dam_section(theta0: float, theta1: float, y0: float, y1: float, segments: int = 24) -> trimesh.Trimesh:
    """Section of an arch dam (convex upstream) between two plan angles and heights."""
    radius = ARCH_RADIUS
    center = np.array([DAM_X + radius, 0.0])
    thetas = np.linspace(theta0, theta1, max(2, segments))
    ys = np.linspace(y0, y1, 12)
    th, yy = np.meshgrid(thetas, ys, indexing="ij")
    thickness = THICKNESS_TOP + (THICKNESS_BASE - THICKNESS_TOP) * np.clip((DAM_CREST - yy) / DAM_CREST, 0, 1)

    def face(r_offset):
        r = radius + r_offset
        x = center[0] - r * np.cos(th)
        z = center[1] + r * np.sin(th)
        return np.stack([x, yy, z], axis=-1)

    return _loft(face(-thickness / 2), face(thickness / 2))


def dam_colliders(layout: ValleyLayout, gate_group: int) -> tuple[list[Collider], list[Collider]]:
    """Colliders of the standing dam and of the breach section (switched off on failure)."""
    b = layout.breach_half_angle

    def arch(t0, t1, y0, y1, group=0):
        return ArchCollider(
            DAM_X + ARCH_RADIUS, 0.0, ARCH_RADIUS, t0, t1, y0, y1, THICKNESS_TOP, THICKNESS_BASE, DAM_CREST, group
        )

    standing = [
        arch(-ARCH_SPAN, -b, DAM_BASE, DAM_CREST),
        arch(b, ARCH_SPAN, DAM_BASE, DAM_CREST),
        arch(-b, b, DAM_BASE, layout.breach_bottom),
    ]
    return standing, [arch(-b, b, layout.breach_bottom, DAM_CREST, gate_group)]


def dam_meshes(layout: ValleyLayout) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh]]:
    """Return the standing part of the dam and the breach split into chunks."""
    span = ARCH_SPAN
    b = layout.breach_half_angle
    standing = trimesh.util.concatenate(
        [
            arch_dam_section(-span, -b, DAM_BASE, DAM_CREST, 40),
            arch_dam_section(b, span, DAM_BASE, DAM_CREST, 40),
            arch_dam_section(-b, b, DAM_BASE, layout.breach_bottom, 8),
        ]
    )
    chunks = []
    angle_cuts = np.linspace(-b, b, 5)
    height_cuts = np.linspace(layout.breach_bottom, DAM_CREST, 6)
    for t0, t1 in zip(angle_cuts[:-1], angle_cuts[1:], strict=True):
        for y0, y1 in zip(height_cuts[:-1], height_cuts[1:], strict=True):
            chunks.append(arch_dam_section(t0, t1, y0, y1, 4))
    return standing, chunks


def house(width: float, depth: float, wall: float, roof: float) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Walls and gabled roof of a house standing on y = 0, as separate meshes."""
    body = trimesh.creation.box([width, wall, depth])
    body.apply_translation([0, wall / 2, 0])
    hw, hd = width / 2 * 1.08, depth / 2 * 1.1
    v = np.array(
        [
            [-hw, wall, -hd],
            [hw, wall, -hd],
            [hw, wall, hd],
            [-hw, wall, hd],
            [-hw, wall + roof, 0],
            [hw, wall + roof, 0],
        ]
    )
    f = np.array([[0, 1, 5], [0, 5, 4], [3, 4, 5], [3, 5, 2], [0, 4, 3], [1, 2, 5], [0, 3, 2], [0, 2, 1]])
    roof_mesh = trimesh.Trimesh(v, f, process=True)
    trimesh.repair.fix_normals(roof_mesh)
    return body, roof_mesh


def place_village(terrain: Heightfield, layout: ValleyLayout) -> tuple[dict[str, trimesh.Trimesh], list[Collider]]:
    rng = np.random.default_rng(layout.seed + 11)
    meshes: dict[str, trimesh.Trimesh] = {}
    colliders: list[Collider] = []
    placed: list[tuple[float, float, float]] = []
    attempts = 0
    while len(placed) < layout.houses and attempts < 4000:
        attempts += 1
        x = rng.uniform(1.55, 2.95)
        side = rng.choice([-1, 1])
        z = river_center(np.array([x]))[0] + side * rng.uniform(0.11, valley_half_width(np.array([x]))[0] - 0.04)
        size = rng.uniform(0.035, 0.065)
        if any((x - px) ** 2 + (z - pz) ** 2 < (0.6 * (size + ps)) ** 2 for px, pz, ps in placed):
            continue
        if abs(x - 2.15) < 0.09:  # keep the bridge approach clear
            continue
        w, d = size, size * rng.uniform(0.6, 0.85)
        wall, roof = size * rng.uniform(0.55, 0.8), size * rng.uniform(0.35, 0.5)
        yaw = rng.uniform(-0.3, 0.3) + (np.pi / 2) * rng.integers(0, 2)
        walls, roof_mesh = house(w, d, wall, roof)
        footprint = np.array([[x + dx, z + dz] for dx in (-w / 2, w / 2) for dz in (-w / 2, w / 2)])
        base = terrain.height_at(footprint[:, 0], footprint[:, 1]).min() - 0.01
        for part in (walls, roof_mesh):
            part.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 1, 0]))
            part.apply_translation([x, base, z])
        height = wall + 0.5 * roof
        colliders.append(BoxCollider((x, base + height / 2, z), (w / 2, height / 2, d / 2), yaw))
        meshes[f"house_{len(placed):02d}"] = walls
        meshes[f"house_{len(placed):02d}_roof"] = roof_mesh
        placed.append((x, z, size))

    # Church: tall tower with a pyramid roof, near the river bend.
    cx = 2.45
    cz = river_center(np.array([cx]))[0] - 0.2
    tower = trimesh.creation.box([0.05, 0.16, 0.05])
    tower.apply_translation([0, 0.08, 0])
    spire = trimesh.creation.cone(radius=0.04, height=0.07, sections=4)
    spire.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    spire.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [0, 1, 0]))
    spire.apply_translation([0, 0.16, 0])
    nave, nave_roof = house(0.13, 0.07, 0.07, 0.04)
    nave.apply_translation([0.09, 0, 0])
    nave_roof.apply_translation([0.09, 0, 0])
    church = trimesh.util.concatenate([tower, nave])
    church_roof = trimesh.util.concatenate([spire, nave_roof])
    church_base = terrain.height_at(np.array([cx]), np.array([cz]))[0] - 0.01
    for part in (church, church_roof):
        part.apply_translation([cx, church_base, cz])
    meshes["church"] = church
    meshes["church_roof"] = church_roof
    colliders.append(BoxCollider((cx, church_base + 0.09, cz), (0.025, 0.09, 0.025)))
    colliders.append(BoxCollider((cx + 0.09, church_base + 0.045, cz), (0.065, 0.045, 0.035)))

    # Bridge over the river: deck on three piers.
    bx = 2.15
    zc = river_center(np.array([bx]))[0]
    floor = terrain.height_at(np.array([bx]), np.array([zc]))[0]
    deck_y = floor + 0.075
    span = 0.34
    deck = trimesh.creation.box([0.06, 0.014, span])
    deck.apply_translation([bx, deck_y, zc])
    parts = [deck]
    colliders.append(BoxCollider((bx, deck_y, zc), (0.03, 0.007, span / 2)))
    for dz in (-span / 3, 0.0, span / 3):
        pier_bottom = terrain.height_at(np.array([bx]), np.array([zc + dz]))[0] - 0.02
        pier = trimesh.creation.box([0.03, deck_y - pier_bottom, 0.022])
        pier.apply_translation([bx, (deck_y + pier_bottom) / 2, zc + dz])
        parts.append(pier)
        colliders.append(
            BoxCollider((bx, (deck_y + pier_bottom) / 2, zc + dz), (0.015, (deck_y - pier_bottom) / 2, 0.011))
        )
    meshes["bridge"] = trimesh.util.concatenate(parts)
    return meshes, colliders


# --- scene builder ------------------------------------------------------------------------------


@register("valley")
def build(config: SceneConfig) -> SceneGeometry:
    opts = config.scene
    layout = ValleyLayout(
        seed=int(opts.get("seed", 3)),
        breach_half_angle=float(opts.get("breach_half_angle", 0.16)),
        houses=int(opts.get("houses", 40)),
    )
    dx = config.fluid.particle_spacing
    bs = dx * BOUNDARY_SPACING_RATIO
    layers = config.solver.boundary_layers

    terrain = build_terrain(layout.seed)
    standing, chunks = dam_meshes(layout)
    village, village_colliders = place_village(terrain, layout)
    dam_static, dam_gate = dam_colliders(layout, gate_group=1)
    breach = trimesh.util.concatenate(chunks)

    solids = [terrain.surface_particles(bs, layers), mesh_surface(standing, bs, layers)]
    solids += [mesh_surface(m, bs, layers) for m in village.values()]
    boundary = np.concatenate(solids)
    gate = mesh_surface(breach, bs, layers)

    # Reservoir: lattice below the water level, above the terrain, upstream of the dam.
    lattice = lattice_in_box([0.0, 0.0, -HALF_WIDTH], [DAM_X + 0.2, WATER_LEVEL, HALF_WIDTH], dx)
    above_ground = lattice[:, 1] > terrain.height_at(lattice[:, 0], lattice[:, 2]) + 0.5 * dx
    upstream = np.hypot(lattice[:, 0] - (DAM_X + ARCH_RADIUS), lattice[:, 2]) > ARCH_RADIUS + 0.045
    fluid = remove_near(lattice[above_ground & upstream], np.concatenate([boundary, gate]), 0.75 * dx)

    meshes = {"terrain": terrain.to_mesh(), "terrain_base": terrain.skirt_mesh(BASE_BOTTOM), "dam": standing, **village}
    meshes.update({f"dam_chunk_{i:02d}": c for i, c in enumerate(chunks)})

    return SceneGeometry(
        fluid=fluid,
        boundary=boundary,
        gates=[Gate(gate, float(opts.get("breach_time", 0.25)), "breach", tuple(dam_gate))],
        colliders=dam_static + village_colliders,
        domain_lower=np.array([0.0, BASE_BOTTOM - 0.05, -HALF_WIDTH]),
        domain_upper=np.array([LENGTH + 0.25, 0.95, HALF_WIDTH]),
        sink_height=BASE_BOTTOM,
        terrain=terrain,
        meshes=meshes,
    )
