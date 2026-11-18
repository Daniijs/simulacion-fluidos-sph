"""Build the render scene in Blender from a simulation output folder.

Run inside Blender (UI, command line or the ``bpy`` module):

    blender -b -P blender/build_scene.py -- --sim output/valley --out output/valley/scene.blend

The simulation uses metres with +Y up; Blender uses +Z up. Every mesh is
converted on load with (x, y, z) -> (x, -z, y), so world coordinates match.

Water is a mesh sequence: a frame-change handler (stored in the .blend as a
text block, see ``water_loader.py``) swaps in ``surface/frame_XXXXX.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from water_loader import load_water  # noqa: E402
from water_loader import register as register_water  # noqa: E402

# ---------------------------------------------------------------------------------------- helpers


def sim_to_blender(v: np.ndarray) -> np.ndarray:
    return np.c_[v[:, 0], -v[:, 2], v[:, 1]]


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal reader for the binary PLY files written by trimesh (xyz + triangle lists)."""
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode().splitlines()
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    elements, current = [], None
    for line in header:
        parts = line.split()
        if parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts[0] == "property":
            current["props"].append(parts[1:])
    offset = end
    vertices = faces = None
    for el in elements:
        if el["name"] == "vertex":
            dtype = np.dtype([(p[-1], types[p[0]]) for p in el["props"]])
            arr = np.frombuffer(data, dtype=dtype, count=el["count"], offset=offset)
            vertices = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
            offset += dtype.itemsize * el["count"]
        elif el["name"] == "face":
            dtype = np.dtype([("n", "u1"), ("idx", "<i4", 3)])
            arr = np.frombuffer(data, dtype=dtype, count=el["count"], offset=offset)
            faces = arr["idx"].astype(np.int64)
            offset += dtype.itemsize * el["count"]
    return vertices, faces


def mesh_object(name: str, vertices: np.ndarray, faces: np.ndarray, collection, smooth: bool = False):
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.astype(np.float32).ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.full(len(faces), smooth))
    mesh.update()
    mesh.validate()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def new_collection(name: str, parent=None):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def node_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    return mat, nt, out


def principled(nt, **inputs):
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    for key, value in inputs.items():
        bsdf.inputs[key].default_value = value
    return bsdf


def ramp(nt, stops: list[tuple[float, tuple]]):
    node = nt.nodes.new("ShaderNodeValToRGB")
    cr = node.color_ramp
    for i, (pos, color) in enumerate(stops):
        el = cr.elements[i] if i < len(cr.elements) else cr.elements.new(pos)
        el.position = pos
        el.color = (*color, 1.0) if len(color) == 3 else color
    return node


# --------------------------------------------------------------------------------------- materials


def terrain_material():
    """Grass on gentle ground, rock on steep slopes, damp earth by the river."""
    mat, nt, out = node_material("Terrain")
    L = nt.links
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep_n = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Normal"], sep_n.inputs[0])
    sep_p = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Position"], sep_p.inputs[0])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 8.0
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 160.0
    fine.inputs["Detail"].default_value = 4.0

    grass = ramp(nt, [(0.3, (0.10, 0.20, 0.04)), (0.55, (0.20, 0.33, 0.07)), (0.75, (0.32, 0.38, 0.12))])
    L.new(noise.outputs["Fac"], grass.inputs["Fac"])
    rock = ramp(nt, [(0.3, (0.13, 0.11, 0.09)), (0.7, (0.27, 0.24, 0.20))])
    L.new(fine.outputs["Fac"], rock.inputs["Fac"])
    earth = ramp(nt, [(0.3, (0.16, 0.11, 0.07)), (0.7, (0.28, 0.20, 0.13))])
    L.new(noise.outputs["Fac"], earth.inputs["Fac"])

    # steepness: 0 on flat ground, 1 on cliffs
    steep = nt.nodes.new("ShaderNodeMapRange")
    steep.inputs["From Min"].default_value = 0.72
    steep.inputs["From Max"].default_value = 0.42
    L.new(sep_n.outputs["Z"], steep.inputs["Value"])
    steep_n = nt.nodes.new("ShaderNodeMath")
    steep_n.operation = "ADD"
    L.new(steep.outputs[0], steep_n.inputs[0])
    jitter = nt.nodes.new("ShaderNodeMath")
    jitter.operation = "MULTIPLY"
    jitter.inputs[1].default_value = 0.35
    L.new(fine.outputs["Fac"], jitter.inputs[0])
    L.new(jitter.outputs[0], steep_n.inputs[1])
    steep_c = nt.nodes.new("ShaderNodeMath")
    steep_c.operation = "SUBTRACT"
    steep_c.use_clamp = True
    steep_c.inputs[1].default_value = 0.17
    L.new(steep_n.outputs[0], steep_c.inputs[0])

    # low ground near the river bed
    low = nt.nodes.new("ShaderNodeMapRange")
    low.inputs["From Min"].default_value = 0.035
    low.inputs["From Max"].default_value = 0.0
    L.new(sep_p.outputs["Z"], low.inputs["Value"])

    cells = nt.nodes.new("ShaderNodeTexVoronoi")
    cells.inputs["Scale"].default_value = 9.0
    cells.inputs["Randomness"].default_value = 0.6
    fields = ramp(
        nt,
        [(0.0, (0.16, 0.30, 0.06)), (0.3, (0.38, 0.40, 0.12)), (0.55, (0.26, 0.36, 0.08)), (0.8, (0.48, 0.42, 0.18))],
    )
    fields.color_ramp.interpolation = "CONSTANT"
    L.new(cells.outputs["Color"], fields.inputs["Fac"])
    plain = nt.nodes.new("ShaderNodeMapRange")  # 1 on the valley floor, 0 on the hills
    plain.inputs["From Min"].default_value = 0.09
    plain.inputs["From Max"].default_value = 0.06
    L.new(sep_p.outputs["Z"], plain.inputs["Value"])
    downstream = nt.nodes.new("ShaderNodeMapRange")
    downstream.inputs["From Min"].default_value = 1.35
    downstream.inputs["From Max"].default_value = 1.5
    L.new(sep_p.outputs["X"], downstream.inputs["Value"])
    field_mask = nt.nodes.new("ShaderNodeMath")
    field_mask.operation = "MULTIPLY"
    L.new(plain.outputs[0], field_mask.inputs[0])
    L.new(downstream.outputs[0], field_mask.inputs[1])
    ground = nt.nodes.new("ShaderNodeMix")
    ground.data_type = "RGBA"
    L.new(field_mask.outputs[0], ground.inputs["Factor"])
    L.new(grass.outputs["Color"], ground.inputs[6])
    L.new(fields.outputs["Color"], ground.inputs[7])

    mix1 = nt.nodes.new("ShaderNodeMix")
    mix1.data_type = "RGBA"
    L.new(low.outputs[0], mix1.inputs["Factor"])
    L.new(ground.outputs[2], mix1.inputs[6])
    L.new(earth.outputs["Color"], mix1.inputs[7])
    mix2 = nt.nodes.new("ShaderNodeMix")
    mix2.data_type = "RGBA"
    L.new(steep_c.outputs[0], mix2.inputs["Factor"])
    L.new(mix1.outputs[2], mix2.inputs[6])
    L.new(rock.outputs["Color"], mix2.inputs[7])

    bsdf = principled(nt, Roughness=0.9)
    L.new(mix2.outputs[2], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.002
    L.new(fine.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def concrete_material():
    mat, nt, out = node_material("Concrete")
    L = nt.links
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 90.0
    noise.inputs["Detail"].default_value = 10.0
    color = ramp(nt, [(0.35, (0.48, 0.47, 0.44)), (0.65, (0.66, 0.65, 0.61))])
    L.new(noise.outputs["Fac"], color.inputs["Fac"])
    bsdf = principled(nt, Roughness=0.8)
    L.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.2
    bump.inputs["Distance"].default_value = 0.001
    L.new(noise.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def random_color_material(name: str, stops, roughness: float):
    """Each object picks its own shade from ``stops`` (Object Info > Random)."""
    mat, nt, out = node_material(name)
    info = nt.nodes.new("ShaderNodeObjectInfo")
    color = ramp(nt, stops)
    color.color_ramp.interpolation = "CONSTANT"
    nt.links.new(info.outputs["Random"], color.inputs["Fac"])
    bsdf = principled(nt, Roughness=roughness)
    nt.links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def water_material():
    """Clear water with depth absorption; fast, aerated water turns white."""
    mat, nt, out = node_material("Water")
    L = nt.links
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "velocity"
    length = nt.nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    L.new(attr.outputs["Vector"], length.inputs[0])
    foam = nt.nodes.new("ShaderNodeMapRange")
    foam.inputs["From Min"].default_value = 1.6
    foam.inputs["From Max"].default_value = 3.4
    L.new(length.outputs["Value"], foam.inputs["Value"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 140.0
    noise.inputs["Detail"].default_value = 6.0
    breakup = nt.nodes.new("ShaderNodeMath")
    breakup.operation = "MULTIPLY"
    breakup.use_clamp = True
    L.new(foam.outputs[0], breakup.inputs[0])
    shaped = nt.nodes.new("ShaderNodeMapRange")
    shaped.inputs["From Min"].default_value = 0.35
    shaped.inputs["From Max"].default_value = 0.7
    shaped.inputs["To Max"].default_value = 1.6
    L.new(noise.outputs["Fac"], shaped.inputs["Value"])
    L.new(shaped.outputs[0], breakup.inputs[1])

    glass = principled(nt, Roughness=0.03, IOR=1.333)
    glass.inputs["Transmission Weight"].default_value = 1.0
    glass.inputs["Base Color"].default_value = (1, 1, 1, 1)
    white = principled(nt, Roughness=0.45)
    white.inputs["Base Color"].default_value = (0.85, 0.88, 0.9, 1)
    white.inputs["Subsurface Weight"].default_value = 0.3
    mix = nt.nodes.new("ShaderNodeMixShader")
    L.new(breakup.outputs[0], mix.inputs["Fac"])
    L.new(glass.outputs[0], mix.inputs[1])
    L.new(white.outputs[0], mix.inputs[2])
    # Without refractive caustics no light would reach the river bed. Letting shadow
    # rays pass through the surface is the usual production trick.
    light_path = nt.nodes.new("ShaderNodeLightPath")
    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    shadow_mix = nt.nodes.new("ShaderNodeMixShader")
    L.new(light_path.outputs["Is Shadow Ray"], shadow_mix.inputs["Fac"])
    L.new(mix.outputs[0], shadow_mix.inputs[1])
    L.new(transparent.outputs[0], shadow_mix.inputs[2])
    L.new(shadow_mix.outputs[0], out.inputs["Surface"])

    volume = nt.nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Color"].default_value = (0.62, 0.66, 0.52, 1)  # silt-laden water
    volume.inputs["Density"].default_value = 1.0
    volume.inputs["Absorption Color"].default_value = (0.62, 0.82, 0.76, 1)
    L.new(volume.outputs[0], out.inputs["Volume"])
    return mat


def base_material():
    """Plinth of the diorama: matte charcoal paint."""
    mat, nt, out = node_material("DioramaBase")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 60.0
    color = ramp(nt, [(0.3, (0.035, 0.035, 0.038)), (0.7, (0.06, 0.06, 0.065))])
    nt.links.new(noise.outputs["Fac"], color.inputs["Fac"])
    bsdf = principled(nt, Roughness=0.75)
    nt.links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def table_material():
    """Dark oak workbench: grain from noise stretched along the planks."""
    mat, nt, out = node_material("Table")
    L = nt.links
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.6, 30.0, 1.0)
    L.new(coord.outputs["Object"], mapping.inputs["Vector"])
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 3.0
    grain.inputs["Detail"].default_value = 12.0
    grain.inputs["Roughness"].default_value = 0.6
    L.new(mapping.outputs["Vector"], grain.inputs["Vector"])
    color = ramp(nt, [(0.35, (0.055, 0.03, 0.018)), (0.65, (0.13, 0.075, 0.04))])
    L.new(grain.outputs["Fac"], color.inputs["Fac"])
    bsdf = principled(nt, Roughness=0.4)
    bsdf.inputs["Coat Weight"].default_value = 0.3
    L.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# ------------------------------------------------------------------------------------------ trees


def tree_prototypes(collection):
    """Two miniature tree models (a conifer and a round broadleaf), Z-up."""
    leaves = random_color_material(
        "Foliage", [(0.0, (0.05, 0.13, 0.03)), (0.4, (0.09, 0.2, 0.05)), (0.75, (0.16, 0.26, 0.06))], 0.8
    )
    bark = random_color_material("Bark", [(0.0, (0.12, 0.08, 0.05))], 0.9)
    protos = []

    bpy.ops.mesh.primitive_cone_add(vertices=10, radius1=0.0065, depth=0.028, location=(0, 0, 0.018))
    crown = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.0012, depth=0.008, location=(0, 0, 0.004))
    trunk = bpy.context.active_object
    protos.append(_join_tree("TreeConifer", crown, trunk, leaves, bark, collection))

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.0075, location=(0, 0, 0.016))
    crown = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.0013, depth=0.01, location=(0, 0, 0.005))
    trunk = bpy.context.active_object
    protos.append(_join_tree("TreeRound", crown, trunk, leaves, bark, collection))
    return protos


def _join_tree(name, crown, trunk, leaves, bark, collection):
    crown.data.materials.append(leaves)
    trunk.data.materials.append(bark)
    trunk.data.polygons.foreach_set("material_index", [0] * len(trunk.data.polygons))
    bpy.ops.object.select_all(action="DESELECT")
    crown.select_set(True)
    trunk.select_set(True)
    bpy.context.view_layer.objects.active = crown
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    collection.objects.link(obj)
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def scatter_trees(terrain, prototypes_collection, density: float = 2600.0, seed: int = 4):
    """Geometry-nodes forest on the slopes, kept clear of the reservoir and the flood plain."""
    v = np.empty(len(terrain.data.vertices) * 3)
    terrain.data.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    n = np.empty(len(terrain.data.vertices) * 3)
    terrain.data.vertices.foreach_get("normal", n)
    n = n.reshape(-1, 3)
    height = v[:, 2]
    upstream = v[:, 0] < 1.2
    weight = np.clip((height - np.where(upstream, 0.47, 0.2)) / 0.08, 0, 1)
    weight *= np.clip((n[:, 2] - 0.55) / 0.2, 0, 1)  # no trees on cliffs
    weight *= np.clip((0.95 - height) / 0.15, 0, 1)  # thin out on the summits
    attr = terrain.data.attributes.new("tree_density", "FLOAT", "POINT")
    attr.data.foreach_set("value", weight.astype(np.float32))

    tree = bpy.data.node_groups.new("Forest", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = tree.nodes, tree.links
    gin, gout = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    named = N.new("GeometryNodeInputNamedAttribute")
    named.data_type = "FLOAT"
    named.inputs["Name"].default_value = "tree_density"
    dist = N.new("GeometryNodeDistributePointsOnFaces")
    dist.inputs[4].default_value = density  # Density
    dist.inputs[6].default_value = seed  # Seed
    L.new(gin.outputs[0], dist.inputs["Mesh"])
    L.new(named.outputs["Attribute"], dist.inputs[5])  # Density Factor
    col_info = N.new("GeometryNodeCollectionInfo")
    col_info.inputs["Collection"].default_value = prototypes_collection
    col_info.inputs["Separate Children"].default_value = True
    col_info.inputs["Reset Children"].default_value = True
    inst = N.new("GeometryNodeInstanceOnPoints")
    inst.inputs["Pick Instance"].default_value = True
    L.new(dist.outputs["Points"], inst.inputs["Points"])
    L.new(col_info.outputs[0], inst.inputs["Instance"])
    rand_rot = N.new("FunctionNodeRandomValue")
    rand_rot.data_type = "FLOAT_VECTOR"
    rand_rot.inputs[1].default_value = (0.08, 0.08, 6.283)  # vector Max
    L.new(rand_rot.outputs["Value"], inst.inputs["Rotation"])
    rand_scale = N.new("FunctionNodeRandomValue")
    rand_scale.data_type = "FLOAT"
    rand_scale.inputs[2].default_value = 0.6
    rand_scale.inputs[3].default_value = 1.35
    L.new(rand_scale.outputs[1], inst.inputs["Scale"])
    pick = N.new("FunctionNodeRandomValue")
    pick.data_type = "INT"
    pick.inputs[4].default_value = 0
    pick.inputs[5].default_value = 1
    pick.inputs[8].default_value = 11  # Seed
    L.new(pick.outputs[2], inst.inputs["Instance Index"])
    join = N.new("GeometryNodeJoinGeometry")
    L.new(gin.outputs[0], join.inputs[0])
    L.new(inst.outputs[0], join.inputs[0])
    L.new(join.outputs[0], gout.inputs[0])
    mod = terrain.modifiers.new("Forest", "NODES")
    mod.node_group = tree


# ------------------------------------------------------------------------------------- world & camera


def setup_world(sun_elevation: float = 40.0, sun_rotation: float = 120.0):
    world = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    for kind in ("MULTIPLE_SCATTERING", "NISHITA", "SINGLE_SCATTERING"):
        try:
            sky.sky_type = kind
            break
        except TypeError:
            continue
    sky.sun_elevation = math.radians(sun_elevation)
    sky.sun_rotation = math.radians(sun_rotation)
    sky.sun_disc = False
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.35
    nt.links.new(sky.outputs[0], bg.inputs[0])
    # Camera rays see a dark studio backdrop; lighting still comes from the sky.
    backdrop = nt.nodes.new("ShaderNodeBackground")
    backdrop.inputs["Color"].default_value = (0.018, 0.02, 0.024, 1)
    path = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(backdrop.outputs[0], mix.inputs[2])
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 4.5
    sun_data.angle = math.radians(1.5)
    sun_data.color = (1.0, 0.95, 0.87)
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(90 - sun_elevation), 0, math.radians(sun_rotation - 180))


def setup_camera(focal_mm: float = 38.0, f_stop: float = 2.0):
    """Camera aimed at ``CamTarget``, focused on ``Focus`` (both animated later)."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = focal_mm
    cam_data.clip_start = 0.01
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = f_stop
    cam_data.dof.aperture_blades = 7
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    target = bpy.data.objects.new("CamTarget", None)
    focus = bpy.data.objects.new("Focus", None)
    for empty in (target, focus):
        bpy.context.scene.collection.objects.link(empty)
    cam_data.dof.focus_object = focus
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    bpy.context.scene.camera = cam
    return cam, target, focus


# Camera path in Blender coordinates (x downstream, z up): wide establishing shot,
# push in on the dam as it fails, then crane down and follow the flood into the village.
CAMERA_KEYS = [
    # (phase, location, aim point, focal length)
    (0.0, (4.55, 0.50, 2.25), (1.25, 0.00, 0.25), 36.0),
    (1.0, (3.95, 0.35, 1.70), (1.15, 0.00, 0.20), 38.0),
    (2.0, (3.55, -0.10, 1.15), (1.75, 0.00, 0.08), 42.0),
    (3.0, (3.45, -0.30, 1.00), (2.20, 0.05, 0.03), 44.0),
]


def animate_camera(cam, target, focus, fronts: list[dict], dam_x: float = 1.0, focus_limit_x: float = 2.4) -> None:
    """Keyframe the camera path and make the focus follow the flood front."""
    n = len(fronts)
    xs = np.array([f["x"] for f in fronts])
    moving = np.nonzero(xs > dam_x + 0.03)[0]
    release = int(moving[0]) if len(moving) else n // 4
    phase_frames = [1, release + 1, release + 1 + int(0.4 * (n - release)), n]
    for (_, loc, aim, lens), frame in zip(CAMERA_KEYS, phase_frames, strict=True):
        cam.location = loc
        cam.keyframe_insert("location", frame=frame)
        cam.data.lens = lens
        cam.data.keyframe_insert("lens", frame=frame)
        target.location = aim
        target.keyframe_insert("location", frame=frame)

    # focus: smoothed flood front, held on the dam until the water moves
    pts = np.array([[f["x"], -f["z"], f["y"]] for f in fronts])
    pts[: release + 1] = [dam_x, 0.0, 0.25]
    pts[:, 0] = np.minimum(pts[:, 0], focus_limit_x)  # keep focus in the village once the wave has passed
    kernel = np.ones(9) / 9
    smooth = np.stack([np.convolve(np.pad(pts[:, k], 4, mode="edge"), kernel, "valid") for k in range(3)], 1)
    for i, p in enumerate(smooth):
        focus.location = p
        focus.keyframe_insert("location", frame=i + 1)


def spray_object(collection):
    """Droplets: one vertex per spray particle, drawn as small white spheres."""
    mat, nt, out = node_material("Spray")
    bsdf = principled(nt, Roughness=0.35)
    bsdf.inputs["Base Color"].default_value = (0.92, 0.95, 0.97, 1)
    bsdf.inputs["Subsurface Weight"].default_value = 0.5
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    obj = bpy.data.objects.new("Spray", bpy.data.meshes.new("Spray"))
    collection.objects.link(obj)
    tree = bpy.data.node_groups.new("SprayPoints", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    to_points = tree.nodes.new("GeometryNodeMeshToPoints")
    to_points.inputs["Radius"].default_value = 0.0011
    set_mat = tree.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    tree.links.new(gin.outputs[0], to_points.inputs["Mesh"])
    tree.links.new(to_points.outputs[0], set_mat.inputs["Geometry"])
    tree.links.new(set_mat.outputs[0], gout.inputs[0])
    obj.modifiers.new("Points", "NODES").node_group = tree
    return obj


# ---------------------------------------------------------------------------------------- build


def build(sim_dir: Path, preview: bool = False) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    meta = json.loads((sim_dir / "simulation.json").read_text())

    world_col = new_collection("Diorama")
    trees_col = new_collection("TreePrototypes")
    water_col = new_collection("Water")

    mats = {
        "terrain": terrain_material(),
        "concrete": concrete_material(),
        "walls": random_color_material(
            "HouseWalls", [(0.0, (0.80, 0.76, 0.66)), (0.33, (0.88, 0.85, 0.78)), (0.66, (0.74, 0.62, 0.48))], 0.85
        ),
        "roof": random_color_material(
            "Roof", [(0.0, (0.42, 0.11, 0.05)), (0.4, (0.55, 0.18, 0.08)), (0.8, (0.30, 0.25, 0.23))], 0.7
        ),
        "base": base_material(),
    }
    for path in sorted((sim_dir / "meshes").glob("*.ply")):
        name = path.stem
        verts, faces = read_ply(path)
        obj = mesh_object(name, sim_to_blender(verts), faces, world_col)
        if name == "terrain":
            key = "terrain"
        elif name == "terrain_base":
            key = "base"
        elif name.startswith("dam") or name == "bridge":
            key = "concrete"
        elif name.endswith("_roof"):
            key = "roof"
        else:
            key = "walls"
        obj.data.materials.append(mats[key])
        if name == "terrain":
            for poly in obj.data.polygons:
                poly.use_smooth = True
            terrain = obj

    protos = tree_prototypes(trees_col)
    trees_col.hide_render = True
    trees_col.hide_viewport = True
    del protos
    scatter_trees(terrain, trees_col)

    bpy.ops.mesh.primitive_plane_add(size=12, location=(1.6, 0, -0.15))
    table = bpy.context.active_object
    table.name = "Table"
    table.data.materials.append(table_material())

    water = bpy.data.objects.new("Water", bpy.data.meshes.new("Water"))
    water_col.objects.link(water)
    water.data.materials.append(water_material())
    water["sim_dir"] = str(sim_dir)

    spray = spray_object(water_col)
    spray["sim_dir"] = str(sim_dir)

    setup_world()
    cam, target, focus = setup_camera()
    front_file = sim_dir / "front.json"
    if front_file.exists():
        animate_camera(cam, target, focus, json.loads(front_file.read_text()))

    frames = len(meta["frames"])
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.fps = 30
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64 if preview else 256
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.volume_bounces = 2
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.5
    scene.render.resolution_x, scene.render.resolution_y = (960, 540) if preview else (3840, 2160)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"
    scene.render.filepath = "//render/frame_"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"

    register_water()
    load_water(scene)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    build(args.sim.resolve(), args.preview)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
