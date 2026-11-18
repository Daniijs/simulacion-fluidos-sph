"""Breakdown passes for the explainer video (run with ``blender -b scene.blend -P passes.py -- ...``).

* ``particles``: the raw simulation particles as small spheres coloured by speed,
* ``clay``: the reconstructed surface with a plain matte material,
* ``final``: the normal render.

Each pass hides what it does not need, renders the frame range into
``//passes/<pass>/`` and skips frames that already exist.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import enable_gpu  # noqa: E402
from water_loader import _replace_mesh, _to_blender, load_water, register  # noqa: E402

QUANT = 65535.0


def particles_object(sim_dir: Path):
    """Point cloud coloured by speed (geometry nodes + attribute-driven emission)."""
    mat = bpy.data.materials.new("ParticleSpeed")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "speed"
    attr.attribute_type = "GEOMETRY"
    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.inputs["From Max"].default_value = 3.5
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].color = (0.02, 0.12, 0.45, 1)
    cr.elements[1].color = (0.9, 0.97, 1.0, 1)
    mid = cr.elements.new(0.55)
    mid.color = (0.05, 0.65, 0.62, 1)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.5
    nt.links.new(attr.outputs["Fac"], rng.inputs["Value"])
    nt.links.new(rng.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Emission Color"])
    bsdf.inputs["Emission Strength"].default_value = 0.25
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    obj = bpy.data.objects.new("Particles", bpy.data.meshes.new("Particles"))
    bpy.context.scene.collection.objects.link(obj)
    tree = bpy.data.node_groups.new("ParticlePoints", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = tree.nodes.new("NodeGroupInput"), tree.nodes.new("NodeGroupOutput")
    points = tree.nodes.new("GeometryNodeMeshToPoints")
    meta = json.loads((sim_dir / "simulation.json").read_text())
    points.inputs["Radius"].default_value = 0.45 * meta["fluid"]["particle_spacing"]
    set_mat = tree.nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value = mat
    tree.links.new(gin.outputs[0], points.inputs["Mesh"])
    tree.links.new(points.outputs[0], set_mat.inputs["Geometry"])
    tree.links.new(set_mat.outputs[0], gout.inputs[0])
    obj.modifiers.new("Points", "NODES").node_group = tree
    return obj, meta


def load_particles(obj, sim_dir: Path, meta: dict, frame_index: int, keep: float = 0.5) -> None:
    data = np.load(sim_dir / "particles" / f"frame_{frame_index:05d}.npz")
    lo = np.asarray(meta["domain_lower"])
    hi = np.asarray(meta["domain_upper"])
    x = data["position"].astype(np.float64) / QUANT * (hi - lo) + lo
    v = data["velocity"].astype(np.float32)
    rng = np.random.default_rng(frame_index)
    sel = rng.random(len(x)) < keep
    mesh = _replace_mesh(obj, _to_blender(x[sel]))
    attr = mesh.attributes.new("speed", "FLOAT", "POINT")
    attr.data.foreach_set("value", np.linalg.norm(v[sel], axis=1).astype(np.float32))


def clay_material():
    mat = bpy.data.materials.new("Clay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.75, 0.78, 0.8, 1)
    bsdf.inputs["Roughness"].default_value = 0.55
    return mat


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="kind", choices=["particles", "clay", "final"], required=True)
    parser.add_argument("--frames", required=True, help="start-end")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--percentage", type=int, default=50)
    args = parser.parse_args(argv)

    scene = bpy.context.scene
    register()
    print("Render device:", enable_gpu())
    scene.cycles.samples = args.samples
    scene.render.resolution_percentage = args.percentage
    water, spray = bpy.data.objects["Water"], bpy.data.objects["Spray"]
    sim_dir = Path(bpy.path.abspath(water["sim_dir"]))
    particles = meta = None
    if args.kind == "particles":
        particles, meta = particles_object(sim_dir)
        water.hide_render = spray.hide_render = True
    elif args.kind == "clay":
        clay = clay_material()
        water.data.materials.clear()
        water.data.materials.append(clay)

    out_dir = Path(bpy.path.abspath(f"//passes/{args.kind}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    first, last = (int(v) for v in args.frames.split("-"))
    for frame in range(first, last + 1):
        target = out_dir / f"frame_{frame:04d}.png"
        if target.exists():
            continue
        scene.frame_set(frame)
        load_water(scene)
        if args.kind == "clay":
            water.data.materials.clear()
            water.data.materials.append(clay)
        if particles is not None:
            load_particles(particles, sim_dir, meta, frame - scene.frame_start)
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {args.kind} {target.name}", flush=True)


if __name__ == "__main__":
    main()
