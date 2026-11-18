"""Frame-change handler that streams the reconstructed water surface into Blender.

The object named ``Water`` carries a custom property ``sim_dir``; for frame
``f`` the handler loads ``<sim_dir>/surface/frame_{f - frame_start:05d}.npz``
and writes vertices, triangles, normals and the ``velocity`` attribute used by
Cycles for motion blur. Copy this file into the .blend as a text block named
``water_loader.py`` with "Register" enabled to make the file self-contained.
"""

from pathlib import Path

import bpy
import numpy as np

_cache: dict[str, int] = {}


def _surface_path(scene, obj) -> Path:
    sim_dir = Path(bpy.path.abspath(obj["sim_dir"]))
    index = max(scene.frame_current - scene.frame_start, 0)
    return sim_dir / "surface" / f"frame_{index:05d}.npz"


def load_water(scene, *_args) -> None:
    obj = bpy.data.objects.get("Water")
    if obj is None or "sim_dir" not in obj:
        return
    path = _surface_path(scene, obj)
    if _cache.get(obj.name) == scene.frame_current or not path.exists():
        return
    data = np.load(path)
    # simulation (x, y, z) with +Y up  ->  Blender (x, -z, y) with +Z up
    v = data["vertices"]
    verts = np.c_[v[:, 0], -v[:, 2], v[:, 1]].astype(np.float32)
    tris = data["triangles"].astype(np.int32)
    vel = data["velocity"].astype(np.float32)
    vel = np.c_[vel[:, 0], -vel[:, 2], vel[:, 1]]

    old = obj.data
    mesh = bpy.data.meshes.new("Water")
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.ravel())
    mesh.loops.add(tris.size)
    mesh.loops.foreach_set("vertex_index", tris.ravel())
    mesh.polygons.add(len(tris))
    mesh.polygons.foreach_set("loop_start", np.arange(0, tris.size, 3, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(tris), dtype=bool))
    mesh.update()
    attr = mesh.attributes.new("velocity", "FLOAT_VECTOR", "POINT")
    attr.data.foreach_set("vector", vel.ravel())
    for mat in old.materials:
        mesh.materials.append(mat)
    obj.data = mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    _cache[obj.name] = scene.frame_current


def register() -> None:
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "load_water":
            handlers.remove(h)
    handlers.append(load_water)


if __name__ == "__main__":
    register()
