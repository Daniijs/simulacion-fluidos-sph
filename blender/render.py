"""Render the animation of a scene built by build_scene.py (run with ``blender -b scene.blend -P render.py``).

Frames that already exist on disk are skipped, so an interrupted render resumes.
"""

import argparse
import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from water_loader import load_water, register  # noqa: E402


def enable_gpu() -> str:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            bpy.context.scene.cycles.device = "GPU"
            return kind
    bpy.context.scene.cycles.device = "CPU"
    return "CPU"


def wait_for_surface(scene, frame: int, timeout: float = 3600.0) -> None:
    sim_dir = Path(bpy.path.abspath(bpy.data.objects["Water"]["sim_dir"]))
    path = sim_dir / "surface" / f"frame_{frame - scene.frame_start:05d}.npz"
    waited = 0.0
    while not path.exists() and waited < timeout:
        time.sleep(5.0)
        waited += 5.0
    time.sleep(2.0)  # let the writer finish the file


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="all")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--percentage", type=int, default=100)
    parser.add_argument("--wait", action="store_true", help="wait for each surface file (render while reconstructing)")
    args = parser.parse_args(argv)

    scene = bpy.context.scene
    register()
    print("Render device:", enable_gpu())
    scene.cycles.samples = args.samples
    scene.render.resolution_percentage = args.percentage
    first, last = scene.frame_start, scene.frame_end
    if args.frames != "all":
        first, last = (int(v) for v in args.frames.split("-"))

    out_dir = Path(bpy.path.abspath("//render"))
    out_dir.mkdir(exist_ok=True)
    for frame in range(first, last + 1):
        target = out_dir / f"frame_{frame:04d}.png"
        if target.exists():
            continue
        if args.wait:
            wait_for_surface(scene, frame)
        scene.frame_set(frame)
        load_water(scene)
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        print(f"Rendered {target.name}", flush=True)


if __name__ == "__main__":
    main()
