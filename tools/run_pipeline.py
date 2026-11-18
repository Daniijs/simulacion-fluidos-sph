"""End-to-end pipeline: simulate -> surface -> camera track -> Blender scene -> render.

Every stage is resumable: finished frames are skipped. Progress is written to
``<out>/status.json`` so a long run can be monitored from outside.

    python tools/run_pipeline.py scenes/valley.toml --out output/valley \\
        --blender "C:/Program Files/Blender Foundation/Blender 5.3/blender.exe"
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = ("sim", "surface", "front", "scene", "render")


class Status:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"started": time.time()}

    def update(self, **kwargs) -> None:
        self.data.update(kwargs, updated=time.time())
        self.path.write_text(json.dumps(self.data, indent=1))


def run(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = Status(out / "status.json")
    stages = args.stages.split(",")

    if "sim" in stages and not (out / "simulation.json").exists():
        from sphfluid.config import load_config
        from sphfluid.simulation import init_backend
        from sphfluid.simulation import run as simulate

        init_backend(args.arch)
        status.update(stage="sim")
        simulate(load_config(args.scene), out, on_frame=lambda info, _: status.update(stage="sim", frame=info.index))

    if "surface" in stages:
        from sphfluid.surface import reconstruct_directory

        status.update(stage="surface")
        reconstruct_directory(
            out, resolution=args.surface_resolution, on_frame=lambda i: status.update(stage="surface", frame=i)
        )

    if "front" in stages:
        from sphfluid.analysis import flood_front

        flood_front(out)

    blend = out / "scene.blend"
    if "scene" in stages:
        status.update(stage="scene")
        subprocess.run(
            [
                args.blender,
                "-b",
                "--factory-startup",
                "-P",
                str(ROOT / "blender" / "build_scene.py"),
                "--",
                "--sim",
                str(out),
                "--out",
                str(blend),
            ]
            + (["--preview"] if args.preview else []),
            check=True,
        )

    if "render" in stages:
        status.update(stage="render")
        cmd = [
            args.blender,
            "-b",
            str(blend),
            "-P",
            str(ROOT / "blender" / "render.py"),
            "--",
            "--frames",
            args.frames,
            "--samples",
            str(args.samples),
            "--percentage",
            str(args.percentage),
        ]
        subprocess.run(cmd, check=True)
    status.update(stage="done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--surface-resolution", type=float, default=0.5)
    parser.add_argument("--frames", default="all", help="'all' or 'start-end'")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--percentage", type=int, default=100, help="render resolution percentage")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
