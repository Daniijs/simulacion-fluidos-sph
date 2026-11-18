"""Command line interface: ``python -m sphfluid <command>``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> None:
    from .config import load_config
    from .simulation import init_backend, run

    init_backend(args.arch)
    out = run(load_config(args.scene), args.output)
    print(f"Frames written to {out}")


def _cmd_surface(args: argparse.Namespace) -> None:
    from .surface import reconstruct_directory

    reconstruct_directory(args.directory, resolution=args.resolution, start=args.start, end=args.end)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sphfluid", description="GPU fluid simulation with Position Based Fluids.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate a scene file")
    run.add_argument("scene", type=Path)
    run.add_argument("-o", "--output", type=Path)
    run.add_argument("--arch", default="auto", choices=["auto", "gpu", "cpu", "cuda", "vulkan"])
    run.set_defaults(func=_cmd_run)

    surface = sub.add_parser("surface", help="reconstruct water surface meshes from stored frames")
    surface.add_argument("directory", type=Path)
    surface.add_argument("--resolution", type=float, default=0.5, help="voxel size in particle radii")
    surface.add_argument("--start", type=int, default=0)
    surface.add_argument("--end", type=int)
    surface.set_defaults(func=_cmd_surface)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
