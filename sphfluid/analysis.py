"""Post-processing of stored frames: flood-front tracking for the camera."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .io import load_metadata, read_frame


def flood_front(directory: str | Path, quantile: float = 99.7, band: float = 0.08) -> list[dict]:
    """Leading edge of the flow along +x for every frame.

    Returns, per frame, the front position ``x`` (a high quantile, robust to
    stray droplets) and the mean ``y``/``z`` of the water within ``band`` metres
    behind it, i.e. where the camera should focus.
    """
    directory = Path(directory)
    meta = load_metadata(directory)
    fronts = []
    for path in sorted((directory / "particles").glob("frame_*.npz")):
        x, _, t = read_frame(path, meta["domain_lower"], meta["domain_upper"])
        front_x = float(np.percentile(x[:, 0], quantile))
        near = x[x[:, 0] > front_x - band]
        fronts.append({"time": t, "x": front_x, "y": float(near[:, 1].mean()), "z": float(near[:, 2].mean())})
    (directory / "front.json").write_text(json.dumps(fronts, indent=1))
    return fronts
