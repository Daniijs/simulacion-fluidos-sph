"""Compact on-disk format for simulation frames.

Each frame is an ``.npz`` file with positions quantised to 16 bits inside the
domain box (sub-millimetre precision for a few metres of domain) and velocities
stored as float16. That is 12 bytes per particle instead of 24.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

QUANT = 65535.0


@dataclass(frozen=True)
class FrameInfo:
    index: int
    time: float
    substeps: int
    num_particles: int


class FrameWriter:
    def __init__(self, directory: str | Path, lower: np.ndarray, upper: np.ndarray, metadata: dict):
        self.directory = Path(directory)
        self.frames_dir = self.directory / "particles"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.extent = np.asarray(upper, dtype=np.float64) - self.lower
        self.metadata = {
            **metadata,
            "domain_lower": self.lower.tolist(),
            "domain_upper": (self.lower + self.extent).tolist(),
        }
        self.frames: list[FrameInfo] = []

    def write(self, info: FrameInfo, positions: np.ndarray, velocities: np.ndarray) -> Path:
        q = np.clip((positions - self.lower) / self.extent, 0.0, 1.0) * QUANT
        path = self.frames_dir / f"frame_{info.index:05d}.npz"
        np.savez(path, position=np.round(q).astype(np.uint16), velocity=velocities.astype(np.float16), time=info.time)
        self.frames.append(info)
        return path

    def close(self) -> None:
        summary = {**self.metadata, "frames": [asdict(f) for f in self.frames]}
        (self.directory / "simulation.json").write_text(json.dumps(summary, indent=2))


def read_frame(path: str | Path, lower, upper) -> tuple[np.ndarray, np.ndarray, float]:
    """Return positions (float32), velocities (float32) and time of a stored frame."""
    data = np.load(path)
    lower = np.asarray(lower, dtype=np.float64)
    extent = np.asarray(upper, dtype=np.float64) - lower
    positions = (data["position"].astype(np.float64) / QUANT * extent + lower).astype(np.float32)
    return positions, data["velocity"].astype(np.float32), float(data["time"])


def load_metadata(directory: str | Path) -> dict:
    return json.loads((Path(directory) / "simulation.json").read_text())
