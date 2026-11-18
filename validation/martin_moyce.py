"""Compare the simulated dam-break surge front with Martin & Moyce (1952).

Experimental data: Phil. Trans. R. Soc. A 244, Table 2 (n² = 2), first record.
Z = z / a is the surge front position and T = n t sqrt(g / a) the scaled time.

The paper normalises its time axis at Z = 1.44 (the instant of release could
not be measured precisely), so the simulated curve is shifted in time to pass
through that same point before comparing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sphfluid.io import load_metadata, read_frame

EXPERIMENT_T = np.array([1.19, 1.56, 1.87, 2.26, 2.57, 2.88, 3.25, 3.59, 3.89, 4.20, 4.54, 4.85])
EXPERIMENT_Z = np.array([1.44, 1.89, 2.33, 2.78, 3.22, 3.67, 4.11, 4.56, 5.00, 5.44, 5.89, 6.33])
GRAVITY = 9.81


def surge_front(directory: Path, column_width: float, height_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    meta = load_metadata(directory)
    spacing = meta["fluid"]["particle_spacing"]
    times, fronts = [], []
    for path in sorted((directory / "particles").glob("frame_*.npz")):
        x, _, t = read_frame(path, meta["domain_lower"], meta["domain_upper"])
        near_floor = x[x[:, 1] < 0.25 * column_width]
        # Leading edge: furthest particle that still belongs to the bulk flow.
        front = np.percentile(near_floor[:, 0], 99.9) + 0.5 * spacing
        times.append(t)
        fronts.append(front)
    n = np.sqrt(height_ratio)
    return n * np.array(times) * np.sqrt(GRAVITY / column_width), np.array(fronts) / column_width


def align(T: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Shift simulated time so that Z = 1.44 happens at T = 1.19, as in the paper."""
    return T + (EXPERIMENT_T[0] - np.interp(EXPERIMENT_Z[0], Z, T))


def compare(T: np.ndarray, Z: np.ndarray) -> dict:
    T = align(T, Z)
    simulated = np.interp(EXPERIMENT_T, T, Z)
    error = simulated - EXPERIMENT_Z
    return {
        "mean_abs_error_Z": float(np.mean(np.abs(error))),
        "max_abs_error_Z": float(np.max(np.abs(error))),
        "mean_relative_error": float(np.mean(np.abs(error) / EXPERIMENT_Z)),
    }


def plot(runs: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors = ["#1f77b4", "#2ec4b6", "#6a4c93"]
    for (label, T, Z), color in zip(runs, colors, strict=False):
        T = align(T, Z)
        ax.plot(T, Z, color=color, lw=2.2, label=label)
    ax.scatter(EXPERIMENT_T, EXPERIMENT_Z, color="#d62728", zorder=3, s=28, label="Martin & Moyce (1952)")
    ax.set_xlabel(r"$T = n\,t\,\sqrt{g/a}$")
    ax.set_ylabel(r"$Z = z/a$")
    ax.set_xlim(0, 5.5)
    ax.set_ylim(1, 8)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Dam break: posición del frente de la ola")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--column-width", type=float, default=0.1)
    parser.add_argument("--height-ratio", type=float, default=2.0)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    runs = []
    for directory in args.directories:
        T, Z = surge_front(directory, args.column_width, args.height_ratio)
        spacing = load_metadata(directory)["fluid"]["particle_spacing"]
        label = f"sphfluid, dx = {spacing * 1000:g} mm"
        runs.append((label, T, Z))
        print(label, json.dumps(compare(T, Z)))
    if args.plot:
        plot(runs, args.plot)


if __name__ == "__main__":
    main()
