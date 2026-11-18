"""Throughput of the original class project (v1) versus sphfluid (v2).

Both run the same workload on the CPU: particles advanced through one second of
simulated time. The metric is particle-steps per second (particles x solver
steps / wall time), so different time-step choices do not skew the result.

    python benchmarks/compare_v1.py --particles 4325
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bench_v1(num_particles: int, steps: int) -> float:
    sys.path.insert(0, str(ROOT / "legacy" / "v1"))
    from Dinamica import Dinamica
    from GestorColision import GestorColision
    from MetodoEuler import MetodoEuler
    from Particula import Particula
    from SistemaParticulas import SistemaParticulas
    from SpatialHashing import SpatialHashing
    from vpython import vector

    h, spacing = 0.06, 0.02
    system = SistemaParticulas(capacidad_inicial=num_particles)
    side = int(np.ceil(num_particles ** (1 / 3)))
    for k in range(num_particles):
        i, j, l = k % side, (k // side) % side, k // side**2
        pos = vector(-1.0 + i * spacing, 0.05 + j * spacing, -0.5 + l * spacing)
        system.addParticula(Particula(pos, vector(0, 0, 0), 0.008, 0.012, visual=False))
    hashing = SpatialHashing(cell_size=h)
    dynamics = Dinamica(system, h=h, densidad_reposo=1000.0, viscosidad=0.4, k_presion=5000.0)
    gravity = vector(0, -9.81, 0)

    start = time.perf_counter()
    for _ in range(steps):
        hashing.update(system.pos, system.num_activas)
        dynamics.calculaFuerzaSPH(hashing)
        MetodoEuler.integra(system, 0.0005, gravity)
        GestorColision.colisionLimites([-1.2, 3.2], [0.0, 1.8], [-0.7, 0.7], 0.02, system)
    return num_particles * steps / (time.perf_counter() - start)


def bench_v2(num_particles: int, steps: int, arch: str) -> float:
    import taichi as ti

    from sphfluid.config import FluidParams, SolverParams
    from sphfluid.geometry import BOUNDARY_SPACING_RATIO, box_walls, lattice_in_box
    from sphfluid.scene import SceneGeometry
    from sphfluid.solver import PBFSolver

    ti.init(arch=getattr(ti, arch), log_level=ti.WARN)
    spacing = 0.01
    side = int(np.ceil(num_particles ** (1 / 3)))
    fluid = lattice_in_box([0, 0, 0], [side * spacing] * 3, spacing)[:num_particles]
    lower, upper = np.zeros(3), np.array([side * spacing * 3, side * spacing * 2, side * spacing])
    walls = box_walls(lower, upper, spacing * BOUNDARY_SPACING_RATIO, 2)
    scene = SceneGeometry(fluid=fluid, boundary=walls, domain_lower=lower, domain_upper=upper)
    solver = PBFSolver(FluidParams(particle_spacing=spacing), SolverParams(), scene)
    solver.step(1e-4)  # compile kernels outside the timed region
    ti.sync()
    start = time.perf_counter()
    for _ in range(steps):
        solver.step(1.0 / 240)
    ti.sync()
    return num_particles * steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=4325)
    parser.add_argument("--v1-steps", type=int, default=20)
    parser.add_argument("--v2-steps", type=int, default=200)
    parser.add_argument("--arch", default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    v1 = bench_v1(args.particles, args.v1_steps)
    v2 = bench_v2(args.particles, args.v2_steps, args.arch)
    print(json.dumps({"particles": args.particles, "arch": args.arch, "v1_particle_steps_per_s": round(v1),
                      "v2_particle_steps_per_s": round(v2), "speedup": round(v2 / v1, 1)}, indent=2))


if __name__ == "__main__":
    main()
