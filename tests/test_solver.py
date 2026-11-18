"""Physical sanity checks of the PBF solver on small CPU problems."""

import numpy as np
import pytest

from sphfluid.colliders import BoxCollider
from sphfluid.config import FluidParams, SolverParams
from sphfluid.geometry import BOUNDARY_SPACING_RATIO, box_walls, lattice_in_box
from sphfluid.scene import Gate, SceneGeometry
from sphfluid.solver import PBFSolver

SPACING = 0.01
FLUID = FluidParams(particle_spacing=SPACING)


def tank(water_height=0.12, gate=None):
    lower, upper = np.zeros(3), np.array([0.2, 0.3, 0.1])
    fluid = lattice_in_box(lower, [0.2 if gate is None else 0.08, water_height, 0.1], SPACING)
    walls = box_walls(lower, upper, SPACING * BOUNDARY_SPACING_RATIO, 2)
    gates = []
    if gate is not None:
        plane = lattice_in_box([0.08, 0, 0], [0.08 + 1e-6, 0.3, 0.1], SPACING * BOUNDARY_SPACING_RATIO)
        collider = BoxCollider((0.085, 0.15, 0.05), (0.005, 0.15, 0.06), group=1)
        gates = [Gate(plane, gate, "gate", (collider,))]
    return SceneGeometry(fluid=fluid, boundary=walls, domain_lower=lower, domain_upper=upper, gates=gates)


def test_initial_density_equals_rest_density_inside_and_at_walls():
    solver = PBFSolver(FLUID, SolverParams(), tank())
    rho = solver.measure_density()
    x = solver.positions()
    below_surface = x[:, 1] < 0.12 - 2 * SPACING
    assert np.median(rho[below_surface]) == pytest.approx(1000.0, rel=1e-3)
    assert np.percentile(rho[below_surface], 99) < 1000.0 * 1.02


def test_water_at_rest_settles_and_keeps_its_volume():
    solver = PBFSolver(FLUID, SolverParams(), tank())
    start = solver.positions().copy()
    for _ in range(60):  # one second: the initial lattice relaxes into a natural packing
        solver.advance(1 / 60)
    speed = np.linalg.norm(solver.velocities(), axis=1)
    assert np.percentile(speed, 99) < 0.06
    assert speed.max() < 0.15
    level_change = solver.positions()[:, 1].max() - start[:, 1].max()
    assert abs(level_change) < 0.5 * SPACING  # < 4 % compression of a 12 cm column


def test_fluid_never_leaves_a_closed_tank():
    solver = PBFSolver(FLUID, SolverParams(), tank(gate=0.0))
    for _ in range(40):
        solver.advance(1 / 60)
    x = solver.positions()
    assert solver.num_active == solver.num_particles
    assert (x >= 0).all() and (x[:, 0] <= 0.2).all() and (x[:, 2] <= 0.1).all()


def test_gate_holds_water_until_it_opens():
    solver = PBFSolver(FLUID, SolverParams(), tank(gate=0.2))
    for _ in range(9):  # 0.15 s, gate still closed
        solver.advance(1 / 60)
    assert solver.positions()[:, 0].max() < 0.08 + SPACING
    for _ in range(12):  # gate opened at 0.2 s
        solver.advance(1 / 60)
    assert solver.positions()[:, 0].max() > 0.15


def test_neighbour_grid_matches_brute_force():
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(4)
    pts = rng.uniform(0, 0.1, (400, 3))
    scene = SceneGeometry(fluid=pts, boundary=np.zeros((0, 3)), domain_lower=np.zeros(3), domain_upper=np.full(3, 0.1))
    solver = PBFSolver(FLUID, SolverParams(), scene)
    rho = solver.measure_density()
    x = solver.positions().astype(np.float64)

    from sphfluid.kernels import poly6

    tree = cKDTree(x)
    expected = np.array(
        [
            solver.mass
            * poly6(
                np.linalg.norm(x[tree.query_ball_point(p, FLUID.support_radius)] - p, axis=1), FLUID.support_radius
            ).sum()
            for p in x
        ]
    )
    np.testing.assert_allclose(rho, expected, rtol=1e-3)
