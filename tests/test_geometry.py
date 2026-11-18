import numpy as np
import trimesh
from scipy.spatial import cKDTree

from sphfluid.geometry import Heightfield, box_walls, lattice_in_box, mesh_surface, remove_near


def test_lattice_fills_box_with_expected_count():
    pts = lattice_in_box([0, 0, 0], [0.1, 0.2, 0.05], 0.01)
    assert len(pts) == 10 * 20 * 5
    assert pts.min() > 0 and pts[:, 1].max() < 0.2


def test_box_walls_leave_no_holes():
    spacing = 0.01
    walls = box_walls([0, 0, 0], [0.3, 0.2, 0.1], spacing, layers=1, open_top=True)
    rng = np.random.default_rng(1)
    probe = np.c_[rng.uniform(0, 0.3, 500), np.zeros(500), rng.uniform(0, 0.1, 500)]  # floor
    dist, _ = cKDTree(walls).query(probe)
    assert dist.max() < spacing


def test_mesh_surface_covers_mesh_with_requested_spacing():
    spacing = 0.01
    mesh = trimesh.creation.box([0.2, 0.1, 0.15])
    pts = mesh_surface(mesh, spacing, layers=1)
    probe, _ = trimesh.sample.sample_surface(mesh, 2000, seed=3)
    dist, _ = cKDTree(pts).query(probe)
    assert dist.max() < spacing


def test_heightfield_samples_follow_steep_slopes_without_gaps():
    xs = np.linspace(0, 1, 51)
    heights = np.repeat((3.0 * xs**2)[:, None], 51, axis=1)  # slope up to 6:1
    terrain = Heightfield(0.0, 0.0, 0.02, heights)
    spacing = 0.01
    pts = terrain.surface_particles(spacing, layers=1)
    rng = np.random.default_rng(2)
    x, z = rng.uniform(0, 1, 3000), rng.uniform(0, 1, 3000)
    probe = np.c_[x, terrain.height_at(x, z), z]
    dist, _ = cKDTree(pts).query(probe)
    assert dist.max() < spacing


def test_remove_near_drops_only_close_points():
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    kept = remove_near(pts, np.array([[0.05, 0, 0]]), 0.1)
    np.testing.assert_array_equal(kept, [[1.0, 0, 0]])
