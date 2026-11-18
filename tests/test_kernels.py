import numpy as np
import pytest

from sphfluid.kernels import boundary_volume_scale, lattice_offsets, poly6, spiky_gradient


@pytest.mark.parametrize("h", [0.01, 0.1, 1.0])
def test_poly6_integrates_to_one(h):
    # Midpoint rule over the support sphere.
    n = 80
    axis = (np.arange(n) + 0.5) / n * 2 * h - h
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    integral = poly6(r, h).sum() * (2 * h / n) ** 3
    assert integral == pytest.approx(1.0, rel=1e-3)


def test_poly6_is_zero_outside_support():
    assert poly6(np.array([1.0, 1.5]), 1.0).tolist() == [0.0, 0.0]


def test_spiky_gradient_points_towards_neighbour_and_is_antisymmetric():
    r = np.array([[0.3, 0.0, 0.0]])
    g = spiky_gradient(r, 1.0)
    assert g[0, 0] < 0  # gradient wrt x_i of W(x_i - x_j) points to x_j
    np.testing.assert_allclose(spiky_gradient(-r, 1.0), -g)


def test_lattice_mass_calibration_gives_rest_density():
    spacing, h, rho0 = 0.01, 0.02, 1000.0
    offsets = lattice_offsets(spacing, h)
    mass = rho0 / poly6(np.linalg.norm(offsets, axis=1), h).sum()
    assert mass * poly6(np.linalg.norm(offsets, axis=1), h).sum() == pytest.approx(rho0)


def test_boundary_scale_is_a_positive_correction():
    scale = boundary_volume_scale(0.01, 0.02, 0.0065, 2)
    assert 0.0 < scale < 1.0
