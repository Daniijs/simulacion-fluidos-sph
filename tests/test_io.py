import numpy as np

from sphfluid.io import FrameInfo, FrameWriter, load_metadata, read_frame


def test_frames_round_trip_with_sub_millimetre_error(tmp_path):
    lower, upper = np.array([0.0, -1.0, 0.0]), np.array([3.0, 1.0, 1.5])
    rng = np.random.default_rng(0)
    x = rng.uniform(lower, upper, (1000, 3))
    v = rng.normal(size=(1000, 3))
    writer = FrameWriter(tmp_path, lower, upper, {"fluid": {"particle_spacing": 0.01}})
    path = writer.write(FrameInfo(0, 0.5, 3, 1000), x, v)
    writer.close()

    meta = load_metadata(tmp_path)
    x2, v2, t = read_frame(path, meta["domain_lower"], meta["domain_upper"])
    assert t == 0.5
    assert np.abs(x2 - x).max() < 1e-4
    np.testing.assert_allclose(v2, v, rtol=2e-3, atol=2e-3)
