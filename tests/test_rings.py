import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lunar_beams import BeamModel, PoseTable, elevation_deg, load_cloud, list_frames, select_rings, to_sensor_frame
from lunar_beams.rings import loam_ring_index
from tests.synthetic import OFFSET, make_scene, scan

MODEL = BeamModel.nominal(128, 27.0, -25.0)


def test_rings_recovered_exactly_in_sensor_frame():
    R = Rotation.from_euler("zyx", [30, 4, -3], degrees=True).as_matrix()
    pts, ring_true = scan(R, np.array([0, 0, -0.3]))
    ring, resid = MODEL.assign(elevation_deg(pts))
    assert np.array_equal(ring, ring_true)
    assert np.abs(resid).max() < 1e-3


def test_wrong_offset_is_detectably_blurry():
    pts, _ = scan(np.eye(3), np.array([0, 0, -0.3]))
    good = MODEL.sharpness(elevation_deg(pts))["median_abs_resid"]
    bad = MODEL.sharpness(elevation_deg(pts + OFFSET))["median_abs_resid"]
    assert good < 0.01 and bad > 0.1


@pytest.mark.parametrize("frame", ["sensor", "rover", "world"])
def test_each_frame_round_trips(tmp_path, frame):
    make_scene(tmp_path / "S", frame, n_frames=4)
    poses = PoseTable.from_scene(tmp_path / "S")
    for f in list_frames(tmp_path / "S"):
        R, p, gap = poses.at(float(f.stem))
        assert gap == 0
        xyz = load_cloud(f)[:, :3]
        e = elevation_deg(to_sensor_frame(xyz, frame, OFFSET, R, p))
        assert MODEL.sharpness(e)["median_abs_resid"] < 0.01


def test_select_every_k_and_angles():
    assert len(select_rings(MODEL, {"type": "every_k", "k": 4})) == 32
    vlp = select_rings(MODEL, {"type": "angles", "angles_deg": list(range(-15, 16, 2))})
    assert len(vlp) == 16
    assert np.abs(MODEL.angles_deg[vlp] - np.arange(15, -16, -2)).max() < 0.21
    with pytest.raises(ValueError):
        select_rings(MODEL, {"type": "angles", "angles_deg": [-30.67]})


def test_loam_ring_index_bottom_is_zero():
    kept = np.array([0, 4, 8])            # top-first convention
    out = loam_ring_index(kept, np.array([0, 4, 8]))
    assert list(out) == [2, 1, 0]
