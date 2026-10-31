"""Generate a fake LuSNAR-style scene (AirSim-like 128-beam scans of a sloped plane).

Used by the tests, and handy for trying the pipeline before the real data finishes downloading:
    python -m tests.synthetic --out data/fake --frame rover
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

OFFSET = np.array([1.0, 0.0, -1.5])


def scan(R_wr: np.ndarray, p_wr: np.ndarray, n_beams=128, upper=27.0, lower=-25.0,
         n_az=360, max_range=30.0, ground_z=0.0, rng=None):
    """Return (sensor-frame points, ring) for one sweep hitting the plane z_world = ground_z (NED)."""
    elev = np.radians(np.linspace(upper, lower, n_beams))
    az = np.radians(np.arange(n_az) * 360.0 / n_az)
    E, A = np.meshgrid(elev, az, indexing="ij")
    d_s = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), -np.sin(E)], -1).reshape(-1, 3)
    ring = np.repeat(np.arange(n_beams), n_az)
    o_w = p_wr + R_wr @ OFFSET
    d_w = d_s @ R_wr.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ground_z - o_w[2]) / d_w[:, 2]
    hit = (t > 0.5) & (t <= max_range)
    pts_s = d_s[hit] * t[hit, None]
    return pts_s, ring[hit]


def make_scene(out: Path, frame: str, n_frames=12, seed=0) -> None:
    rng = np.random.default_rng(seed)
    (out / "LiDAR").mkdir(parents=True, exist_ok=True)
    pose_rows = []
    for i in range(n_frames):
        t_ns = 1_700_000_000_000_000_000 + i * 100_000_000  # 10 Hz
        p = np.array([2.0 * i, 0.5 * i, -0.3])               # rover centre ~0.3 m above ground (NED)
        rot = Rotation.from_euler("zyx", [10 * i, 3 * np.sin(i), 2 * np.cos(i)], degrees=True)
        R = rot.as_matrix()
        pts_s, _ = scan(R, p, rng=rng)
        if frame == "sensor":
            pts = pts_s
        elif frame == "rover":
            pts = pts_s + OFFSET
        else:
            pts = (pts_s + OFFSET) @ R.T + p
        cat = rng.choice([-1, 0, 174], size=len(pts), p=[0.9, 0.07, 0.03])
        np.savetxt(out / "LiDAR" / f"{t_ns}.txt", np.column_stack([pts, cat]), fmt="%.6f %.6f %.6f %d")
        x, y, z, w = rot.as_quat()
        pose_rows.append([t_ns, *p, w, x, y, z] + [0.0] * 9)
    np.savetxt(out / "Rover_pose.txt", np.array(pose_rows), fmt="%.0f" + " %.9f" * 16)
    (out / "IMU.txt").write_text("")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake")
    ap.add_argument("--frame", default="rover", choices=["sensor", "rover", "world"])
    a = ap.parse_args()
    make_scene(Path(a.out) / f"Fake_{a.frame}", a.frame)
    print("wrote", Path(a.out) / f"Fake_{a.frame}")
