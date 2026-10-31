"""Reading and writing LuSNAR files.

Layout (from the official README):
    <scene>/LiDAR/<timestamp>.txt   columns: x y z category_id
    <scene>/Rover_pose.txt          columns: t[ns] px py pz qw qx qy qz vx vy vz bwx bwy bwz bax bay baz
    <scene>/IMU.txt
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def find_scenes(root: str | Path) -> list[Path]:
    """Every folder under `root` that has a LiDAR/ subfolder, sorted by path."""
    root = Path(root)
    scenes = sorted({p.parent for p in root.rglob("LiDAR") if p.is_dir()})
    if not scenes:
        raise FileNotFoundError(f"No LiDAR/ folders found under {root}. Check the download path.")
    return scenes


def list_frames(scene: str | Path) -> list[Path]:
    """LiDAR frames of a scene, sorted by the numeric timestamp in the file name."""
    files = list((Path(scene) / "LiDAR").glob("*.txt"))

    def key(p: Path):
        try:
            return (0, float(p.stem))
        except ValueError:
            return (1, p.stem)

    return sorted(files, key=key)


def frame_timestamp(path: str | Path) -> float:
    return float(Path(path).stem)


def _numeric_rows(path: Path) -> np.ndarray:
    """Load a whitespace/comma separated numeric file, skipping header or comment lines."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue  # header or comment
    if not rows:
        raise ValueError(f"No numeric rows in {path}")
    return np.asarray(rows, dtype=np.float64)


def load_cloud(path: str | Path) -> np.ndarray:
    """(N, C) array; columns 0-2 are xyz, column 3 the semantic category."""
    data = _numeric_rows(Path(path))
    if data.ndim == 1:
        data = data[None, :]
    return data


def save_cloud(path: str | Path, data: np.ndarray, ring: np.ndarray | None = None) -> None:
    """Write x y z category [ring] with the same layout as the originals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [data[:, 0], data[:, 1], data[:, 2]]
    fmt = ["%.6f", "%.6f", "%.6f"]
    if data.shape[1] > 3:
        cols.append(data[:, 3])
        fmt.append("%d")
    if ring is not None:
        cols.append(ring)
        fmt.append("%d")
    np.savetxt(path, np.column_stack(cols), fmt=fmt)


class PoseTable:
    """Ground-truth rover poses with interpolation at arbitrary timestamps.

    Position is linearly interpolated, orientation is slerped. Quaternions in
    Rover_pose.txt are (qw, qx, qy, qz); scipy wants (x, y, z, w).
    """

    def __init__(self, t: np.ndarray, p: np.ndarray, q_wxyz: np.ndarray):
        order = np.argsort(t)
        t, p, q_wxyz = t[order], p[order], q_wxyz[order]
        t, keep = np.unique(t, return_index=True)
        self.t, self.p = t, p[keep]
        q_xyzw = q_wxyz[keep][:, [1, 2, 3, 0]]
        self.rot = Rotation.from_quat(q_xyzw)
        self._slerp = Slerp(self.t, self.rot)

    @classmethod
    def from_scene(cls, scene: str | Path) -> "PoseTable":
        path = Path(scene) / "Rover_pose.txt"
        d = _numeric_rows(path)
        if d.shape[1] < 8:
            raise ValueError(f"{path} has {d.shape[1]} columns, expected >= 8")
        return cls(d[:, 0], d[:, 1:4], d[:, 4:8])

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """(R_world_from_rover 3x3, p_world 3, gap_to_nearest_pose_in_same_units_as_t)."""
        gap = float(np.min(np.abs(self.t - t)))
        tc = float(np.clip(t, self.t[0], self.t[-1]))
        if tc != t:
            warnings.warn(f"timestamp {t} outside pose range; clamped")
        R = self._slerp([tc]).as_matrix()[0]
        p = np.array([np.interp(tc, self.t, self.p[:, i]) for i in range(3)])
        return R, p, gap
