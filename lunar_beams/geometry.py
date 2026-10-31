"""Frame conversions. All frames are AirSim/NED style: x forward, y right, z DOWN."""
from __future__ import annotations

import numpy as np

FRAMES = ("sensor", "rover", "world")


def to_sensor_frame(xyz: np.ndarray, frame: str, offset_in_rover, R_world_rover=None,
                    p_world_rover=None) -> np.ndarray:
    """Express points in the lidar's own frame so elevation angles are meaningful.

    The lidar is mounted with the same orientation as the rover, translated by
    `offset_in_rover`.
    """
    offset = np.asarray(offset_in_rover, dtype=float)
    if frame == "sensor":
        return xyz
    if frame == "rover":
        return xyz - offset
    if frame == "world":
        if R_world_rover is None or p_world_rover is None:
            raise ValueError("world frame needs the rover pose for this frame")
        body = (xyz - p_world_rover) @ R_world_rover  # == R^T (x - p), row-vector form
        return body - offset
    raise ValueError(f"unknown frame {frame!r}; expected one of {FRAMES}")


def elevation_deg(xyz_sensor: np.ndarray) -> np.ndarray:
    """Elevation above the horizontal plane, positive UP (z is down, hence the minus)."""
    x, y, z = xyz_sensor[:, 0], xyz_sensor[:, 1], xyz_sensor[:, 2]
    return np.degrees(np.arctan2(-z, np.hypot(x, y)))
