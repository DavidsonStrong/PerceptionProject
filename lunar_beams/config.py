"""Config loading and small helpers shared by the scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = REPO_ROOT / "configs" / "lidar.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_scenes(all_scenes: list[Path], wanted: list[str] | None) -> list[Path]:
    """Filter scenes by substring match on their path (e.g. 'Moon_1' or '1')."""
    if not wanted:
        return all_scenes
    out = [s for s in all_scenes if any(w in str(s) for w in wanted)]
    if not out:
        raise SystemExit(f"No scene matches {wanted}. Available: {[str(s) for s in all_scenes]}")
    return out


def time_scale(frame_ts: np.ndarray, pose_t: np.ndarray) -> float:
    """Factor to multiply LiDAR file-name timestamps by so they line up with pose timestamps.

    Usually 1 (both in ns), but this guards against e.g. file names in seconds.
    """
    best, best_frac = 1.0, -1.0
    for e in range(-9, 10, 3):
        s = 10.0 ** e
        ts = frame_ts * s
        frac = np.mean((ts >= pose_t[0]) & (ts <= pose_t[-1]))
        if frac > best_frac + 1e-9:
            best, best_frac = s, frac
    return best
