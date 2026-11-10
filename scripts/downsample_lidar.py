"""Downsample LuSNAR 128-beam LiDAR scans to fewer beams for an ablation study.

LuSNAR's simulated LiDAR has 128 beams evenly spaced over a -25 to +27 deg
vertical FOV (spacing 52/127 deg). The scans carry no ring index, so each
point's beam is recovered from its elevation angle. In the LiDAR frame z
points down (ground returns have z > 0), so elevation = atan2(-z, hypot(x, y)).

An N-beam scan keeps every (128 / N)-th beam, starting from the lowest beam,
so it covers the same vertical FOV as the original sensor at coarser spacing.

The original LiDAR folder is only read. Each beam count is written to its own
sibling folder (LiDAR_64, LiDAR_32, ...) with the same filenames and line
format, so existing loaders work by just changing the folder name.

Usage:
    python scripts/downsample_lidar.py Moon_7/Moon_7
    python scripts/downsample_lidar.py Moon_7/Moon_7 --beams 64 32 16 8 --overwrite
"""

import argparse
import sys
from pathlib import Path

import numpy as np

N_BEAMS = 128
FOV_MIN_DEG = -25.0
FOV_MAX_DEG = 27.0
BEAM_SPACING_DEG = (FOV_MAX_DEG - FOV_MIN_DEG) / (N_BEAMS - 1)
# Points further than this from any beam angle are reported (half spacing is ~0.2 deg).
MAX_RESIDUAL_DEG = 0.05


def beam_index(xyz):
    """Return the 0-based beam index (0 = lowest beam) and angular residual per point."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elevation = np.degrees(np.arctan2(-z, np.hypot(x, y)))
    k = (elevation - FOV_MIN_DEG) / BEAM_SPACING_DEG
    ring = np.clip(np.round(k).astype(int), 0, N_BEAMS - 1)
    residual = np.abs(k - ring) * BEAM_SPACING_DEG
    return ring, residual


def downsample_scan(lines, beam_counts):
    """Split one scan's text lines into per-beam-count subsets, keeping lines verbatim."""
    xyz = np.array([line.split(",")[:3] for line in lines], dtype=np.float64).reshape(-1, 3)
    ring, residual = beam_index(xyz)
    out = {}
    for n in beam_counts:
        keep = ring % (N_BEAMS // n) == 0
        out[n] = [line for line, k in zip(lines, keep) if k]
    return out, int((residual > MAX_RESIDUAL_DEG).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path, help="scene folder containing the LiDAR/ directory")
    parser.add_argument("--beams", type=int, nargs="+", default=[64, 32, 16, 8])
    parser.add_argument("--overwrite", action="store_true", help="rewrite scans that already exist in the output folders")
    args = parser.parse_args()

    src = args.scene / "LiDAR"
    if not src.is_dir():
        sys.exit(f"No LiDAR folder found at {src}")
    for n in args.beams:
        if n <= 0 or n >= N_BEAMS or N_BEAMS % n:
            sys.exit(f"--beams values must divide {N_BEAMS} and be smaller than it, got {n}")

    out_dirs = {n: args.scene / f"LiDAR_{n}" for n in args.beams}
    for d in out_dirs.values():
        d.mkdir(exist_ok=True)

    scans = sorted(src.glob("*.txt"))
    n_off_grid = 0
    for i, scan in enumerate(scans, 1):
        if not args.overwrite and all((d / scan.name).exists() for d in out_dirs.values()):
            continue
        lines = [line for line in scan.read_text().splitlines() if line.strip()]
        subsets, off_grid = downsample_scan(lines, args.beams)
        n_off_grid += off_grid
        for n, kept in subsets.items():
            (out_dirs[n] / scan.name).write_text("\n".join(kept) + "\n" if kept else "")
        if i % 100 == 0 or i == len(scans):
            print(f"{i}/{len(scans)} scans")

    if n_off_grid:
        print(f"Warning: {n_off_grid} points were more than {MAX_RESIDUAL_DEG} deg from any beam angle")
    print("Wrote " + ", ".join(str(d) for d in out_dirs.values()))


if __name__ == "__main__":
    main()
