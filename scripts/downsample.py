#!/usr/bin/env python3
"""Assign every LuSNAR point to a beam and write beam-downsampled copies of the dataset.

    python scripts/downsample.py --scene Moon_1 --max-frames 50    # quick test
    python scripts/downsample.py                                    # everything, every config
    python scripts/downsample.py --configs b32_uniform vlp16_like --with-ring

Output layout (each config is a drop-in replacement for the raw dataset):
    data/processed/<config>/<scene>/LiDAR/<timestamp>.txt   x y z category [ring]
    data/processed/<config>/<scene>/Rover_pose.txt, IMU.txt (copied)
    outputs/downsample_stats.csv                             per-frame point counts

Points stay in their ORIGINAL frame; only rows are removed. With --with-ring a
5th column holds the ring index re-numbered 0..M-1 with 0 = bottom beam
(the convention LeGO-LOAM / FAST-LIO expect).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lunar_beams import BeamModel, PoseTable, elevation_deg, find_scenes, list_frames, load_cloud, save_cloud, select_rings, to_sensor_frame  # noqa: E402
from lunar_beams.config import REPO_ROOT, load_config, pick_scenes, time_scale  # noqa: E402
from lunar_beams.io import frame_timestamp  # noqa: E402
from lunar_beams.rings import loam_ring_index  # noqa: E402


def process_frame(job: dict) -> list[dict]:
    data = load_cloud(job["path"])
    xyz_s = to_sensor_frame(data[:, :3], job["frame"], job["offset"], job.get("R"), job.get("p"))
    model = BeamModel(job["angles"])
    ring, resid = model.assign(elevation_deg(xyz_s))
    ambiguous = np.abs(resid) > job["max_resid"]
    rows = []
    for name, kept in job["selections"].items():
        mask = np.isin(ring, kept) & ~ambiguous
        out = Path(job["out_root"]) / name / job["scene_rel"] / "LiDAR" / Path(job["path"]).name
        r_out = loam_ring_index(kept, ring[mask]) if job["with_ring"] else None
        save_cloud(out, data[mask], r_out)
        rows.append({"scene": job["scene_rel"], "frame": Path(job["path"]).stem, "config": name,
                     "n_beams": len(kept), "n_in": len(data), "n_out": int(mask.sum()),
                     "n_ambiguous": int(ambiguous.sum()),
                     "median_abs_resid": float(np.median(np.abs(resid))) if len(resid) else np.nan})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "data" / "processed"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "lidar.yaml"))
    ap.add_argument("--configs", nargs="*", help="names from configs/lidar.yaml (default: all)")
    ap.add_argument("--scene", nargs="*", help="substring(s) of scene paths (default: all)")
    ap.add_argument("--max-frames", type=int, help="limit frames per scene (for testing)")
    ap.add_argument("--with-ring", action="store_true", help="append LOAM-style ring column")
    ap.add_argument("--max-resid", type=float, default=0.35,
                    help="drop points further than this (in beam spacings) from any beam")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    frame = cfg.get("point_frame", "UNKNOWN")
    if frame not in ("sensor", "rover", "world"):
        raise SystemExit("Set point_frame in configs/lidar.yaml first (run scripts/check_frame.py).")
    s = cfg["sensor"]
    model = BeamModel.from_config(s, REPO_ROOT)
    names = args.configs or list(cfg["configs"])
    selections = {n: select_rings(model, cfg["configs"][n]) for n in names}
    for n, kept in selections.items():
        a = model.angles_deg[kept]
        print(f"{n:14s}: {len(kept):3d} beams, {a.min():+6.2f} .. {a.max():+6.2f} deg")

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    stats = []
    for scene in pick_scenes(find_scenes(data_root), args.scene):
        rel = scene.relative_to(data_root) if scene != data_root else Path(".")
        frames = list_frames(scene)[: args.max_frames]
        poses, scale = None, 1.0
        if frame == "world":
            poses = PoseTable.from_scene(scene)
            scale = time_scale(np.array([frame_timestamp(f) for f in frames]), poses.t)

        jobs = []
        for f in frames:
            job = {"path": str(f), "frame": frame, "offset": s["offset_in_rover"], "angles": model.angles_deg,
                   "selections": selections, "out_root": str(out_root), "scene_rel": str(rel),
                   "with_ring": args.with_ring, "max_resid": args.max_resid}
            if poses is not None:
                job["R"], job["p"], _ = poses.at(frame_timestamp(f) * scale)
            jobs.append(job)

        print(f"\n{rel}: {len(jobs)} frames")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, rows in enumerate(ex.map(process_frame, jobs, chunksize=8)):
                stats.extend(rows)
                if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                    print(f"  {i + 1}/{len(jobs)}", flush=True)

        for name in names:
            for extra in ("Rover_pose.txt", "IMU.txt"):
                if (scene / extra).exists():
                    dst = out_root / name / rel / extra
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(scene / extra, dst)

    stats_path = REPO_ROOT / "outputs" / "downsample_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)

    resid = np.array([r["median_abs_resid"] for r in stats])
    amb = sum(r["n_ambiguous"] for r in stats) / max(1, sum(r["n_in"] for r in stats))
    print(f"\nmedian |resid| across frames: {np.nanmedian(resid):.3f} beam spacings; "
          f"ambiguous points dropped: {100 * amb:.2f}%")
    if np.nanmedian(resid) > 0.1:
        print("WARNING: points do not line up well with the beam model. Re-check point_frame.")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
