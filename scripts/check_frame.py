#!/usr/bin/env python3
"""Figure out which coordinate frame the LuSNAR LiDAR/*.txt files are stored in.

For a sample of frames it tests three hypotheses (sensor, rover, world) by
checking how sharply the points' elevation angles cluster onto the 128 beam
angles. The right frame gives a comb of narrow spikes; wrong frames give mush.

    python scripts/check_frame.py --scene Moon_1
    python scripts/check_frame.py --scene Moon_1 --save-beam-table   # also measure real beam angles

Then put the reported frame into configs/lidar.yaml -> point_frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lunar_beams import BeamModel, PoseTable, elevation_deg, find_scenes, list_frames, load_cloud, to_sensor_frame  # noqa: E402
from lunar_beams.config import REPO_ROOT, load_config, pick_scenes, time_scale  # noqa: E402
from lunar_beams.io import frame_timestamp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "lidar.yaml"))
    ap.add_argument("--scene", nargs="*", help="substring(s) of scene path; default: first scene")
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--save-beam-table", action="store_true",
                    help="measure the actual beam angles and write configs/beam_table_measured.txt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    s = cfg["sensor"]
    model = BeamModel.nominal(s["n_beams"], s["upper_deg"], s["lower_deg"])
    scenes = pick_scenes(find_scenes(args.data_root), args.scene)
    scene = scenes[0]
    frames = list_frames(scene)
    idx = np.unique(np.linspace(0, len(frames) - 1, min(args.n_frames, len(frames))).astype(int))
    sample = [frames[i] for i in idx]
    print(f"scene: {scene}   ({len(frames)} frames, sampling {len(sample)})\n")

    # --- 1. coordinate magnitudes and centroid drift ---------------------------
    clouds = [load_cloud(f) for f in sample]
    xyz_all = np.vstack([c[:, :3] for c in clouds])
    cents = np.array([c[:, :3].mean(0) for c in clouds])
    r = np.linalg.norm(xyz_all, axis=1)
    print("coordinate check")
    print(f"  columns per point     : {clouds[0].shape[1]}  (expect 4: x y z category)")
    print(f"  points per frame      : {int(np.mean([len(c) for c in clouds]))}")
    print(f"  max |xyz| from origin : {r.max():.1f} m   (<= ~32 m suggests a local frame)")
    print(f"  centroid drift        : {np.linalg.norm(cents[-1] - cents[0]):.1f} m between first and last sampled frame")

    # --- 2. elevation sharpness under each hypothesis -------------------------
    poses = None
    try:
        poses = PoseTable.from_scene(scene)
        ts = np.array([frame_timestamp(f) for f in sample])
        scale = time_scale(ts, poses.t)
        if scale != 1.0:
            print(f"  note: scaling file-name timestamps by {scale:g} to match Rover_pose.txt")
    except Exception as e:  # noqa: BLE001
        print(f"  (no usable Rover_pose.txt: {e}; skipping world-frame test)")

    elev = {}
    for frame in ("sensor", "rover", "world"):
        if frame == "world" and poses is None:
            continue
        es = []
        for f, c in zip(sample, clouds):
            kw = {}
            if frame == "world":
                R, p, _ = poses.at(frame_timestamp(f) * scale)
                kw = {"R_world_rover": R, "p_world_rover": p}
            es.append(elevation_deg(to_sensor_frame(c[:, :3], frame, s["offset_in_rover"], **kw)))
        elev[frame] = np.concatenate(es)

    print("\nbeam-alignment test (median |residual| in beam spacings: ~0 = correct frame, ~0.25 = wrong)")
    scores = {}
    for frame, e in elev.items():
        sc = model.sharpness(e)
        scores[frame] = sc["median_abs_resid"]
        print(f"  {frame:6s}: median |resid| = {sc['median_abs_resid']:.3f}   "
              f"within 0.1 = {100 * sc['frac_within_0.1']:5.1f}%   inside FOV = {100 * sc['frac_inside_fov']:5.1f}%")

    best = min(scores, key=scores.get)
    print()
    if scores[best] < 0.08:
        print(f"==> Points are in the **{best}** frame. Set `point_frame: {best}` in configs/lidar.yaml")
    elif scores[best] < 0.18:
        print(f"==> Most likely **{best}** frame, but alignment is imperfect (median {scores[best]:.3f}).")
        print("    Beam angles may differ from the nominal spec: rerun with --save-beam-table and inspect the plot.")
    else:
        print("==> No hypothesis lines up. Inspect the histogram; the beam angles, mounting offset,")
        print("    or pose convention may differ from the paper. Ask your partner / me before downsampling.")

    # --- 3. plot ---------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(elev), 1, figsize=(11, 2.6 * len(elev)), sharex=True)
    axes = np.atleast_1d(axes)
    bins = np.arange(s["lower_deg"] - 2, s["upper_deg"] + 2, 0.02)
    for ax, (frame, e) in zip(axes, elev.items()):
        ax.hist(e, bins=bins, color="#3b6ea5")
        ax.set_title(f"{frame} frame hypothesis  (median |resid| {scores[frame]:.3f})", fontsize=10, loc="left")
        ax.set_ylabel("points")
    axes[-1].set_xlabel("elevation angle (deg, up positive)")
    fig.tight_layout()
    png = out / f"frame_check_{scene.name}.png"
    fig.savefig(png, dpi=120)
    print(f"\nhistogram saved to {png}")

    # --- 4. optional: measure real beam angles --------------------------------
    if args.save_beam_table:
        e = elev[best]
        ring, resid = model.assign(e)
        measured = model.angles_deg.copy()
        counts = np.bincount(ring, minlength=model.n_beams)
        for k in range(model.n_beams):
            m = (ring == k) & (np.abs(resid) < 0.3)
            if m.sum() >= 20:
                measured[k] = np.median(e[m])
        dev = measured - model.angles_deg
        print(f"\nrings with returns: {(counts > 0).sum()} / {model.n_beams}")
        print(f"measured - nominal angle: mean {dev.mean():+.4f} deg, max |dev| {np.abs(dev).max():.4f} deg")
        path = REPO_ROOT / "configs" / "beam_table_measured.txt"
        np.savetxt(path, measured, fmt="%.5f", header="measured beam elevation angles (deg), ring 0 = top")
        print(f"wrote {path}. To use it set sensor.beam_table: configs/beam_table_measured.txt")


if __name__ == "__main__":
    main()
