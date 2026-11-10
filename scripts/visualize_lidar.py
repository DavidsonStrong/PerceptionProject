"""Compare one LuSNAR LiDAR frame across the original and downsampled beam counts.

For each beam count (one row) the figure shows:
  - a top-down view coloured by beam, where each beam traces a ring on the ground;
  - an azimuth vs elevation view, where each beam is one horizontal line, so
    dropped beams show up as missing lines with even gaps between the rest.

With --3d it also saves a topographic 3D view: one perspective point cloud per
beam count, coloured by terrain height on a shared scale, so you can see how
much of the terrain shape survives at each beam count.

Usage:
    python scripts/visualize_lidar.py Moon_7/Moon_7
    python scripts/visualize_lidar.py Moon_7/Moon_7 --frames 0 800 1600
    python scripts/visualize_lidar.py Moon_7/Moon_7 --frames 800 --3d --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from downsample_lidar import FOV_MIN_DEG, N_BEAMS, beam_index


def load_scan(path):
    if path.stat().st_size == 0:
        return np.empty((0, 4))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def plot_frame(scene, name, beam_counts):
    fig, axes = plt.subplots(len(beam_counts), 2, figsize=(13, 3.2 * len(beam_counts)),
                             gridspec_kw={"width_ratios": [1, 2.6]}, squeeze=False)
    for row, n in enumerate(beam_counts):
        folder = "LiDAR" if n == N_BEAMS else f"LiDAR_{n}"
        pts = load_scan(scene / folder / name)
        ring, _ = beam_index(pts[:, :3])
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        azimuth = np.degrees(np.arctan2(y, x))
        elevation = np.degrees(np.arctan2(-z, np.hypot(x, y)))  # z points down in LuSNAR

        top, side = axes[row]
        top.scatter(x, y, c=ring, cmap="turbo", vmin=0, vmax=N_BEAMS // 2, s=0.3, linewidths=0)
        top.set_aspect("equal")
        top.set_xlim(-30, 30)
        top.set_ylim(-30, 30)
        top.set_ylabel(f"{n} beams\ny [m]")

        side.scatter(azimuth, elevation, c=ring, cmap="turbo", vmin=0, vmax=N_BEAMS // 2, s=0.3, linewidths=0)
        side.set_xlim(-180, 180)
        side.set_ylim(FOV_MIN_DEG - 1, 2)
        side.set_ylabel("elevation [deg]")
        side.set_title(f"{folder}: {len(pts)} points, {len(np.unique(ring))} beams with returns", fontsize=10, loc="left")
    axes[-1][0].set_xlabel("x [m]")
    axes[-1][1].set_xlabel("azimuth [deg]")
    fig.suptitle(f"{scene.name} frame {Path(name).stem}")
    fig.tight_layout()
    return fig


def plot_frame_3d(scene, name, beam_counts, radius, z_scale):
    scans = {}
    for n in beam_counts:
        pts = load_scan(scene / ("LiDAR" if n == N_BEAMS else f"LiDAR_{n}") / name)
        scans[n] = pts[np.hypot(pts[:, 0], pts[:, 1]) <= radius]
    # Height is -z because z points down in LuSNAR; share one colour scale across beam counts.
    heights = np.concatenate([-p[:, 2] for p in scans.values()])
    vmin, vmax = np.percentile(heights, [1, 99]) if len(heights) else (0, 1)
    pad = 0.1 * (vmax - vmin) + 0.05
    zlo, zhi = vmin - pad, vmax + pad

    cols = min(3, len(beam_counts))
    rows = -(-len(beam_counts) // cols)
    fig = plt.figure(figsize=(6.5 * cols, 4.6 * rows))
    for i, n in enumerate(beam_counts, 1):
        pts = scans[n]
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        sc = ax.scatter(pts[:, 0], pts[:, 1], -pts[:, 2], c=-pts[:, 2], cmap="terrain",
                        vmin=vmin, vmax=vmax, s=0.6, linewidths=0, depthshade=False)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(zlo, zhi)
        ax.set_box_aspect((1, 1, z_scale * (zhi - zlo) / (2 * radius)), zoom=1.25)
        ax.zaxis.set_major_locator(plt.MaxNLocator(3))
        ax.view_init(elev=25, azim=-60)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("height [m]")
        ax.set_title(f"{n} beams: {len(pts)} points", fontsize=10)
    fig.subplots_adjust(left=0, right=0.88, bottom=0, top=0.93, wspace=0, hspace=0)
    fig.colorbar(sc, cax=fig.add_axes((0.93, 0.25, 0.012, 0.5)), label="height [m] (sensor frame, up = -z)")
    fig.suptitle(f"{scene.name} frame {Path(name).stem}, within {radius:g} m, height x{z_scale:g}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path, help="scene folder containing LiDAR/ and LiDAR_<n>/ directories")
    parser.add_argument("--frames", type=int, nargs="+", help="frame indices in timestamp order (default: middle frame)")
    parser.add_argument("--beams", type=int, nargs="+", default=[128, 64, 32, 16, 8])
    parser.add_argument("--out", type=Path, help="folder for the PNGs (default: <scene>/lidar_viz)")
    parser.add_argument("--3d", dest="view_3d", action="store_true", help="also save a topographic 3D view")
    parser.add_argument("--radius", type=float, default=20.0, help="3D view: only plot points within this range [m]")
    parser.add_argument("--z-scale", type=float, default=10.0, help="3D view: vertical exaggeration")
    parser.add_argument("--show", action="store_true", help="also open each figure in a window")
    args = parser.parse_args()

    scans = sorted((args.scene / "LiDAR").glob("*.txt"))
    frames = args.frames if args.frames is not None else [len(scans) // 2]
    out = args.out or args.scene / "lidar_viz"
    out.mkdir(exist_ok=True)

    for i in frames:
        name = scans[i].name
        figs = {"": plot_frame(args.scene, name, args.beams)}
        if args.view_3d:
            figs["_3d"] = plot_frame_3d(args.scene, name, args.beams, args.radius, args.z_scale)
        for suffix, fig in figs.items():
            path = out / f"frame_{i:04d}_{Path(name).stem}{suffix}.png"
            fig.savefig(path, dpi=130)
            print(f"Saved {path}")
            if not args.show:
                plt.close(fig)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
