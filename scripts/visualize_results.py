"""
visualize_results.py
--------------------
Makes pictures of the Point Transformer results on LuSNAR.

Figure 1  (scan_<name>.png)
    One LiDAR scan seen from above, three panels side by side:
    Ground truth | Prediction | Errors (red = wrong)
Figure 2  (iou_vs_paper.png)
    Bar chart of our IoU per class next to the LuSNAR paper's numbers.

Usage (run from ~/PerceptionProject):
    python scripts/visualize_results.py                 # picks the test scan with the most crater points
    python scripts/visualize_results.py --scan scene7_1693385274849390592
    python scripts/visualize_results.py --open3d        # also opens an interactive 3D window
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save images without needing a window
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# SETTINGS - edit if your paths are different
# ---------------------------------------------------------------------------
DATA_ROOT = "/home/jayanti/PerceptionProject/data/lusnar_preprocessed_pointcept"
RESULT_DIR = "/home/jayanti/PerceptionProject/third_party/Pointcept/exp/lusnar/pt1_lusnar_fp32/result"
OUT_DIR = "/home/jayanti/PerceptionProject/results/pt1_128beam/figures"

CLASS_NAMES = ["Regolith", "Rock", "Crater"]
# One colour per class (index 0, 1, 2). Rock and crater use LuSNAR's own label colours.
CLASS_COLORS = np.array([
    [0.80, 0.78, 0.74],  # regolith: light grey
    [0.91, 0.98, 0.31],  # rock:     yellow  (#E8FA50)
    [0.70, 0.35, 1.00],  # crater:   bright purple (lighter version of LuSNAR's #7800C8, visible on black)
])

# Final test results (fill in from your log) and the paper's Table VI.
OURS = {"Regolith": 99.55, "Rock": 97.69, "Crater": 87.68, "mIoU": 94.97}
PAPER = {"Regolith": 99.61, "Rock": 98.16, "Crater": 88.67, "mIoU": 95.48}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_scan_with_most_craters():
    """Look through all test scans and return the one with the most crater points."""
    best_name, best_count = None, -1
    for folder in glob.glob(os.path.join(DATA_ROOT, "test", "*")):
        name = os.path.basename(folder)
        if not os.path.isfile(os.path.join(RESULT_DIR, f"{name}_pred.npy")):
            continue
        n_crater = (np.load(os.path.join(folder, "segment.npy")) == 2).sum()
        if n_crater > best_count:
            best_name, best_count = name, n_crater
    print(f"Picked {best_name} ({best_count} crater points)")
    return best_name


def load_scan(name):
    """Return positions, true labels and predicted labels for one test scan."""
    folder = os.path.join(DATA_ROOT, "test", name)
    xyz = np.load(os.path.join(folder, "coord.npy"))
    truth = np.load(os.path.join(folder, "segment.npy"))
    pred = np.load(os.path.join(RESULT_DIR, f"{name}_pred.npy"))
    return xyz, truth, pred


def plot_scan(xyz, truth, pred, name):
    """Three top-down panels: ground truth, prediction, errors."""
    valid = truth >= 0                       # skip the few 'ignore' points
    xyz, truth, pred = xyz[valid], truth[valid], pred[valid]
    wrong = truth != pred

    # Draw low points first so rocks (which stick up) end up on top.
    order = np.argsort(xyz[:, 2])
    x, y = xyz[order, 0], xyz[order, 1]
    truth, pred, wrong = truth[order], pred[order], wrong[order]

    panels = [
        ("Ground truth", CLASS_COLORS[truth]),
        ("Point Transformer prediction", CLASS_COLORS[pred]),
        (f"Errors ({100 * wrong.mean():.2f}% of points wrong)", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), facecolor="black")
    for ax, (title, colors) in zip(axes, panels):
        if colors is not None:
            ax.scatter(x, y, c=colors, s=0.8, linewidths=0)
        else:
            # Error panel: correct points as dim grey, then wrong points larger and in red ON TOP
            ax.scatter(x[~wrong], y[~wrong], c=[[0.30, 0.30, 0.30]], s=0.6, linewidths=0)
            ax.scatter(x[wrong], y[wrong], c=[[1.0, 0.15, 0.15]], s=4, linewidths=0)
        ax.set_title(title, color="white", fontsize=14)
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.axis("off")
        ax.plot(0, 0, marker="^", color="cyan", markersize=8)   # where the rover is

    # Legend for the class colours
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=9, label=n)
               for n, c in zip(CLASS_NAMES, CLASS_COLORS)]
    handles.append(plt.Line2D([], [], marker="^", ls="", color="cyan", markersize=9, label="Rover"))
    handles.append(plt.Line2D([], [], marker="o", ls="", color=(1.0, 0.15, 0.15), markersize=9, label="Wrong prediction"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               labelcolor="white", fontsize=12)
    fig.suptitle(f"LuSNAR test scan {name}  (top-down view, 30 m range)", color="white", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    path = os.path.join(OUT_DIR, f"scan_{name}.png")
    fig.savefig(path, dpi=200, facecolor="black")
    plt.close(fig)
    print("Saved", path)


def plot_iou_bars():
    """Grouped bar chart: our IoU vs the paper's, per class and mIoU."""
    labels = list(OURS.keys())
    pos = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(pos - width / 2, [PAPER[k] for k in labels], width, label="LuSNAR paper", color="#9aa5b1")
    b2 = ax.bar(pos + width / 2, [OURS[k] for k in labels], width, label="Ours (reproduction)", color="#3b6fb6")
    ax.bar_label(b1, fmt="%.1f", fontsize=9)
    ax.bar_label(b2, fmt="%.1f", fontsize=9)
    ax.set_xticks(pos, labels)
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(80, 101)            # zoomed in so the small differences are visible
    ax.set_title("Point Transformer on LuSNAR (128 beams): reproduction vs paper")
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    path = os.path.join(OUT_DIR, "iou_vs_paper.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved", path)


def show_open3d(xyz, pred):
    """Interactive 3D window coloured by prediction (drag to rotate, scroll to zoom)."""
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(CLASS_COLORS[pred])
    o3d.visualization.draw_geometries([cloud], window_name="Point Transformer prediction")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", default=None, help="test scan folder name, e.g. scene7_1693385274849390592")
    parser.add_argument("--open3d", action="store_true", help="also open an interactive 3D viewer")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    name = args.scan or pick_scan_with_most_craters()

    xyz, truth, pred = load_scan(name)
    plot_scan(xyz, truth, pred, name)
    plot_iou_bars()

    if args.open3d:
        show_open3d(xyz, pred)


if __name__ == "__main__":
    main()
