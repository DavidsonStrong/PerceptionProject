"""
prepare_lusnar.py
-----------------
Converts LuSNAR LiDAR scans (.txt files) into the folder format that
Pointcept's DefaultDataset can read.

Input  (one .txt per scan, one point per line):   x, y, z, label
Output (one folder per scan):
    data/lusnar/train/scene1_<timestamp>/coord.npy    (N, 3) float32  -> point positions
    data/lusnar/train/scene1_<timestamp>/segment.npy  (N,)   int32    -> class of each point

Usage:
    python prepare_lusnar.py
"""

import os
import glob
import numpy as np

# ---------------------------------------------------------------------------
# 1. SETTINGS - edit these paths to match where you downloaded LuSNAR
# ---------------------------------------------------------------------------

# Folder that contains the LiDAR .txt files for each scene.
# Change these to your real paths (each one should be full of <timestamp>.txt files).
SCENE_LIDAR_FOLDERS = {
    1: "/home/jayanti/PerceptionProject/data/raw/Moon_1/LiDAR",
    2: "/home/jayanti/PerceptionProject/data/raw/Moon_2/LiDAR",
    3: "/home/jayanti/PerceptionProject/data/raw/Moon_3/LiDAR",
    4: "/home/jayanti/PerceptionProject/data/raw/Moon_4/LiDAR",
    5: "/home/jayanti/PerceptionProject/data/raw/Moon_5/LiDAR",
    6: "/home/jayanti/PerceptionProject/data/raw/Moon_6/LiDAR",
    7: "/home/jayanti/PerceptionProject/data/raw/Moon_7/LiDAR",
    8: "/home/jayanti/PerceptionProject/data/raw/Moon_8/LiDAR",
    9: "/home/jayanti/PerceptionProject/data/raw/Moon_9/LiDAR",
}

# Same train/test split as the LuSNAR paper (Table VI).
TRAIN_SCENES = [1, 2, 4, 6, 8, 9]
TEST_SCENES = [3, 5, 7]

# Where to write the converted data (inside your Pointcept folder).
OUTPUT_ROOT = "/home/jayanti/PerceptionProject/data/lusnar_preprocessed_pointcept"

# LuSNAR label IDs  ->  the class numbers we will train with (0, 1, 2).
# (LuSNAR uses -1 = regolith, 0 = crater, 174 = rock.)
LABEL_MAP = {
    -1: 0,   # lunar regolith
    174: 1,  # rock
    0: 2,    # impact crater
}
IGNORE = -1  # Pointcept skips points with label -1 when computing the loss


# ---------------------------------------------------------------------------
# 2. HELPER: convert one scan
# ---------------------------------------------------------------------------

def convert_one_scan(txt_path, out_folder):
    # Read the text file. Each row is: x, y, z, label
    data = np.loadtxt(txt_path, delimiter=",")

    xyz = data[:, 0:3].astype(np.float32)        # first three columns
    raw_labels = data[:, 3].astype(np.int64)     # last column

    # LuSNAR comes from AirSim, which uses "z points DOWN".
    # Flip z so that "up" is positive (nicer for viewing; the network doesn't care).
    xyz[:, 2] = -xyz[:, 2]

    # Turn LuSNAR label IDs into 0 / 1 / 2.
    # Anything we don't recognise becomes IGNORE so it can't break training.
    labels = np.full(len(raw_labels), IGNORE, dtype=np.int32)
    for lusnar_id, our_id in LABEL_MAP.items():
        labels[raw_labels == lusnar_id] = our_id

    # Save in Pointcept's format.
    os.makedirs(out_folder, exist_ok=True)
    np.save(os.path.join(out_folder, "coord.npy"), xyz)
    np.save(os.path.join(out_folder, "segment.npy"), labels)

    # Return the raw IDs so we can spot unexpected labels.
    return np.unique(raw_labels)


# ---------------------------------------------------------------------------
# 3. MAIN: loop over every scene and every scan
# ---------------------------------------------------------------------------

def main():
    all_ids_seen = set()

    for scene_id, lidar_folder in SCENE_LIDAR_FOLDERS.items():
        # Decide whether this scene goes to "train" or "test".
        if scene_id in TRAIN_SCENES:
            split = "train"
        elif scene_id in TEST_SCENES:
            split = "test"
        else:
            continue

        txt_files = sorted(glob.glob(os.path.join(lidar_folder, "*.txt")))
        print(f"Scene {scene_id} ({split}): {len(txt_files)} scans")

        for txt_path in txt_files:
            timestamp = os.path.splitext(os.path.basename(txt_path))[0]
            out_folder = os.path.join(OUTPUT_ROOT, split, f"scene{scene_id}_{timestamp}")
            ids = convert_one_scan(txt_path, out_folder)
            all_ids_seen.update(ids.tolist())

    # Sanity check: warn about label IDs that are not in LABEL_MAP.
    unknown = all_ids_seen - set(LABEL_MAP.keys())
    print("Label IDs found in the data:", sorted(all_ids_seen))
    if unknown:
        print("WARNING: these IDs were not in LABEL_MAP and were set to ignore:", sorted(unknown))
    print("Done.")


if __name__ == "__main__":
    main()
