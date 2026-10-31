#!/usr/bin/env python3
"""Download the LuSNAR lidar + poses (not the ~92 GB of images) from Hugging Face.

    python scripts/download_lusnar.py --list                 # see how the HF repo is laid out
    python scripts/download_lusnar.py                        # lidar, poses, IMU for all scenes
    python scripts/download_lusnar.py --include "*Moon_1*LiDAR*"   # custom pattern(s)

If the Hugging Face mirror is slow or unavailable, use the official CSTCloud
link instead (see README) and unpack into data/raw/.
"""
from __future__ import annotations

import argparse
import fnmatch
import tarfile
import zipfile
from pathlib import Path

REPO_ID = "JeremyLuo/LuSNAR"
DEFAULT_PATTERNS = ["*LiDAR*", "*Rover_pose*", "*IMU*", "*README*"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", default="data/raw")
    ap.add_argument("--include", nargs="+", default=DEFAULT_PATTERNS,
                    help="glob patterns of repo files to fetch")
    ap.add_argument("--list", action="store_true", help="only list files in the HF repo and exit")
    ap.add_argument("--no-extract", action="store_true", help="don't unpack .zip/.tar archives")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    if args.list:
        for f in files:
            print(f)
        print(f"\n{len(files)} files")
        return

    matched = [f for f in files if any(fnmatch.fnmatch(f, p) for p in args.include)]
    print(f"{len(matched)} of {len(files)} repo files match {args.include}")
    if not matched:
        print("Nothing matched. Run with --list and adjust --include. "
              "(The repo may store data as archives, e.g. --include '*.zip'.)")
        return

    snapshot_download(REPO_ID, repo_type="dataset", local_dir=args.local_dir,
                      allow_patterns=args.include)

    if not args.no_extract:
        for arc in list(Path(args.local_dir).rglob("*")):
            if arc.suffix == ".zip":
                print("extracting", arc)
                with zipfile.ZipFile(arc) as z:
                    z.extractall(arc.parent)
            elif arc.name.endswith((".tar", ".tar.gz", ".tgz")):
                print("extracting", arc)
                with tarfile.open(arc) as t:
                    t.extractall(arc.parent, filter="data")
    print("done ->", args.local_dir)


if __name__ == "__main__":
    main()
