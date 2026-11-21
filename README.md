# Lunar lidar beam reduction

Effect of fewer lidar beams on odometry and 3D semantic segmentation in
feature-poor lunar terrain, using the [LuSNAR](https://github.com/zqyu9/LuSNAR-dataset) dataset.

LuSNAR does not label which beam each point came from. This repo recovers the
beam (ring) index from geometry: every beam of a spinning lidar fires at a fixed
elevation angle, so each point's elevation angle, measured from the lidar's
origin, tells you its beam. The scripts then write beam-downsampled copies of the
dataset (64, 32, 16, 8 beams, and real-sensor-like angle sets).

```
configs/lidar.yaml        sensor model, point frame, downsampling configs  <- edit this
lunar_beams/              library: file IO, pose interpolation, frames, ring assignment
scripts/download_lusnar.py   fetch lidar + poses only (skips ~92 GB of images)
scripts/check_frame.py       work out which coordinate frame the points are in
scripts/downsample.py        assign rings, write downsampled datasets + stats
scripts/downsample_lidar.py  standalone: write <scene>/LiDAR_<n>/ copies (every k-th beam from the bottom)
scripts/visualize_lidar.py   plot one frame across beam counts (2D, and 3D with --3d) from LiDAR_<n>/
tests/                    unit tests + synthetic LuSNAR-style scene generator
data/                     (git-ignored) raw/ and processed/ data
outputs/                  (git-ignored) plots, stats CSVs
```

---

## Step-by-step setup

### 0. What you need

- Python 3.10+ and git
- A GitHub account each
- Disk space: raw lidar is ~14 GB. Each downsampled config adds roughly
  14 GB × (beams kept / 128), and all the default configs together add ~32 GB.
  Only generate the configs you need (step 7).

### 1. Create the GitHub repo (one person: "owner")

1. On GitHub: **New repository** → name it (e.g. `lunar-lidar-beams`) → Private →
   don't add a README/gitignore (this folder already has them) → Create.
2. In this folder:
   ```bash
   git init -b main
   git add .
   git commit -m "Starter: ring recovery and beam downsampling"
   git remote add origin https://github.com/<owner>/lunar-lidar-beams.git
   git push -u origin main
   ```
3. **Settings → Collaborators → Add people** → add your partner.
4. Optional but recommended: **Settings → Branches → Add rule** for `main`,
   tick "Require a pull request before merging". The included GitHub Action
   runs the tests on every push and PR.

### 2. Clone it (partner)

```bash
git clone https://github.com/<owner>/lunar-lidar-beams.git
cd lunar-lidar-beams
```

### 3. Python environment (both of you)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # should print "7 passed"
```

### 4. Download the data (both of you, or one person onto a shared drive)

Data never goes into git. Everything lives under `data/` (git-ignored). If the
data sits elsewhere, e.g. an external drive, either pass `--data-root` to every
script or symlink it: `ln -s /path/to/LuSNAR data/raw`.

**Option A, Hugging Face (lets you skip the images):**

```bash
python scripts/download_lusnar.py --list     # look at how the mirror is organised first
python scripts/download_lusnar.py            # default: *LiDAR*, *Rover_pose*, *IMU*
```

If `--list` shows the data packed as archives (for example one `.zip` per scene),
pass matching patterns, e.g. `--include "*.zip"`; archives are unpacked automatically.

**Option B, official CSTCloud link:** <https://pan.cstcloud.cn/s/2Ie7D5PSLU>
(password `fjZt`). Download only each scene's `LiDAR/` folder plus
`Rover_pose.txt` and `IMU.txt`, and unpack into `data/raw/`.

Expected layout (scene folder names may differ):

```
data/raw/<scene>/LiDAR/<timestamp_ns>.txt   x y z category_id   (-1 regolith, 0 crater, 174 rock)
data/raw/<scene>/Rover_pose.txt             t_ns px py pz qw qx qy qz ...
data/raw/<scene>/IMU.txt
```

The scripts find any folder containing `LiDAR/`, so nesting doesn't matter.

**No data yet?** Generate a fake scene to try the pipeline:
`python -m tests.synthetic --out data/raw --frame rover`

### 5. Find out which frame the points are in (do this once, together)

```bash
python scripts/check_frame.py --scene <part of a scene name>
```

It tests three hypotheses (sensor, rover body, world) and reports which one makes
the elevation angles snap onto the 128 beams. It also saves a histogram to
`outputs/frame_check_<scene>.png`: the correct frame looks like a comb of sharp
spikes about 0.41° apart. Run it on two or three different scenes to be sure.

Then edit `configs/lidar.yaml`:

```yaml
point_frame: rover     # whatever it reported
```

Commit that change via a pull request so you both use the same setting.

If the result is "most likely ... but imperfect", run it again with
`--save-beam-table`. This measures the actual beam angles and writes
`configs/beam_table_measured.txt`. Set `sensor.beam_table` to that file and commit it.

### 6. Quick test of the downsampling

```bash
python scripts/downsample.py --scene <scene> --max-frames 50 --configs b32_uniform vlp16_like
```

Check the last lines: the median residual should be well under 0.1 beam spacings.
Open a few output files in CloudCompare (or Open3D) next to the originals to see the rings.

### 7. Full run

```bash
python scripts/downsample.py --with-ring --workers 8
# or only what you need:
python scripts/downsample.py --configs b64_uniform b32_uniform b16_uniform --with-ring
```

Output:

```
data/processed/<config>/<scene>/LiDAR/<timestamp>.txt   x y z category [ring]
data/processed/<config>/<scene>/Rover_pose.txt, IMU.txt (copied unchanged)
outputs/downsample_stats.csv                             points kept per frame/config
```

Each `data/processed/<config>/` folder is a drop-in replacement for the raw
dataset, so your odometry and segmentation code just takes a different root path.
Points stay in their original frame and only rows are removed, so semantic labels
are preserved. With `--with-ring`, column 5 is the ring index re-numbered from
`0` (bottom beam) to `M-1`, the convention LeGO-LOAM and FAST-LIO expect.

### 8. Adding or changing beam configurations

Everything lives in `configs/lidar.yaml` under `configs:`.

- `type: every_k` keeps every k-th beam. This keeps the 52° vertical FOV with coarser spacing.
- `type: angles` keeps the beam nearest each target angle (`angles_deg: [...]` or
  `linspace: [lo, hi, n]`). Use this to mimic a real sensor's FOV.

Target angles must be inside LuSNAR's -25° to +27° range.

---

## Working together in git

- Never commit to `main` directly. Use one branch per task:
  ```bash
  git checkout main && git pull
  git checkout -b odometry-kiss-icp
  # ...work...
  git add -A && git commit -m "Add KISS-ICP runner for downsampled configs"
  git push -u origin odometry-kiss-icp
  ```
  Then open a pull request on GitHub. The other person skims it and merges.
- Pull `main` often (`git pull`), especially before starting a new branch.
- Commit code, configs and small result tables (CSV, final plots). Never commit
  `data/` or large outputs. `.gitignore` already blocks them.
- Keep experiment settings in `configs/`, not hard-coded, so results are reproducible.
- A natural split: one person owns odometry (`odometry/`), the other
  segmentation (`segmentation/`). Shared code goes in `lunar_beams/` and is
  changed via PR only.

## Next steps (suggested)

- **Odometry:** KISS-ICP needs no ring info and is the easiest to run on many
  configs. A-LOAM needs `N_SCANS` set for each config. LeGO-LOAM needs the ring
  field plus `N_SCAN`, `ang_res_y` and `ang_bottom`. FAST-LIO takes the ring
  field. All of them need the `.txt` frames converted to their input format
  (ROS bag or PCD). Evaluate with `evo` against `Rover_pose.txt`, using the
  RPE/ATE metrics from the LuSNAR paper.
- **Segmentation:** train or evaluate RandLA-Net or Point Transformer on each
  config's `data/processed/<config>/`. The paper's split is scenes 1, 2, 4, 6, 8, 9
  for training and 3, 5, 7 for testing.
- **Fixed point budget:** for comparison, also hold points per frame constant
  while varying beams (subsample azimuth on the high-beam configs).

## Troubleshooting

- *"Set point_frame in configs/lidar.yaml first"*: run step 5.
- *No hypothesis lines up in check_frame*: the beam angles, mounting offset or
  pose convention differ from the paper. Look at the histogram. A comb of spikes
  with a constant shift points to an offset issue; uniform mush points to the
  world frame with a pose problem.
- *Far fewer rings than 128 in the output*: expected. Upward beams hit sky and
  return nothing within the 30 m range.
