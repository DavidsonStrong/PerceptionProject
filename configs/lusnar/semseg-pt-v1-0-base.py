# =============================================================================
# Point Transformer (v1) on LuSNAR LiDAR scans
# Put this file at:  Pointcept/configs/lusnar/semseg-pt-v1-0-base.py
# Based on Pointcept's configs/s3dis/semseg-pt-v1-0-base.py
# =============================================================================

# Start from Pointcept's default settings, then override what we need below.
_base_ = ["../_base_/default_runtime.py"]

# ---------------------------------------------------------------------------
# General training settings
# ---------------------------------------------------------------------------
batch_size = 4          # scans per training step (total across all GPUs). Lower it if you run out of GPU memory.
num_worker = 8          # CPU processes that load data in the background
mix_prob = 0.0          # turn off "Mix3D" (mixing two scans together) to keep things simple
enable_amp = False      # False to ensure no overflow, True formixed precision: faster and uses less memory
enable_wandb = False    # don't log to Weights & Biases (avoids a login prompt)

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
model = dict(
    type="DefaultSegmentor",                # wrapper: backbone -> per-point class scores -> loss
    backbone=dict(
        type="PointTransformer-Seg50",      # Point Transformer v1 (Zhao et al., 2021)
        in_channels=3,                      # input features per point: just x, y, z
        num_classes=3,                      # regolith, rock, crater
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

# ---------------------------------------------------------------------------
# Optimiser and learning-rate schedule
# ---------------------------------------------------------------------------
epoch = 10              # total passes through the training set
eval_epoch = 2         # evaluate on the test scenes 10 times (i.e. every 3 epochs)
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)  # lr x0.1 at 60% and 80% of training

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dataset_type = "DefaultDataset"   # Pointcept's generic loader: reads coord.npy / segment.npy folders
data_root = "/home/jayanti/PerceptionProject/data/lusnar_preprocessed_pointcept"
grid_size = 0.04                  # voxel size in metres (keeps small rocks; LuSNAR rocks are 0.1 m+)

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["regolith", "rock", "crater"],   # order must match LABEL_MAP in prepare_lusnar.py

    # ----- training scans (scenes 1, 2, 4, 6, 8, 9) -----
    train=dict(
        type=dataset_type,
        split="train_every3.json", #"train",
        data_root=data_root,
        transform=[
            # Random changes so the network sees a slightly different scan every time
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),  # spin around vertical axis
            dict(type="RandomScale", scale=[0.9, 1.1]),                                  # shrink/grow by up to 10%
            dict(type="RandomFlip", p=0.5),                                              # mirror
            dict(type="RandomJitter", sigma=0.005, clip=0.02),                           # tiny noise on each point
            # Keep one point per 4 cm voxel
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            # Pack the data for the network; x,y,z are also used as the input features
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=["coord"]),
        ],
        test_mode=False,
    ),

    # ----- evaluation during training (scenes 3, 5, 7) -----
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),   # keep labels of ALL original points
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
                 return_grid_coord=True, return_inverse=True),            # "inverse" maps predictions back to every point
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                 feat_keys=["coord"]),
        ],
        test_mode=False,
    ),

    # ----- final test (scenes 3, 5, 7) -----
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            # In test mode, every point is covered by splitting the scan into several voxelised copies
            voxelize=dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=("coord",)),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1])],   # no test-time augmentation (just the original scan)
            ],
        ),
    ),
)
