"""Beam (ring) model: which elevation angle each laser fires at, and point -> ring assignment.

Convention: ring 0 is the TOP beam (as AirSim numbers its channels). When we
write rings out for LOAM-style tools we re-index so 0 is the BOTTOM beam,
which is what LeGO-LOAM / FAST-LIO expect.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BeamModel:
    angles_deg: np.ndarray  # shape (n_beams,), sorted descending (ring 0 = top)

    @classmethod
    def nominal(cls, n_beams: int, upper_deg: float, lower_deg: float) -> "BeamModel":
        return cls(np.linspace(upper_deg, lower_deg, n_beams))

    @classmethod
    def from_config(cls, sensor_cfg: dict, repo_root: Path | None = None) -> "BeamModel":
        table = sensor_cfg.get("beam_table")
        if table:
            path = Path(table)
            if not path.is_absolute() and repo_root is not None:
                path = repo_root / path
            return cls(np.sort(np.loadtxt(path))[::-1])
        return cls.nominal(sensor_cfg["n_beams"], sensor_cfg["upper_deg"], sensor_cfg["lower_deg"])

    @property
    def n_beams(self) -> int:
        return len(self.angles_deg)

    @property
    def spacing_deg(self) -> np.ndarray:
        """Local beam spacing at each ring (used to normalise residuals)."""
        a = self.angles_deg
        if len(a) == 1:
            return np.array([1.0])
        d = np.abs(np.diff(a))
        return np.concatenate([[d[0]], np.minimum(d[:-1], d[1:]), [d[-1]]])

    def assign(self, elev_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest ring for each point and the residual in units of local beam spacing.

        Returns (ring, resid) where |resid| ~ 0 means the point sits exactly on a
        beam and 0.5 means it is halfway between two beams (ambiguous).
        """
        asc = self.angles_deg[::-1]  # ascending for searchsorted
        idx = np.clip(np.searchsorted(asc, elev_deg), 1, len(asc) - 1)
        left, right = asc[idx - 1], asc[idx]
        pick_right = np.abs(right - elev_deg) < np.abs(elev_deg - left)
        asc_idx = np.where(pick_right, idx, idx - 1)
        ring = (len(asc) - 1) - asc_idx  # back to ring-0-is-top
        resid = (elev_deg - self.angles_deg[ring]) / self.spacing_deg[ring]
        return ring.astype(np.int32), resid

    def sharpness(self, elev_deg: np.ndarray) -> dict:
        """How well the points line up on the beams. ~0 = perfect, ~0.25 = random."""
        _, resid = self.assign(elev_deg)
        inside = (elev_deg <= self.angles_deg[0] + 0.5 * self.spacing_deg[0]) & \
                 (elev_deg >= self.angles_deg[-1] - 0.5 * self.spacing_deg[-1])
        r = np.abs(resid[inside])
        if r.size == 0:
            return {"median_abs_resid": float("nan"), "frac_within_0.1": 0.0, "frac_inside_fov": 0.0}
        return {
            "median_abs_resid": float(np.median(r)),
            "frac_within_0.1": float(np.mean(r < 0.1)),
            "frac_inside_fov": float(np.mean(inside)),
        }


def select_rings(model: BeamModel, cfg: dict) -> np.ndarray:
    """Ring indices (ring-0-is-top convention) to KEEP for one downsampling config."""
    kind = cfg["type"]
    if kind == "every_k":
        k = int(cfg["k"])
        offset = int(cfg.get("offset", 0))
        return np.arange(offset, model.n_beams, k)
    if kind == "angles":
        if "linspace" in cfg:
            lo, hi, n = cfg["linspace"]
            targets = np.linspace(lo, hi, int(n))
        else:
            targets = np.asarray(cfg["angles_deg"], dtype=float)
        lo_ok = model.angles_deg[-1] - model.spacing_deg[-1]
        hi_ok = model.angles_deg[0] + model.spacing_deg[0]
        out_of_fov = targets[(targets < lo_ok) | (targets > hi_ok)]
        if out_of_fov.size:
            raise ValueError(f"target angles {out_of_fov} are outside the simulated FOV "
                             f"[{model.angles_deg[-1]:.2f}, {model.angles_deg[0]:.2f}]")
        rings = np.argmin(np.abs(model.angles_deg[None, :] - targets[:, None]), axis=1)
        rings = np.unique(rings)
        if len(rings) < len(targets):
            raise ValueError("two target angles snapped to the same ring; spacing too fine")
        return rings
    raise ValueError(f"unknown config type {kind!r}")


def loam_ring_index(kept_rings: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Re-index kept rings to 0..M-1 with 0 = bottom beam (LeGO-LOAM / FAST-LIO convention)."""
    kept_sorted_bottom_up = np.sort(kept_rings)[::-1]  # larger ring index = lower beam
    lut = np.full(int(max(kept_rings.max(), ring.max())) + 1, -1, dtype=np.int32)
    lut[kept_sorted_bottom_up] = np.arange(len(kept_sorted_bottom_up))
    return lut[ring]
