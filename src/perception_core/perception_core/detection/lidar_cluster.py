"""Classical geometric LiDAR obstacle detector.

Pipeline: ROI crop -> RANSAC ground-plane removal -> DBSCAN clustering ->
per-cluster oriented bounding box (PCA on the ground plane) -> size/shape based
class heuristic. This is a robust, GPU-free baseline that always works, and it
serves as the default detector so the whole stack is runnable out of the box.

Only NumPy, SciPy and scikit-learn are required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from perception_core.common.types import CLASS_DIMENSIONS, Box3D, Detection, Frame, ObjectClass
from perception_core.detection.base import Detector


@dataclass
class LidarClusterConfig:
    # Region of interest in the LiDAR frame (metres).
    x_range: Tuple[float, float] = (-50.0, 50.0)
    y_range: Tuple[float, float] = (-40.0, 40.0)
    z_range: Tuple[float, float] = (-3.0, 4.0)
    # RANSAC ground-plane removal.
    ground_ransac: bool = True
    ground_dist_thresh: float = 0.2
    ground_max_iter: int = 50
    ground_max_normal_tilt: float = 0.3  # reject planes whose normal tilts > this (rad-ish, 1-|nz|)
    # DBSCAN clustering (run on x, y, z).
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 8
    # Cluster acceptance filters.
    min_points: int = 12
    min_extent: float = 0.2
    max_extent: float = 20.0
    max_height: float = 4.5
    seed: Optional[int] = 0


def fit_ground_plane(
    points: np.ndarray, dist_thresh: float, max_iter: int, max_tilt: float, rng: np.random.Generator
) -> np.ndarray:
    """RANSAC plane fit. Returns a boolean mask of ground (inlier) points."""
    n = len(points)
    if n < 3:
        return np.zeros(n, dtype=bool)
    xyz = points[:, :3]
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(max_iter):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = xyz[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if 1.0 - abs(normal[2]) > max_tilt:  # keep only near-horizontal planes
            continue
        dist = np.abs((xyz - p0) @ normal)
        mask = dist < dist_thresh
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    return best_mask
def oriented_box_from_points(pts: np.ndarray) -> Box3D:
    """Fit a minimal oriented BEV box to a cluster via PCA on the (x, y) plane."""
    xy = pts[:, :2]
    mean = xy.mean(axis=0)
    centred = xy - mean
    cov = np.cov(centred.T) if len(pts) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, int(np.argmax(evals))]
    yaw = float(np.arctan2(principal[1], principal[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, s], [-s, c]])  # world -> box-local
    local = centred @ rot.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    l, w = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if w > l:  # keep length along the box's x axis
        l, w = w, l
        yaw += np.pi / 2.0
    mid_local = (lo + hi) / 2.0
    center_xy = mean + mid_local @ rot  # rotate box-local midpoint back to world
    z_lo, z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
    return Box3D(
        x=float(center_xy[0]), y=float(center_xy[1]), z=(z_lo + z_hi) / 2.0,
        l=max(l, 1e-2), w=max(w, 1e-2), h=max(z_hi - z_lo, 1e-2),
        yaw=float((yaw + np.pi) % (2 * np.pi) - np.pi),
    )


def classify_by_size(box: Box3D) -> ObjectClass:
    """Assign a coarse class from footprint dimensions (nearest-prior heuristic)."""
    dims = box.dimensions
    best, best_err = ObjectClass.UNKNOWN, np.inf
    for cls, prior in CLASS_DIMENSIONS.items():
        if cls is ObjectClass.UNKNOWN:
            continue
        err = float(np.sum(np.abs(np.sort(dims) - np.sort(prior)) / prior))
        if err < best_err:
            best_err, best = err, cls
    return best if best_err < 2.0 else ObjectClass.UNKNOWN


class LidarClusterDetector(Detector):
    name = "lidar_cluster"

    def __init__(self, config: Optional[LidarClusterConfig] = None) -> None:
        self.cfg = config or LidarClusterConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def _crop_roi(self, points: np.ndarray) -> np.ndarray:
        c = self.cfg
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        mask = (
            (x >= c.x_range[0]) & (x <= c.x_range[1])
            & (y >= c.y_range[0]) & (y <= c.y_range[1])
            & (z >= c.z_range[0]) & (z <= c.z_range[1])
        )
        return points[mask]

    def detect(self, frame: Frame) -> List[Detection]:
        if frame.lidar is None or len(frame.lidar) == 0:
            return []
        cfg = self.cfg
        pts = self._crop_roi(np.asarray(frame.lidar, dtype=float))
        if len(pts) < cfg.dbscan_min_samples:
            return []
        if cfg.ground_ransac:
            ground = fit_ground_plane(
                pts, cfg.ground_dist_thresh, cfg.ground_max_iter, cfg.ground_max_normal_tilt, self.rng
            )
            pts = pts[~ground]
        if len(pts) < cfg.dbscan_min_samples:
            return []
        labels = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit_predict(pts[:, :3])

        detections: List[Detection] = []
        for lab in set(labels):
            if lab == -1:
                continue
            cluster = pts[labels == lab]
            if len(cluster) < cfg.min_points:
                continue
            box = oriented_box_from_points(cluster)
            extent = max(box.l, box.w)
            if extent < cfg.min_extent or extent > cfg.max_extent or box.h > cfg.max_height:
                continue
            label = classify_by_size(box)
            score = float(np.clip(len(cluster) / 200.0, 0.1, 1.0))
            detections.append(
                Detection(box=box, score=score, label=label, source=self.name, num_points=len(cluster))
            )
        return detections

