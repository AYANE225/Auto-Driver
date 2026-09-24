"""Optional VGGT camera front-end: lift images into a pseudo-LiDAR point cloud.

`VGGT <https://github.com/facebookresearch/vggt>`_ (Visual Geometry Grounded
Transformer, CVPR 2025) is a **public**, pretrained feed-forward model that
predicts dense 3D point maps (and depth / camera pose) from one or more images.
This module wraps the public checkpoint ``facebook/VGGT-1B`` so a *camera-only*
:class:`~perception_core.common.types.Frame` can be turned into the same
``(N, 4)`` point-cloud layout the LiDAR pipeline already consumes — the predicted
point map becomes a "pseudo-LiDAR" sweep that flows, unchanged, through the
existing RANSAC-ground + DBSCAN clusterer, tracker and predictor.

Design notes / honest limitations:

* ``torch`` and the ``vggt`` package are imported **lazily**, only when the model
  actually runs, so ``perception_core`` and its CI stay free of any heavy
  torch/CUDA dependency (mirrors :mod:`perception_core.detection.yolo`).
* Only the **public** VGGT weights are used; no research code is involved.
* Monocular geometry is recovered **up to scale** — pass ``scale`` (or use a
  stereo/multi-view input) if metric size matters. The default assumes a roughly
  metric checkpoint output.

The torch-free post-processing (:func:`pointmap_to_cloud`) is unit-tested with
synthetic point maps; the learned inference path requires the ``vggt`` extra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from perception_core.common.types import Detection, Frame
from perception_core.detection.base import Detector
from perception_core.detection.lidar_cluster import LidarClusterConfig, LidarClusterDetector

# VGGT emits world points in an OpenCV-style camera frame (x right, y down,
# z forward). The pipeline works in an ego frame (x forward, y left, z up), so
# map (x, y, z)_cam -> (z, -x, -y)_ego.
_CAM_TO_EGO = np.array([[0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0]])


@dataclass
class VggtConfig:
    checkpoint: str = "facebook/VGGT-1B"  # public HuggingFace checkpoint
    device: str = ""                      # "", "cpu", "cuda", "cuda:0" ("" = auto)
    dtype: str = "bfloat16"               # inference autocast dtype on CUDA
    conf_percentile: float = 50.0         # drop the least-confident pixels below this pct
    max_points: int = 60000              # subsample the point map to keep clustering fast
    scale: float = 1.0                    # multiply recovered geometry (monocular is up-to-scale)
    seed: int = 0


def pointmap_to_cloud(
    points: np.ndarray,
    conf: Optional[np.ndarray] = None,
    *,
    conf_percentile: float = 50.0,
    max_points: int = 60000,
    cam_to_ego: Optional[np.ndarray] = None,
    scale: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Flatten a VGGT ``(..., 3)`` point map into an ``(N, 4)`` ego-frame cloud.

    Drops non-finite and low-confidence pixels, rotates the camera-frame points
    into the pipeline's ego frame (x forward, y left, z up), applies an optional
    metric ``scale`` and subsamples to ``max_points``. The 4th column carries the
    (normalised) confidence as a stand-in for LiDAR intensity.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    c = None if conf is None else np.asarray(conf, dtype=float).reshape(-1)
    if c is not None and c.shape[0] != pts.shape[0]:
        raise ValueError(f"conf has {c.shape[0]} entries but point map has {pts.shape[0]}")

    keep = np.isfinite(pts).all(axis=1)
    if c is not None:
        keep &= np.isfinite(c)
    pts, c = pts[keep], (None if c is None else c[keep])
    if pts.shape[0] == 0:
        return np.zeros((0, 4))

    if c is not None and 0.0 < conf_percentile < 100.0:
        thresh = float(np.percentile(c, conf_percentile))
        sel = c >= thresh
        pts, c = pts[sel], c[sel]

    R = _CAM_TO_EGO if cam_to_ego is None else np.asarray(cam_to_ego, dtype=float)
    pts = (pts @ R.T) * float(scale)

    if pts.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(pts.shape[0], max_points, replace=False)
        pts, c = pts[idx], (None if c is None else c[idx])

    if c is None:
        intensity = np.ones((pts.shape[0], 1))
    else:
        lo, hi = float(c.min()), float(c.max())
        intensity = ((c - lo) / (hi - lo) if hi > lo else np.ones_like(c)).reshape(-1, 1)
    return np.hstack([pts, intensity])


class VggtDepthBackend:
    """Lazily-loaded wrapper around the public VGGT model.

    :meth:`infer_cloud` takes a list of ``HxWx3`` uint8 images and returns an
    ``(N, 4)`` pseudo-LiDAR cloud in the ego frame. The model and ``torch`` are
    only imported on first use.
    """

    def __init__(self, config: Optional[VggtConfig] = None) -> None:
        self.cfg = config or VggtConfig()
        self._model = None
        self._torch = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            import torch
            from vggt.models.vggt import VGGT
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "VggtDepthBackend requires the 'vggt' extra (torch + the public "
                "vggt package): pip install 'perception_core[vggt]' and see "
                "https://github.com/facebookresearch/vggt"
            ) from exc
        self._torch = torch
        device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._model = VGGT.from_pretrained(self.cfg.checkpoint).to(device).eval()

    def infer_cloud(self, images: List[np.ndarray]) -> np.ndarray:
        """Run VGGT on ``images`` and return an ``(N, 4)`` ego-frame cloud."""
        if not images:
            return np.zeros((0, 4))
        self._lazy_load()
        torch = self._torch
        # (S, 3, H, W) float tensor in [0, 1]
        batch = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in images])
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self._device)
        autocast = (
            torch.autocast("cuda", dtype=getattr(torch, self.cfg.dtype))
            if self._device.startswith("cuda") else torch.autocast("cpu", enabled=False)
        )
        with torch.no_grad(), autocast:
            pred = self._model(tensor)
        points = pred["world_points"].float().cpu().numpy()          # (B, S, H, W, 3)
        conf = pred.get("world_points_conf")
        conf = None if conf is None else conf.float().cpu().numpy()   # (B, S, H, W)
        return pointmap_to_cloud(
            points, conf, conf_percentile=self.cfg.conf_percentile,
            max_points=self.cfg.max_points, scale=self.cfg.scale, seed=self.cfg.seed,
        )


class VggtLidarDetector(Detector):
    """Camera-only detector: VGGT pseudo-LiDAR → the classical LiDAR clusterer.

    Composes a :class:`VggtDepthBackend` (image → pseudo-LiDAR) with an ordinary
    :class:`~perception_core.detection.lidar_cluster.LidarClusterDetector`, so a
    Frame carrying only ``images`` produces 3D detections that flow through the
    exact same tracking/prediction stack as real LiDAR. Slots straight into the
    pipeline's ``detector`` interface.
    """

    name = "vggt_pseudo_lidar"

    def __init__(
        self,
        backend: Optional[VggtDepthBackend] = None,
        cluster: Optional[LidarClusterDetector] = None,
        config: Optional[VggtConfig] = None,
        cluster_config: Optional[LidarClusterConfig] = None,
        cameras: Optional[List[str]] = None,
    ) -> None:
        self.backend = backend or VggtDepthBackend(config)
        self.cluster = cluster or LidarClusterDetector(cluster_config)
        self.cameras = cameras  # which camera keys to feed (None -> all, sorted)

    def _select_images(self, frame: Frame) -> List[np.ndarray]:
        if not frame.images:
            return []
        keys = self.cameras if self.cameras is not None else sorted(frame.images)
        return [frame.images[k] for k in keys if k in frame.images]

    def detect(self, frame: Frame) -> List[Detection]:
        images = self._select_images(frame)
        if not images:
            return []
        cloud = self.backend.infer_cloud(images)
        if cloud.shape[0] == 0:
            return []
        pseudo = Frame(timestamp=frame.timestamp, frame_id=frame.frame_id,
                       lidar=cloud, ego_pose=frame.ego_pose)
        detections = self.cluster.detect(pseudo)
        for det in detections:
            det.source = self.name
        return detections
