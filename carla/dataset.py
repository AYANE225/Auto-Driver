"""Reader for scenarios recorded from CARLA by :mod:`record_scenario`.

The recorder (which runs under CARLA's Python 3.8 client) writes a
self-describing, framework-neutral dataset to disk. This reader runs on the
ROS 2 / analysis side (Python 3.11) and reconstructs :class:`perception_core.Frame`
objects, so **CARLA data flows through exactly the same pipeline as the
synthetic generator** — no CARLA or ROS dependency required to replay it.

Layout on disk::

    <dataset>/
        meta.json              # scenario, carla version, dt, sensor list
        calib.json             # per-camera lidar->cam extrinsics + intrinsics
        frames/
            000000.npz         # points, ego_pose, timestamp, gt boxes/labels/ids
            000000_<cam>.jpg   # optional camera image(s)
            ...
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

import numpy as np

from perception_core.common.types import (
    Box3D,
    Detection,
    Frame,
    ObjectClass,
    SensorCalibration,
)


class CarlaDataset:
    """Random-access reader over a recorded CARLA scenario."""

    def __init__(self, root: str, load_images: bool = False) -> None:
        self.root = root
        self.load_images = load_images
        with open(os.path.join(root, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.cameras: List[str] = list(self.meta.get("cameras", []))
        self.dt: float = float(self.meta.get("dt", 0.1))
        self._frames_dir = os.path.join(root, "frames")
        self._ids = sorted(
            int(f[:-4]) for f in os.listdir(self._frames_dir)
            if f.endswith(".npz") and f[:-4].isdigit()
        )
        self.calib = self._load_calib()

    # -- construction helpers -------------------------------------------------
    def _load_calib(self) -> Optional[SensorCalibration]:
        path = os.path.join(self.root, "calib.json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            raw = json.load(fh)
        lidar_to_cam = {c: np.array(m, dtype=float) for c, m in raw.get("lidar_to_cam", {}).items()}
        intrinsics = {c: np.array(k, dtype=float) for c, k in raw.get("intrinsics", {}).items()}
        image_size = {c: tuple(s) for c, s in raw.get("image_size", {}).items()}
        return SensorCalibration(lidar_to_cam=lidar_to_cam, intrinsics=intrinsics, image_size=image_size)

    # -- access ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.read_frame(i)

    def read_frame(self, index: int) -> Frame:
        fid = self._ids[index]
        data = np.load(os.path.join(self._frames_dir, f"{fid:06d}.npz"))

        images = {}
        if self.load_images and self.cameras:
            from PIL import Image  # optional, only when images are requested
            for cam in self.cameras:
                img_path = os.path.join(self._frames_dir, f"{fid:06d}_{cam}.jpg")
                if os.path.exists(img_path):
                    images[cam] = np.asarray(Image.open(img_path))

        return Frame(
            timestamp=float(data["timestamp"]),
            frame_id=fid,
            lidar=data["points"].astype(np.float32),
            images=images,
            calib=self.calib,
            ego_pose=data["ego_pose"].astype(float) if "ego_pose" in data else None,
            ground_truth=self._boxes_to_detections(data),
        )

    @staticmethod
    def _boxes_to_detections(data) -> List[Detection]:
        if "gt_boxes" not in data or len(data["gt_boxes"]) == 0:
            return []
        boxes = np.atleast_2d(data["gt_boxes"])
        labels = data["gt_labels"] if "gt_labels" in data else ["unknown"] * len(boxes)
        dets: List[Detection] = []
        for row, label in zip(boxes, labels):
            dets.append(Detection(
                box=Box3D.from_array(row),
                score=1.0,
                label=ObjectClass.from_str(str(label)),
                source="carla_gt",
            ))
        return dets
