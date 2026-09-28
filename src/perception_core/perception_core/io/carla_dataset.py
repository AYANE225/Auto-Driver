"""Reader for scenarios recorded from CARLA by ``carla/record_scenario.py``.

The recorder (which runs under CARLA's own Python client) writes a
self-describing, framework-neutral dataset to disk. This reader reconstructs
:class:`~perception_core.common.types.Frame` objects from it, so **CARLA data
flows through exactly the same pipeline as the synthetic generator and KITTI** —
no CARLA or ROS dependency is needed to replay a recording.

Layout on disk::

    <dataset>/
        meta.json              # scenario, CARLA version, dt, sensor list
        calib.json             # per-camera lidar->cam (optical) extrinsics + intrinsics
        frames/
            000000.npz         # points, ego_pose, timestamp, gt boxes/labels/ids
            000000_<cam>.jpg   # optional camera image(s)
            ...

Ground-truth boxes are stored in the world frame. The CARLA actor id of each
box is exposed as ``Detection.attributes['gt_id']`` (same convention as the
KITTI reader) so tracking ID switches are scored against a stable identity, and
``Detection.num_points`` holds the number of LiDAR returns inside the box so an
evaluation can skip objects the sensor never saw (fully occluded / out of range).
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

import numpy as np

from perception_core.common.geometry import invert_se3, points_in_box, transform_box
from perception_core.common.types import (
    Box3D,
    Detection,
    Frame,
    ObjectClass,
    SensorCalibration,
)


class CarlaDataset:
    """Random-access reader over a recorded CARLA scenario."""

    def __init__(self, root: str, load_images: bool = False, count_points: bool = True) -> None:
        self.root = root
        self.load_images = load_images
        self.count_points = count_points
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
        points = data["points"].astype(np.float32)
        ego_pose = data["ego_pose"].astype(float) if "ego_pose" in data else None

        images = {}
        if self.load_images and self.cameras:
            from PIL import Image  # optional, only when images are requested
            for cam in self.cameras:
                img_path = os.path.join(self._frames_dir, f"{fid:06d}_{cam}.jpg")
                if os.path.exists(img_path):
                    with Image.open(img_path) as im:
                        images[cam] = np.asarray(im.convert("RGB"))

        return Frame(
            timestamp=float(data["timestamp"]),
            frame_id=fid,
            lidar=points,
            images=images,
            calib=self.calib,
            ego_pose=ego_pose,
            ground_truth=self._ground_truth(data, points, ego_pose),
        )

    def _ground_truth(self, data, points: np.ndarray,
                      ego_pose: Optional[np.ndarray]) -> List[Detection]:
        if "gt_boxes" not in data or len(data["gt_boxes"]) == 0:
            return []
        boxes = np.atleast_2d(data["gt_boxes"])
        labels = data["gt_labels"] if "gt_labels" in data else ["unknown"] * len(boxes)
        ids = data["gt_ids"] if "gt_ids" in data else np.arange(len(boxes))
        to_sensor = invert_se3(ego_pose) if ego_pose is not None else np.eye(4)
        dets: List[Detection] = []
        for row, label, gid in zip(boxes, labels, ids):
            box = Box3D.from_array(row)
            n_pts = 0
            if self.count_points and len(points):
                n_pts = int(points_in_box(points, transform_box(box, to_sensor)).sum())
            dets.append(Detection(
                box=box, score=1.0, label=ObjectClass.from_str(str(label)),
                source="carla_gt", num_points=n_pts, attributes={"gt_id": float(gid)},
            ))
        return dets
