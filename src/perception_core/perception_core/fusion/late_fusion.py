"""Decision-level camera-LiDAR fusion primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from perception_core.common.geometry import project_to_image, transform_points
from perception_core.common.types import Detection, ObjectClass, SensorCalibration


@dataclass
class Box2D:
    u1: float
    v1: float
    u2: float
    v2: float

    @property
    def area(self) -> float:
        return max(0.0, self.u2 - self.u1) * max(0.0, self.v2 - self.v1)

    def iou(self, other: "Box2D") -> float:
        iu1, iv1 = max(self.u1, other.u1), max(self.v1, other.v1)
        iu2, iv2 = min(self.u2, other.u2), min(self.v2, other.v2)
        inter = max(0.0, iu2 - iu1) * max(0.0, iv2 - iv1)
        union = self.area + other.area - inter
        return float(inter / union) if union > 1e-9 else 0.0


@dataclass
class Detection2D:
    box: Box2D
    score: float = 1.0
    label: ObjectClass = ObjectClass.UNKNOWN
    camera: str = ""


def project_box_to_image(
    box, lidar_to_cam: np.ndarray, K: np.ndarray, image_size: Optional[tuple] = None
) -> Optional[Box2D]:
    """Project a 3D box (in the LiDAR frame) to its 2D image bounding box.

    Returns ``None`` if the box is entirely behind the camera.
    """
    corners_cam = transform_points(box.corners(), lidar_to_cam)
    front = corners_cam[:, 2] > 0.1
    if not np.any(front):
        return None
    uv, _ = project_to_image(corners_cam[front], K)
    u1, v1 = uv.min(axis=0)
    u2, v2 = uv.max(axis=0)
    if image_size is not None:
        w, h = image_size
        u1, u2 = np.clip([u1, u2], 0, w)
        v1, v2 = np.clip([v1, v2], 0, h)
        if u2 - u1 < 1 or v2 - v1 < 1:
            return None
    return Box2D(float(u1), float(v1), float(u2), float(v2))
@dataclass
class LateFusionConfig:
    iou_threshold: float = 0.3     # min 2D IoU to accept a camera<->LiDAR match
    score_boost: float = 0.3       # confidence added to fused (camera-confirmed) boxes
    overwrite_label: bool = True   # let the camera class override the size heuristic


class LateFusion:
    def __init__(self, config: LateFusionConfig = None) -> None:
        self.cfg = config or LateFusionConfig()

    def fuse(
        self,
        lidar_detections: List[Detection],
        camera_detections: Dict[str, List[Detection2D]],
        calib: Optional[SensorCalibration],
    ) -> List[Detection]:
        """Return LiDAR detections with camera-confirmed labels and scores.

        If there is no calibration or no camera detections the input is returned
        unchanged, so the pipeline degrades gracefully to LiDAR-only.
        """
        if calib is None or not camera_detections:
            return lidar_detections

        fused: List[Detection] = []
        for det in lidar_detections:
            best_iou, best_cam_det = 0.0, None
            for cam, cam_dets in camera_detections.items():
                if not calib.has_camera(cam):
                    continue
                box2d = project_box_to_image(
                    det.box, calib.lidar_to_cam[cam], calib.intrinsics[cam],
                    calib.image_size.get(cam),
                )
                if box2d is None:
                    continue
                for cd in cam_dets:
                    iou = box2d.iou(cd.box)
                    if iou > best_iou:
                        best_iou, best_cam_det = iou, cd

            if best_cam_det is not None and best_iou >= self.cfg.iou_threshold:
                label = best_cam_det.label if self.cfg.overwrite_label else det.label
                score = min(1.0, det.score + self.cfg.score_boost * best_cam_det.score)
                attrs = dict(det.attributes)
                attrs["fusion_iou"] = float(best_iou)
                fused.append(
                    Detection(box=det.box, score=score, label=label, source="fusion",
                              num_points=det.num_points, attributes=attrs)
                )
            else:
                fused.append(det)
        return fused

