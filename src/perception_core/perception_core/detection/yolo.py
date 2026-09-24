"""Optional YOLO camera detector feeding the late-fusion path.

Requires the ``yolo`` extra (``ultralytics`` + ``opencv-python``). ``ultralytics``
is imported lazily so the rest of the library — and CI — stays free of any heavy
torch/GPU dependency. The detector emits 2D image-space :class:`Detection2D`
boxes keyed by camera name, which :class:`~perception_core.fusion.late_fusion.LateFusion`
matches against projected LiDAR boxes to transfer class labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from perception_core.common.types import Frame, ObjectClass
from perception_core.fusion.late_fusion import Box2D, Detection2D

# Map the COCO class names produced by the stock YOLO weights onto our taxonomy.
# Classes not listed (traffic light, stop sign, ...) are ignored.
_COCO_TO_OBJECTCLASS: Dict[str, ObjectClass] = {
    "person": ObjectClass.PEDESTRIAN,
    "bicycle": ObjectClass.BICYCLE,
    "car": ObjectClass.CAR,
    "motorcycle": ObjectClass.MOTORCYCLE,
    "motorbike": ObjectClass.MOTORCYCLE,
    "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK,
}


@dataclass
class YoloConfig:
    model: str = "yolov8n.pt"      # any ultralytics checkpoint or local .pt path
    conf: float = 0.25             # confidence threshold
    iou: float = 0.45              # NMS IoU threshold
    device: str = ""               # "", "cpu", "cuda:0" ... ("" = ultralytics auto)
    imgsz: int = 640
    verbose: bool = False
    #: restrict to these classes (default: everything in the COCO map above)
    keep: Optional[List[ObjectClass]] = None


class YoloCameraDetector:
    """Runs an ultralytics YOLO model on every image in a :class:`Frame`.

    Exposes :meth:`detect_2d` returning ``{camera_name: [Detection2D, ...]}`` so
    it slots directly into :class:`~perception_core.pipeline.PerceptionPipeline`'s
    ``camera_detector`` hook.
    """

    name = "yolo"

    def __init__(self, config: Optional[YoloConfig] = None) -> None:
        self.cfg = config or YoloConfig()
        try:
            from ultralytics import YOLO  # lazy: only needed when actually used
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "YoloCameraDetector requires the 'yolo' extra: "
                "pip install 'perception_core[yolo]'"
            ) from exc
        self._model = YOLO(self.cfg.model)
        self._keep = set(self.cfg.keep) if self.cfg.keep is not None else None

    def detect_2d(self, frame: Frame) -> Dict[str, List[Detection2D]]:
        if not frame.images:
            return {}
        out: Dict[str, List[Detection2D]] = {}
        for cam, image in frame.images.items():
            out[cam] = self._detect_image(image, cam)
        return out

    def _detect_image(self, image: np.ndarray, cam: str) -> List[Detection2D]:
        results = self._model.predict(
            image, conf=self.cfg.conf, iou=self.cfg.iou,
            device=self.cfg.device or None, imgsz=self.cfg.imgsz,
            verbose=self.cfg.verbose,
        )
        dets: List[Detection2D] = []
        for res in results:
            names = res.names  # class index -> COCO name
            boxes = res.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            for (u1, v1, u2, v2), score, cls_idx in zip(xyxy, confs, clses):
                label = _COCO_TO_OBJECTCLASS.get(names.get(int(cls_idx), ""), None)
                if label is None:
                    continue
                if self._keep is not None and label not in self._keep:
                    continue
                dets.append(Detection2D(
                    box=Box2D(float(u1), float(v1), float(u2), float(v2)),
                    score=float(score), label=label, camera=cam,
                ))
        return dets
