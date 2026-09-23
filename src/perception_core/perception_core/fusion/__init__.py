"""Late (decision-level) camera-LiDAR fusion.

Each 3D LiDAR detection is projected into the calibrated camera images and
matched against 2D image detections (e.g. YOLO) by 2D IoU. A successful match
transfers the semantically richer camera class label onto the geometrically
accurate LiDAR box and boosts its confidence. LiDAR-only detections survive as
lower-confidence obstacles, so the fusion is fail-safe.
"""

from perception_core.fusion.late_fusion import (
    Box2D,
    Detection2D,
    LateFusion,
    LateFusionConfig,
    project_box_to_image,
)

__all__ = [
    "Box2D",
    "Detection2D",
    "LateFusion",
    "LateFusionConfig",
    "project_box_to_image",
]
