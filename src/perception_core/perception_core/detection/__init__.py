"""Detector backends.

All detectors implement the :class:`~perception_core.detection.base.Detector`
interface, so the pipeline is agnostic to whether obstacles come from a
classical LiDAR clusterer, a YOLO camera model or ground-truth replay.
"""

from perception_core.detection.base import Detector
from perception_core.detection.lidar_cluster import LidarClusterDetector, LidarClusterConfig
from perception_core.detection.mock import GroundTruthDetector

__all__ = [
    "Detector",
    "LidarClusterDetector",
    "LidarClusterConfig",
    "GroundTruthDetector",
]
