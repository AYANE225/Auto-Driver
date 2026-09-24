"""End-to-end perception pipeline wiring the four stages together.

    detect  ->  (camera-LiDAR fuse)  ->  track  ->  predict

The pipeline is deliberately framework-agnostic: it consumes :class:`Frame`
objects (from the synthetic generator, a dataset reader or the CARLA/ROS bridge)
and returns a :class:`PerceptionOutput`. Detections are lifted into the world
frame using ``frame.ego_pose`` before tracking, so track states and forecasts
live in a stable global frame even while the ego vehicle moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional

from perception_core.common.geometry import transform_box
from perception_core.common.types import Detection, Frame, PerceptionOutput
from perception_core.detection.base import Detector
from perception_core.detection.lidar_cluster import LidarClusterConfig, LidarClusterDetector
from perception_core.fusion.late_fusion import LateFusion, LateFusionConfig
from perception_core.prediction.physics import MotionPredictor, PredictorConfig
from perception_core.tracking.mot import MultiObjectTracker, TrackerConfig


@dataclass
class PipelineConfig:
    lidar: LidarClusterConfig = field(default_factory=LidarClusterConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    fusion: LateFusionConfig = field(default_factory=LateFusionConfig)
    track_in_world: bool = True


class PerceptionPipeline:
    def __init__(
        self,
        detector: Optional[Detector] = None,
        tracker: Optional[MultiObjectTracker] = None,
        predictor: Optional[MotionPredictor] = None,
        fusion: Optional[LateFusion] = None,
        camera_detector=None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.cfg = config or PipelineConfig()
        self.detector = detector or LidarClusterDetector(self.cfg.lidar)
        self.tracker = tracker or MultiObjectTracker(self.cfg.tracker)
        self.predictor = predictor or MotionPredictor(self.cfg.predictor)
        self.fusion = fusion or LateFusion(self.cfg.fusion)
        self.camera_detector = camera_detector  # optional, e.g. YoloCameraDetector
        self.last_timings: Dict[str, float] = {}  # per-stage latency (ms) of the last frame

    def process(self, frame: Frame) -> PerceptionOutput:
        t0 = perf_counter()
        detections: List[Detection] = self.detector.detect(frame)
        t_detect = perf_counter()

        if self.camera_detector is not None and frame.images:
            cam_dets = self.camera_detector.detect_2d(frame)
            detections = self.fusion.fuse(detections, cam_dets, frame.calib)
        t_fuse = perf_counter()

        if self.cfg.track_in_world and frame.ego_pose is not None:
            detections = [
                Detection(box=transform_box(d.box, frame.ego_pose), score=d.score,
                          label=d.label, source=d.source, num_points=d.num_points,
                          attributes=d.attributes)
                for d in detections
            ]

        tracks = self.tracker.update(detections, frame.timestamp)
        t_track = perf_counter()
        predictions = self.predictor.predict_all(tracks)
        t_predict = perf_counter()

        self.last_timings = {
            "detect_ms": (t_detect - t0) * 1e3,
            "fuse_ms": (t_fuse - t_detect) * 1e3,
            "track_ms": (t_track - t_fuse) * 1e3,   # incl. world-frame lift
            "predict_ms": (t_predict - t_track) * 1e3,
            "total_ms": (t_predict - t0) * 1e3,
        }
        return PerceptionOutput(
            timestamp=frame.timestamp, frame_id=frame.frame_id,
            detections=detections, tracks=tracks, predictions=predictions,
        )

    def reset(self) -> None:
        self.tracker.reset()
