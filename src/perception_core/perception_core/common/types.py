"""Framework-agnostic value types shared across the perception pipeline.

These are plain dataclasses backed by NumPy arrays. They intentionally do not
depend on ROS messages so the core algorithms can be unit-tested in isolation
and reused from a ROS node, a CARLA client or an offline batch script alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "ObjectClass",
    "CLASS_DIMENSIONS",
    "Box3D",
    "Detection",
    "TrackState",
    "Track",
    "TrajectoryPoint",
    "Trajectory",
    "PredictedObject",
    "SensorCalibration",
    "Frame",
    "PerceptionOutput",
]


class ObjectClass(str, Enum):
    """Coarse object taxonomy shared by every detector backend."""

    UNKNOWN = "unknown"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    VAN = "van"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    PEDESTRIAN = "pedestrian"

    @classmethod
    def from_str(cls, name: str) -> "ObjectClass":
        try:
            return cls(str(name).strip().lower())
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_vru(self) -> bool:
        """Vulnerable road user (pedestrian / cyclist / motorcyclist)."""
        return self in (ObjectClass.PEDESTRIAN, ObjectClass.BICYCLE, ObjectClass.MOTORCYCLE)


# Nominal (length, width, height) priors in metres, used to regularise the
# raw oriented boxes produced by the geometric LiDAR clusterer.
CLASS_DIMENSIONS: Dict[ObjectClass, np.ndarray] = {
    ObjectClass.CAR: np.array([4.5, 1.9, 1.6]),
    ObjectClass.TRUCK: np.array([8.0, 2.6, 3.2]),
    ObjectClass.BUS: np.array([11.0, 2.9, 3.4]),
    ObjectClass.VAN: np.array([5.5, 2.2, 2.3]),
    ObjectClass.BICYCLE: np.array([1.8, 0.7, 1.5]),
    ObjectClass.MOTORCYCLE: np.array([2.1, 0.8, 1.5]),
    ObjectClass.PEDESTRIAN: np.array([0.7, 0.7, 1.75]),
    ObjectClass.UNKNOWN: np.array([1.0, 1.0, 1.0]),
}
@dataclass
class Box3D:
    """Oriented 3D bounding box in a right-handed frame (x forward, y left, z up).

    ``yaw`` is the heading around +z in radians. ``l``/``w``/``h`` are the full
    extents along the box's local x/y/z axes.
    """

    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def dimensions(self) -> np.ndarray:
        return np.array([self.l, self.w, self.h], dtype=float)

    @property
    def volume(self) -> float:
        return float(self.l * self.w * self.h)

    def corners(self) -> np.ndarray:
        """Return the 8 box corners as a (8, 3) array in the parent frame."""
        dx, dy, dz = self.l / 2.0, self.w / 2.0, self.h / 2.0
        local = np.array(
            [
                [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],
                [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz],
            ]
        )
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return local @ rot.T + self.center

    def bev_corners(self) -> np.ndarray:
        """Return the 4 ground-plane footprint corners as a (4, 2) array (CCW)."""
        dx, dy = self.l / 2.0, self.w / 2.0
        local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([self.x, self.y])

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.l, self.w, self.h, self.yaw], dtype=float)

    @classmethod
    def from_array(cls, a) -> "Box3D":
        a = np.asarray(a, dtype=float).reshape(-1)
        return cls(*a[:7])

    def copy(self) -> "Box3D":
        return Box3D(self.x, self.y, self.z, self.l, self.w, self.h, self.yaw)
@dataclass
class Detection:
    """A single-frame detection produced by a :class:`~perception_core.detection.base.Detector`."""

    box: Box3D
    score: float = 1.0
    label: ObjectClass = ObjectClass.UNKNOWN
    source: str = ""  # e.g. "lidar_cluster", "yolo", "fusion", "ground_truth"
    num_points: int = 0
    attributes: Dict[str, float] = field(default_factory=dict)


class TrackState(Enum):
    """Lifecycle state of a track (classic tentative/confirmed/deleted model)."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class Track:
    """A confirmed or tentative object track carrying an estimated velocity."""

    track_id: int
    box: Box3D
    label: ObjectClass = ObjectClass.UNKNOWN
    score: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (vx, vy) m/s in world frame
    yaw_rate: float = 0.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: TrackState = TrackState.TENTATIVE
    history: List[np.ndarray] = field(default_factory=list)  # past (x, y) centers

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def is_confirmed(self) -> bool:
        return self.state is TrackState.CONFIRMED


@dataclass
class TrajectoryPoint:
    t: float  # seconds relative to prediction time
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class Trajectory:
    """One predicted motion hypothesis for an object."""

    points: List[TrajectoryPoint] = field(default_factory=list)
    confidence: float = 1.0
    mode: str = "constant_velocity"

    def as_array(self) -> np.ndarray:
        """Return the trajectory as a (T, 2) array of (x, y) waypoints."""
        if not self.points:
            return np.zeros((0, 2))
        return np.array([[p.x, p.y] for p in self.points], dtype=float)


@dataclass
class PredictedObject:
    """Multi-modal motion forecast attached to a track."""

    track_id: int
    label: ObjectClass
    current_box: Box3D
    trajectories: List[Trajectory] = field(default_factory=list)

    @property
    def best(self) -> Optional[Trajectory]:
        if not self.trajectories:
            return None
        return max(self.trajectories, key=lambda t: t.confidence)
@dataclass
class SensorCalibration:
    """Extrinsics/intrinsics needed to fuse LiDAR and cameras.

    All transforms are 4x4 homogeneous matrices. ``lidar_to_cam[cam]`` maps a
    point expressed in the LiDAR frame into the camera frame; ``intrinsics[cam]``
    is the 3x3 pinhole matrix. Missing entries simply disable that camera.
    """

    lidar_to_cam: Dict[str, np.ndarray] = field(default_factory=dict)
    intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)
    image_size: Dict[str, tuple] = field(default_factory=dict)  # cam -> (width, height)

    def has_camera(self, cam: str) -> bool:
        return cam in self.lidar_to_cam and cam in self.intrinsics


@dataclass
class Frame:
    """A single synchronised sensor sample fed to the pipeline."""

    timestamp: float
    frame_id: int = 0
    lidar: Optional[np.ndarray] = None  # (N, >=3) x, y, z[, intensity] in LiDAR frame
    images: Dict[str, np.ndarray] = field(default_factory=dict)  # cam -> HxWx3 uint8
    calib: Optional[SensorCalibration] = None
    ego_pose: Optional[np.ndarray] = None  # 4x4 world <- ego(lidar) transform
    ground_truth: Optional[List[Detection]] = None

    @property
    def num_points(self) -> int:
        return 0 if self.lidar is None else int(self.lidar.shape[0])


@dataclass
class PerceptionOutput:
    """Bundle of everything the pipeline produces for one frame."""

    timestamp: float
    frame_id: int = 0
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    predictions: List[PredictedObject] = field(default_factory=list)



