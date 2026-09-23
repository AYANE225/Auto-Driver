"""perception_core: framework-agnostic autonomous-driving perception stack.

The package deliberately avoids any hard dependency on ROS, CARLA or a deep
learning framework. It exposes a small set of dataclasses (:mod:`perception_core.common.types`)
and pluggable components that are wired together by :class:`perception_core.pipeline.PerceptionPipeline`:

    detection  ->  fusion  ->  tracking  ->  prediction

Each stage is independently unit-tested. Heavy or optional backends (YOLO,
PointPillars, Open3D visualisation) are imported lazily so the core runs on a
plain NumPy/SciPy install.
"""

from perception_core.common.types import (
    Box3D,
    Detection,
    Frame,
    ObjectClass,
    PerceptionOutput,
    PredictedObject,
    SensorCalibration,
    Track,
    TrackState,
    Trajectory,
)

__version__ = "0.1.0"

__all__ = [
    "Box3D",
    "Detection",
    "Frame",
    "ObjectClass",
    "PerceptionOutput",
    "PredictedObject",
    "SensorCalibration",
    "Track",
    "TrackState",
    "Trajectory",
    "__version__",
]
