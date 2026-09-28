"""Planning values: metres, seconds, radians; right-handed world XY.

Vehicle x/y is the rear axle midpoint. Object boxes retain the perception
convention (box centre). A MotionPlan's times are relative to its timestamp.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleConfig:
    wheelbase: float = 2.7
    length: float = 4.5
    width: float = 1.9
    rear_to_center: float = 1.35
    max_steer: float = 0.55
    max_steer_rate: float = 0.7
    max_accel: float = 2.0
    comfortable_brake: float = 3.0
    emergency_brake: float = 7.0
    max_lateral_accel: float = 3.0
    max_jerk: float = 5.0

    def __post_init__(self):
        values = [v for k, v in vars(self).items() if k != "rear_to_center"]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("vehicle limits and dimensions must be finite and positive")
        if not np.isfinite(self.rear_to_center) or abs(self.rear_to_center) > self.length / 2:
            raise ValueError("rear_to_center must lie within the vehicle")
        if self.max_steer >= np.pi / 2 or self.comfortable_brake > self.emergency_brake:
            raise ValueError("invalid steering or braking limits")


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    steering: float = 0.0
    acceleration: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all() or self.speed < 0:
            raise ValueError("vehicle state must be finite; reverse is not supported")


@dataclass
class MotionPlan:
    timestamp: float
    times: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    path_xy: np.ndarray
    status: str
    target_offset: float = 0.0
    feasible: bool = True
    candidate_count: int = 0
    rejected: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)


@dataclass
class ControlCommand:
    steering: float  # front-wheel angle, radians; positive turns left
    acceleration: float  # m/s^2, negative means brake
    emergency: bool = False
