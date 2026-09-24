"""Multi-object tracking: constant-velocity Kalman filtering plus greedy/optimal
association. A clean, dependency-light re-implementation of the AB3DMOT recipe,
with an optional Interacting Multiple Model (CV + constant-turn) estimator.
"""

from perception_core.tracking.imm import IMMFilter
from perception_core.tracking.kalman import ConstantVelocityKF
from perception_core.tracking.mot import MultiObjectTracker, TrackerConfig

__all__ = ["ConstantVelocityKF", "IMMFilter", "MultiObjectTracker", "TrackerConfig"]
