"""Physics-based motion predictors (constant velocity / constant turn rate).

These are the honest, interpretable baselines used across the AV industry for
short-horizon forecasting. ``MotionPredictor`` rolls each track's estimated
state forward and can emit multiple weighted hypotheses (e.g. a turning object
also gets a straight-line fallback), matching the multi-modal output that
downstream planners expect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from perception_core.common.types import (
    PredictedObject,
    Track,
    Trajectory,
    TrajectoryPoint,
)


@dataclass
class PredictorConfig:
    horizon: float = 3.0            # seconds to forecast
    step: float = 0.5              # temporal resolution of the forecast
    mode: str = "auto"             # "cv" | "ctrv" | "auto"
    min_turn_speed: float = 1.0    # below this speed heading is unreliable -> CV
    turn_rate_thresh: float = 0.03  # |yaw_rate| above which "auto" chooses CTRV


def _timesteps(cfg: PredictorConfig) -> np.ndarray:
    n = max(1, int(round(cfg.horizon / cfg.step)))
    return np.arange(1, n + 1) * cfg.step


def _cv_trajectory(track: Track, cfg: PredictorConfig, confidence: float) -> Trajectory:
    x0, y0 = track.box.x, track.box.y
    vx, vy = track.velocity
    yaw = float(np.arctan2(vy, vx)) if track.speed > 1e-3 else track.box.yaw
    pts = [TrajectoryPoint(t=float(t), x=float(x0 + vx * t), y=float(y0 + vy * t), yaw=yaw)
           for t in _timesteps(cfg)]
    return Trajectory(points=pts, confidence=confidence, mode="constant_velocity")


def _ctrv_trajectory(track: Track, cfg: PredictorConfig, confidence: float) -> Trajectory:
    x, y = track.box.x, track.box.y
    v = track.speed
    w = track.yaw_rate
    yaw0 = float(np.arctan2(track.velocity[1], track.velocity[0])) if v > 1e-3 else track.box.yaw
    pts: List[TrajectoryPoint] = []
    for t in _timesteps(cfg):
        if abs(w) < 1e-6:
            nx, ny, nyaw = x + v * np.cos(yaw0) * t, y + v * np.sin(yaw0) * t, yaw0
        else:
            nyaw = yaw0 + w * t
            nx = x + v / w * (np.sin(nyaw) - np.sin(yaw0))
            ny = y + v / w * (-np.cos(nyaw) + np.cos(yaw0))
        pts.append(TrajectoryPoint(t=float(t), x=float(nx), y=float(ny), yaw=float(nyaw)))
    return Trajectory(points=pts, confidence=confidence, mode="ctrv")


class MotionPredictor:
    def __init__(self, config: PredictorConfig = None) -> None:
        self.cfg = config or PredictorConfig()

    def predict(self, track: Track) -> PredictedObject:
        cfg = self.cfg
        turning = track.speed >= cfg.min_turn_speed and abs(track.yaw_rate) >= cfg.turn_rate_thresh
        trajectories: List[Trajectory] = []
        if cfg.mode == "cv" or (cfg.mode == "auto" and not turning):
            trajectories.append(_cv_trajectory(track, cfg, confidence=1.0))
        elif cfg.mode == "ctrv":
            trajectories.append(_ctrv_trajectory(track, cfg, confidence=1.0))
        else:  # auto + turning -> CTRV primary, CV as a lower-confidence fallback
            trajectories.append(_ctrv_trajectory(track, cfg, confidence=0.7))
            trajectories.append(_cv_trajectory(track, cfg, confidence=0.3))
        return PredictedObject(
            track_id=track.track_id, label=track.label,
            current_box=track.box.copy(), trajectories=trajectories,
        )

    def predict_all(self, tracks: List[Track]) -> List[PredictedObject]:
        return [self.predict(t) for t in tracks]
