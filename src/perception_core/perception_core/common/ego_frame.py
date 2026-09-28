"""Re-express world-frame pipeline outputs in another frame (e.g. the ego frame).

The pipeline tracks and forecasts in a stable world frame, while a sensor-centric
view (a bird's-eye view that follows the car, or a camera overlay) needs them in
the current ego/LiDAR frame. These helpers apply one rigid transform to every
box, velocity, history trail and forecast waypoint, leaving the inputs untouched.
"""
from __future__ import annotations

from copy import copy
from typing import List

import numpy as np

from perception_core.common.geometry import transform_box
from perception_core.common.types import (
    Detection,
    PerceptionOutput,
    Trajectory,
    TrajectoryPoint,
)

__all__ = ["detections_to_frame", "output_to_frame"]


def detections_to_frame(dets: List[Detection], T: np.ndarray) -> List[Detection]:
    """Apply the 4x4 transform ``T`` to every detection box."""
    return [
        Detection(box=transform_box(d.box, T), score=d.score, label=d.label,
                  source=d.source, num_points=d.num_points, attributes=d.attributes)
        for d in dets
    ]


def output_to_frame(out: PerceptionOutput, T: np.ndarray) -> PerceptionOutput:
    """Apply ``T`` to all tracks (box, velocity, history) and forecasts in ``out``."""
    R = T[:3, :3]
    dyaw = float(np.arctan2(T[1, 0], T[0, 0]))

    def _xy(x: float, y: float, z: float) -> np.ndarray:
        return (R @ np.array([x, y, z]) + T[:3, 3])[:2]

    tracks = []
    for tr in out.tracks:
        t2 = copy(tr)
        t2.box = transform_box(tr.box, T)
        t2.velocity = R[:2, :2] @ np.asarray(tr.velocity, dtype=float)
        # The tracker is planar: use the current box height for past centres.
        t2.history = [_xy(p[0], p[1], tr.box.z) for p in tr.history]
        tracks.append(t2)

    preds = []
    for p in out.predictions:
        p2 = copy(p)
        p2.current_box = transform_box(p.current_box, T)
        p2.trajectories = []
        for traj in p.trajectories:
            pts = []
            for tp in traj.points:
                x, y = _xy(tp.x, tp.y, p.current_box.z)
                yaw = (tp.yaw + dyaw + np.pi) % (2 * np.pi) - np.pi
                pts.append(TrajectoryPoint(t=tp.t, x=float(x), y=float(y), yaw=float(yaw)))
            p2.trajectories.append(Trajectory(points=pts, confidence=traj.confidence, mode=traj.mode))
        preds.append(p2)

    return PerceptionOutput(timestamp=out.timestamp, frame_id=out.frame_id,
                            detections=detections_to_frame(out.detections, T),
                            tracks=tracks, predictions=preds)
