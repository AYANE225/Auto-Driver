"""Conversions between perception_core objects and ROS 2 messages.

Keeps all ROS <-> perception_core marshalling in one place so the nodes stay
thin. Frame convention matches perception_core: x forward, y left, z up, yaw
about +z.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from av_perception_msgs.msg import (
    PredictedObject as PredictedObjectMsg,
    PredictedObjectArray,
    PredictedTrajectory,
    TrackedObject,
    TrackedObjectArray,
)
from perception_core.common.types import PerceptionOutput, Track

# Distinct colours cycled by track id (mirrors perception_core.viz.bev palette).
_PALETTE = [
    (0.90, 0.10, 0.29), (0.24, 0.71, 0.29), (0.26, 0.39, 0.85),
    (0.96, 0.51, 0.19), (0.57, 0.12, 0.71), (0.26, 0.83, 0.96),
    (0.94, 0.20, 0.90), (0.75, 0.94, 0.27), (0.98, 0.75, 0.83),
    (0.27, 0.60, 0.56), (0.86, 0.75, 1.00), (0.60, 0.39, 0.14),
]


def _color(track_id: int, alpha: float = 1.0) -> ColorRGBA:
    r, g, b = _PALETTE[track_id % len(_PALETTE)]
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(alpha))


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


# --------------------------------------------------------------------------- #
# Point cloud
# --------------------------------------------------------------------------- #
def pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Return an (N, 4) float64 array of x, y, z, intensity.

    Intensity is filled with zeros when the incoming cloud lacks the field.
    """
    field_names = {f.name for f in msg.fields}
    xyz = pc2.read_points_numpy(msg, field_names=("x", "y", "z")).reshape(-1, 3)
    if "intensity" in field_names:
        inten = pc2.read_points_numpy(msg, field_names=("intensity",)).reshape(-1, 1)
    else:
        inten = np.zeros((xyz.shape[0], 1), dtype=xyz.dtype)
    return np.hstack([xyz, inten]).astype(np.float64)


def numpy_to_pointcloud2(arr: np.ndarray, header: Header) -> PointCloud2:
    """Pack an (N, 3+) array into a float32 PointCloud2 (x, y, z, intensity)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("expected an (N, >=3) point array")
    if arr.shape[1] < 4:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 4 - arr.shape[1]), np.float32)])
    fields = [
        PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, n in enumerate(("x", "y", "z", "intensity"))
    ]
    return pc2.create_cloud(header, fields, arr[:, :4])


# --------------------------------------------------------------------------- #
# Tracks
# --------------------------------------------------------------------------- #
def track_to_msg(track: Track) -> TrackedObject:
    box = track.box
    vx, vy = (float(v) for v in np.asarray(track.velocity).reshape(-1)[:2])
    return TrackedObject(
        id=int(track.track_id),
        label=track.label.value,
        score=float(track.score),
        pose=Pose(
            position=Point(x=float(box.x), y=float(box.y), z=float(box.z)),
            orientation=yaw_to_quaternion(float(box.yaw)),
        ),
        dimensions=Vector3(x=float(box.l), y=float(box.w), z=float(box.h)),
        velocity=Vector3(x=vx, y=vy, z=0.0),
        yaw_rate=float(track.yaw_rate),
        age=min(int(track.age), 255),
        hits=min(int(track.hits), 255),
    )


def tracks_to_msg(tracks: Sequence[Track], header: Header) -> TrackedObjectArray:
    return TrackedObjectArray(header=header, objects=[track_to_msg(t) for t in tracks])


def predictions_to_msg(predictions: Sequence, header: Header) -> PredictedObjectArray:
    objects = []
    for pred in predictions:
        trajs = []
        for traj in pred.trajectories:
            xy = np.asarray(traj.as_array()).reshape(-1, 2)
            z = float(pred.current_box.z)
            trajs.append(PredictedTrajectory(
                confidence=float(traj.confidence),
                mode=str(traj.mode),
                times=[float(p.t) for p in traj.points],
                waypoints=[Point(x=float(x), y=float(y), z=z) for x, y in xy],
            ))
        objects.append(PredictedObjectMsg(
            id=int(pred.track_id), label=pred.label.value, trajectories=trajs,
        ))
    return PredictedObjectArray(header=header, objects=objects)


# --------------------------------------------------------------------------- #
# RViz / Foxglove markers
# --------------------------------------------------------------------------- #
def _marker(frame_id: str, stamp: Optional[TimeMsg], ns: str, mid: int, mtype: int) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = ns
    m.id = int(mid)
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def build_marker_array(
    output: PerceptionOutput, frame_id: str, stamp: Optional[TimeMsg] = None,
) -> MarkerArray:
    """Boxes, labels, velocity arrows and predicted paths for one cycle."""
    arr = MarkerArray()
    clear = _marker(frame_id, stamp, "", 0, Marker.CUBE)
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)

    for tr in output.tracks:
        arr.markers.append(_track_cube(tr, frame_id, stamp))
        arr.markers.append(_track_label(tr, frame_id, stamp))
        vel = _velocity_arrow(tr, frame_id, stamp)
        if vel is not None:
            arr.markers.append(vel)

    for pred in output.predictions:
        line = _prediction_line(pred, frame_id, stamp)
        if line is not None:
            arr.markers.append(line)
    return arr


def _track_cube(tr: Track, frame_id: str, stamp: Optional[TimeMsg]) -> Marker:
    m = _marker(frame_id, stamp, "tracks", tr.track_id, Marker.CUBE)
    box = tr.box
    m.pose.position = Point(x=float(box.x), y=float(box.y), z=float(box.z))
    m.pose.orientation = yaw_to_quaternion(float(box.yaw))
    m.scale = Vector3(x=float(box.l), y=float(box.w), z=float(box.h))
    m.color = _color(tr.track_id, alpha=0.55)
    return m


def _track_label(tr: Track, frame_id: str, stamp: Optional[TimeMsg]) -> Marker:
    m = _marker(frame_id, stamp, "labels", tr.track_id, Marker.TEXT_VIEW_FACING)
    box = tr.box
    speed = float(np.hypot(*np.asarray(tr.velocity).reshape(-1)[:2]))
    m.pose.position = Point(x=float(box.x), y=float(box.y), z=float(box.z + box.h / 2 + 0.6))
    m.scale.z = 0.9
    m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    m.text = f"#{tr.track_id} {tr.label.value} {speed:.1f}m/s"
    return m


def _velocity_arrow(tr: Track, frame_id: str, stamp: Optional[TimeMsg]) -> Optional[Marker]:
    vx, vy = (float(v) for v in np.asarray(tr.velocity).reshape(-1)[:2])
    if math.hypot(vx, vy) < 0.3:
        return None
    m = _marker(frame_id, stamp, "velocity", tr.track_id, Marker.ARROW)
    box = tr.box
    m.points = [
        Point(x=float(box.x), y=float(box.y), z=float(box.z)),
        Point(x=float(box.x + vx), y=float(box.y + vy), z=float(box.z)),
    ]
    m.scale = Vector3(x=0.15, y=0.35, z=0.35)  # shaft dia, head dia, head len
    m.color = _color(tr.track_id, alpha=0.9)
    return m


def _prediction_line(pred, frame_id: str, stamp: Optional[TimeMsg]) -> Optional[Marker]:
    traj = pred.best
    if traj is None or not traj.points:
        return None
    m = _marker(frame_id, stamp, "prediction", pred.track_id, Marker.LINE_STRIP)
    z = float(pred.current_box.z)
    m.points = [Point(x=float(pred.current_box.x), y=float(pred.current_box.y), z=z)]
    m.points += [Point(x=float(x), y=float(y), z=z) for x, y in np.asarray(traj.as_array()).reshape(-1, 2)]
    m.scale.x = 0.2
    m.color = _color(pred.track_id, alpha=0.9)
    return m
