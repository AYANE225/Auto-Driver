"""Geometry helpers: rigid transforms, box overlap (BEV / 3D IoU) and projection.

The rotated-rectangle intersection uses exact Sutherland-Hodgman polygon
clipping rather than an approximation, so BEV IoU is correct for arbitrary
headings. No Shapely dependency is required.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from perception_core.common.types import Box3D


# --------------------------------------------------------------------------- #
# Rigid-body transforms (SE(3))                                               #
# --------------------------------------------------------------------------- #
def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform without a full matrix inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) (or (N, >=3)) point array.

    Extra columns (e.g. intensity) are preserved unchanged.
    """
    pts = np.asarray(points, dtype=float)
    xyz = pts[:, :3]
    out = xyz @ T[:3, :3].T + T[:3, 3]
    if pts.shape[1] > 3:
        return np.concatenate([out, pts[:, 3:]], axis=1)
    return out


def points_in_box(points: np.ndarray, box: Box3D) -> np.ndarray:
    """Boolean mask of points (N, >=3) lying inside the oriented ``box``."""
    xyz = np.asarray(points, dtype=float)[:, :3] - box.center
    local = xyz @ rot_z(box.yaw)[:3, :3]  # inverse rotation == R^T, and R^T = rot_z(-yaw)
    return (
        (np.abs(local[:, 0]) <= box.l / 2.0)
        & (np.abs(local[:, 1]) <= box.w / 2.0)
        & (np.abs(local[:, 2]) <= box.h / 2.0)
    )


def project_to_image(points_cam: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points (N, 3) to pixels. Returns (uv (N,2), depth (N,))."""
    pts = np.asarray(points_cam, dtype=float)
    depth = pts[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uvw = pts @ K.T
    uv = uvw[:, :2] / safe[:, None]
    return uv, depth
# --------------------------------------------------------------------------- #
# Convex polygon overlap (Sutherland-Hodgman)                                 #
# --------------------------------------------------------------------------- #
def signed_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
    return abs(signed_area(poly))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if signed_area(poly) >= 0 else poly[::-1]


def _line_intersection(p1, p2, a, b):
    """Intersection of segment p1->p2 with the infinite line a->b."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return p2
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


def clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip convex ``subject`` polygon against convex ``clip`` polygon (CCW)."""
    clip = _ensure_ccw(np.asarray(clip, dtype=float))
    output = [row for row in np.asarray(subject, dtype=float)]

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        current, output = output, []
        if not current:
            break
        s = current[-1]
        for e in current:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(_line_intersection(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(_line_intersection(s, e, a, b))
            s = e
    return np.array(output) if output else np.zeros((0, 2))


def bev_iou(box_a: Box3D, box_b: Box3D) -> float:
    """Bird's-eye-view IoU between two oriented boxes."""
    inter = polygon_area(clip_polygon(box_a.bev_corners(), box_b.bev_corners()))
    area_a = box_a.l * box_a.w
    area_b = box_b.l * box_b.w
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def iou_3d(box_a: Box3D, box_b: Box3D) -> float:
    """Volumetric 3D IoU (BEV overlap times vertical overlap)."""
    inter_bev = polygon_area(clip_polygon(box_a.bev_corners(), box_b.bev_corners()))
    top = min(box_a.z + box_a.h / 2.0, box_b.z + box_b.h / 2.0)
    bot = max(box_a.z - box_a.h / 2.0, box_b.z - box_b.h / 2.0)
    inter = inter_bev * max(0.0, top - bot)
    union = box_a.volume + box_b.volume - inter
    return float(inter / union) if union > 1e-9 else 0.0


def transform_box(box: Box3D, T: np.ndarray) -> Box3D:
    """Rigidly transform an oriented box by a 4x4 SE(3) matrix."""
    center = T[:3, :3] @ box.center + T[:3, 3]
    dyaw = float(np.arctan2(T[1, 0], T[0, 0]))
    yaw = (box.yaw + dyaw + np.pi) % (2 * np.pi) - np.pi
    return Box3D(float(center[0]), float(center[1]), float(center[2]),
                 box.l, box.w, box.h, yaw)


