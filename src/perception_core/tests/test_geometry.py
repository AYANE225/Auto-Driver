import numpy as np
import pytest

from perception_core.common.geometry import (
    bev_iou,
    clip_polygon,
    invert_se3,
    iou_3d,
    make_se3,
    points_in_box,
    polygon_area,
    rot_z,
    transform_box,
    transform_points,
)
from perception_core.common.types import Box3D


def test_bev_iou_identical():
    b = Box3D(1, 2, 0, 4, 2, 1.5, 0.3)
    assert bev_iou(b, b) == pytest.approx(1.0)
    assert iou_3d(b, b) == pytest.approx(1.0)


def test_bev_iou_disjoint():
    a = Box3D(0, 0, 0, 4, 2, 1.5, 0.0)
    b = Box3D(100, 0, 0, 4, 2, 1.5, 0.0)
    assert bev_iou(a, b) == 0.0


def test_bev_iou_known_overlap():
    a = Box3D(0, 0, 0, 4, 2, 1.5, 0.0)
    b = Box3D(2, 0, 0, 4, 2, 1.5, 0.0)  # 2 m overlap along length
    assert bev_iou(a, b) == pytest.approx(4.0 / 12.0, abs=1e-6)


def test_bev_iou_rotation_invariant():
    a = Box3D(0, 0, 0, 4, 2, 1.5, 0.0)
    # rotating both boxes by the same angle must not change IoU
    for yaw in (0.0, 0.5, 1.2, -2.0):
        ra = Box3D(0, 0, 0, 4, 2, 1.5, yaw)
        rb = Box3D(2 * np.cos(yaw), 2 * np.sin(yaw), 0, 4, 2, 1.5, yaw)
        b = Box3D(2, 0, 0, 4, 2, 1.5, 0.0)
        assert bev_iou(ra, rb) == pytest.approx(bev_iou(a, b), abs=1e-6)


def test_polygon_area_square():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    assert polygon_area(sq) == pytest.approx(4.0)


def test_clip_polygon_contained():
    big = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=float)
    small = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    assert polygon_area(clip_polygon(small, big)) == pytest.approx(4.0)


def test_points_in_box():
    box = Box3D(0, 0, 0, 4, 2, 2, 0.0)
    pts = np.array([[0, 0, 0], [1.9, 0.9, 0.9], [3, 0, 0], [0, 0, 1.5]], dtype=float)
    assert points_in_box(pts, box).tolist() == [True, True, False, False]


def test_se3_inverse_roundtrip():
    T = make_se3(rot_z(0.7), [1, 2, 3])
    pts = np.random.default_rng(0).normal(size=(10, 3))
    back = transform_points(transform_points(pts, T), invert_se3(T))
    assert np.allclose(back, pts, atol=1e-9)


def test_transform_points_preserves_intensity():
    T = make_se3(rot_z(0.3), [1, 0, 0])
    pts = np.array([[1.0, 0.0, 0.0, 0.5]])
    out = transform_points(pts, T)
    assert out[0, 3] == pytest.approx(0.5)


def test_transform_box_translation_and_yaw():
    box = Box3D(1, 0, 0, 4, 2, 1.5, 0.0)
    T = make_se3(rot_z(np.pi / 2), [0, 0, 0])
    out = transform_box(box, T)
    assert out.x == pytest.approx(0.0, abs=1e-9)
    assert out.y == pytest.approx(1.0, abs=1e-9)
    assert out.yaw == pytest.approx(np.pi / 2, abs=1e-9)
