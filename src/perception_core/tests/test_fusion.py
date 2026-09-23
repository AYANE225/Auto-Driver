import numpy as np

from perception_core.common.types import Box3D, Detection, ObjectClass, SensorCalibration
from perception_core.fusion.late_fusion import (
    Box2D,
    Detection2D,
    LateFusion,
    project_box_to_image,
)


def _front_camera_calib():
    # LiDAR (x fwd, y left, z up) -> camera (x right, y down, z fwd)
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    return SensorCalibration(
        lidar_to_cam={"front": T}, intrinsics={"front": K}, image_size={"front": (640, 480)}
    )


def test_projection_in_front_of_camera():
    calib = _front_camera_calib()
    box = Box3D(10, 0, 0, 4, 2, 1.5, 0.0)
    box2d = project_box_to_image(box, calib.lidar_to_cam["front"], calib.intrinsics["front"], (640, 480))
    assert box2d is not None
    # box centred ahead -> projects near image centre
    assert 200 < (box2d.u1 + box2d.u2) / 2 < 440


def test_projection_behind_camera_returns_none():
    calib = _front_camera_calib()
    box = Box3D(-10, 0, 0, 4, 2, 1.5, 0.0)  # behind the ego
    assert project_box_to_image(box, calib.lidar_to_cam["front"], calib.intrinsics["front"]) is None


def test_fusion_transfers_camera_label():
    calib = _front_camera_calib()
    box = Box3D(12, 1, 0, 4, 2, 1.5, 0.0)
    box2d = project_box_to_image(box, calib.lidar_to_cam["front"], calib.intrinsics["front"], (640, 480))
    lidar_dets = [Detection(box=box, score=0.4, label=ObjectClass.UNKNOWN, source="lidar_cluster")]
    cam_dets = {"front": [Detection2D(box=box2d, score=0.95, label=ObjectClass.CAR, camera="front")]}
    fused = LateFusion().fuse(lidar_dets, cam_dets, calib)
    assert fused[0].label is ObjectClass.CAR
    assert fused[0].source == "fusion"
    assert fused[0].score > 0.4


def test_fusion_passthrough_without_calibration():
    lidar_dets = [Detection(box=Box3D(5, 0, 0, 4, 2, 1.5, 0.0), label=ObjectClass.CAR)]
    assert LateFusion().fuse(lidar_dets, {}, None) is lidar_dets


def test_box2d_iou():
    a = Box2D(0, 0, 10, 10)
    b = Box2D(5, 0, 15, 10)
    assert a.iou(b) == 1.0 / 3.0
