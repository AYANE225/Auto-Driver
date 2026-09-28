import numpy as np
import pytest

from perception_core.common.types import Box3D, PerceptionOutput, SensorCalibration, Track
from perception_core.viz.camera import CameraOverlay, clip_segment_to_near_plane


def test_near_plane_clips_crossing_segment_and_rejects_hidden_segment():
    p, q = np.array([0., 0., -1.]), np.array([2., 0., 3.])
    a, b = clip_segment_to_near_plane(p, q, 1.)
    assert np.allclose(a, [1, 0, 1]) and np.allclose(b, q)
    assert clip_segment_to_near_plane(p, np.array([1, 0, -2]), 1.) is None


def test_camera_overlay_projects_visible_tracks_without_mutating_image():
    pytest.importorskip('PIL')
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    calib = SensorCalibration(lidar_to_cam={'front': extrinsic},
                               intrinsics={'front': np.array([[100, 0, 80], [0, 100, 60], [0, 0, 1]])},
                               image_size={'front': (160, 120)})
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    overlay = CameraOverlay(calib, 'front')
    visible = PerceptionOutput(0, tracks=[Track(1, Box3D(10, 0, 0, 4, 2, 2))])
    hidden = PerceptionOutput(0, tracks=[Track(1, Box3D(-10, 0, 0, 4, 2, 2))])
    assert overlay.draw(image, visible).sum() > 0
    assert overlay.draw(image, hidden).sum() == 0
    assert image.sum() == 0
