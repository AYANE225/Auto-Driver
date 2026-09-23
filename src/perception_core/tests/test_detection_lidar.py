import numpy as np

from perception_core.common.geometry import bev_iou
from perception_core.common.types import Frame
from perception_core.detection.lidar_cluster import (
    LidarClusterConfig,
    LidarClusterDetector,
    oriented_box_from_points,
)
from perception_core.io.synthetic import generate_frames, make_default_scene


def test_detector_recovers_all_actors():
    frame = generate_frames(make_default_scene(), num_frames=1, seed=1)[0]
    dets = LidarClusterDetector(LidarClusterConfig()).detect(frame)
    for gt in frame.ground_truth:
        best = max((bev_iou(gt.box, d.box) for d in dets), default=0.0)
        assert best > 0.3, f"missed {gt.label} at ({gt.box.x:.1f},{gt.box.y:.1f}) best IoU={best:.2f}"


def test_detector_handles_empty_cloud():
    frame = Frame(timestamp=0.0, lidar=np.zeros((0, 4)))
    assert LidarClusterDetector().detect(frame) == []
    assert LidarClusterDetector().detect(Frame(timestamp=0.0, lidar=None)) == []


def test_oriented_box_from_axis_aligned_points():
    rng = np.random.default_rng(0)
    # dense rectangle 4 (x) by 2 (y), thin in z
    xs = rng.uniform(-2, 2, 500)
    ys = rng.uniform(-1, 1, 500)
    zs = rng.uniform(0, 0.1, 500)
    box = oriented_box_from_points(np.column_stack([xs, ys, zs]))
    assert box.l == box.l  # not NaN
    assert sorted([box.l, box.w])[1] == box.l  # longer side stored as length
    assert box.l > box.w
    assert abs(box.l - 4.0) < 0.4 and abs(box.w - 2.0) < 0.4
