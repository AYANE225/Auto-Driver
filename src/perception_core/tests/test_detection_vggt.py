import numpy as np

from perception_core.common.types import Frame
from perception_core.detection.vggt import (
    VggtLidarDetector,
    pointmap_to_cloud,
)


def test_pointmap_axis_convention_cam_to_ego():
    # cam frame: x right, y down, z forward -> ego: x forward, y left, z up.
    cam = np.array([
        [0.0, 0.0, 10.0],   # straight ahead        -> ego (10, 0, 0)
        [2.0, 0.0, 10.0],   # 2 m to the right      -> ego (10, -2, 0)
        [0.0, 1.0, 10.0],   # 1 m down              -> ego (10, 0, -1)
    ])
    cloud = pointmap_to_cloud(cam, conf_percentile=0.0)
    assert cloud.shape == (3, 4)
    np.testing.assert_allclose(cloud[0, :3], [10.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(cloud[1, :3], [10.0, -2.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(cloud[2, :3], [10.0, 0.0, -1.0], atol=1e-9)


def test_pointmap_drops_nonfinite():
    pts = np.array([[0, 0, 5], [np.nan, 0, 5], [0, np.inf, 5], [1, 0, 5]], dtype=float)
    cloud = pointmap_to_cloud(pts, conf_percentile=0.0)
    assert cloud.shape[0] == 2  # only the two finite rows survive


def test_pointmap_confidence_percentile_filter():
    pts = np.tile(np.array([0.0, 0.0, 5.0]), (4, 1))
    conf = np.array([1.0, 2.0, 3.0, 4.0])
    cloud = pointmap_to_cloud(pts, conf, conf_percentile=50.0)
    assert cloud.shape[0] == 2  # keeps conf >= median (3, 4)
    assert np.all((cloud[:, 3] >= 0.0) & (cloud[:, 3] <= 1.0))  # intensity normalised


def test_pointmap_subsamples_to_max_points():
    pts = np.random.default_rng(0).normal(size=(5000, 3))
    cloud = pointmap_to_cloud(pts, conf_percentile=0.0, max_points=1000, seed=1)
    assert cloud.shape == (1000, 4)


def test_pointmap_scale_and_empty():
    pts = np.array([[0.0, 0.0, 4.0]])
    cloud = pointmap_to_cloud(pts, conf_percentile=0.0, scale=2.0)
    np.testing.assert_allclose(cloud[0, :3], [8.0, 0.0, 0.0], atol=1e-9)
    assert pointmap_to_cloud(np.zeros((0, 3))).shape == (0, 4)


def _scene_cloud(seed: int = 0) -> np.ndarray:
    """A synthetic pseudo-LiDAR cloud: flat ground + one car-sized cluster."""
    rng = np.random.default_rng(seed)
    gx = rng.uniform(0, 30, 3000)
    gy = rng.uniform(-10, 10, 3000)
    gz = rng.normal(0, 0.02, 3000)
    ground = np.column_stack([gx, gy, gz, np.full(3000, 0.2)])
    # a filled 4 x 2 x 1.6 m box centred at (12, 0)
    ox = rng.uniform(10, 14, 600)
    oy = rng.uniform(-1, 1, 600)
    oz = rng.uniform(0.0, 1.6, 600)
    obj = np.column_stack([ox, oy, oz, np.full(600, 0.9)])
    return np.vstack([ground, obj])


class _FakeBackend:
    """Stand-in for VggtDepthBackend that returns a canned cloud (no torch)."""

    def __init__(self, cloud: np.ndarray) -> None:
        self.cloud = cloud
        self.calls = 0

    def infer_cloud(self, images):
        self.calls += 1
        return self.cloud


def test_vggt_detector_composes_backend_and_clusterer():
    backend = _FakeBackend(_scene_cloud())
    det = VggtLidarDetector(backend=backend)
    frame = Frame(timestamp=0.0, images={"cam_front": np.zeros((4, 4, 3), np.uint8)})
    detections = det.detect(frame)
    assert backend.calls == 1
    assert len(detections) >= 1                      # the car cluster is recovered
    assert all(d.source == "vggt_pseudo_lidar" for d in detections)


def test_vggt_detector_without_images_returns_empty():
    backend = _FakeBackend(_scene_cloud())
    det = VggtLidarDetector(backend=backend)
    assert det.detect(Frame(timestamp=0.0)) == []   # no images -> no inference
    assert backend.calls == 0


def test_vggt_detector_empty_cloud_returns_empty():
    det = VggtLidarDetector(backend=_FakeBackend(np.zeros((0, 4))))
    frame = Frame(timestamp=0.0, images={"cam": np.zeros((4, 4, 3), np.uint8)})
    assert det.detect(frame) == []
