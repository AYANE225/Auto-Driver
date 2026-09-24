import numpy as np

from perception_core.common.types import Box3D, Detection, ObjectClass
from perception_core.tracking.imm import IMMFilter
from perception_core.tracking.kalman import ConstantVelocityKF
from perception_core.tracking.mot import MultiObjectTracker, TrackerConfig


def _det(x, y, label=ObjectClass.CAR):
    return Detection(box=Box3D(x, y, 0, 4, 2, 1.5, 0.0), score=0.9, label=label)


def _ct_truth(t, v, w):
    """Ground-truth constant-turn position starting at the origin, heading +x."""
    return np.array([v / w * np.sin(w * t), v / w * (1.0 - np.cos(w * t))])


def test_imm_recovers_constant_velocity():
    imm = IMMFilter(process_noise=1.0, measurement_noise=0.1)
    imm.init_state([0.0, 0.0])
    dt, vx, vy = 0.1, 3.0, -1.0
    rng = np.random.default_rng(1)
    for k in range(1, 60):
        imm.predict(dt)
        truth = np.array([vx * k * dt, vy * k * dt])
        imm.update(truth + rng.normal(0, 0.05, 2))
    assert np.allclose(imm.velocity, [vx, vy], atol=0.4)
    assert abs(imm.yaw_rate) < 0.15  # straight motion -> near-zero turn rate


def test_imm_estimates_turn_rate_sign():
    imm = IMMFilter(process_noise=1.0, measurement_noise=0.1, turn_rate_noise=0.5)
    v, w, dt = 8.0, 0.4, 0.1
    imm.init_state(_ct_truth(0.0, v, w), [v, 0.0])
    rng = np.random.default_rng(0)
    for k in range(1, 50):
        imm.predict(dt)
        imm.update(_ct_truth(k * dt, v, w) + rng.normal(0, 0.05, 2))
    assert imm.yaw_rate > 0.1                     # recovered a positive turn rate
    assert imm.mode_probs[1] > imm.mode_probs[0]  # constant-turn mode dominates


def test_imm_forecasts_turn_better_than_constant_velocity():
    v, w, dt = 8.0, 0.4, 0.1
    imm = IMMFilter(process_noise=1.0, measurement_noise=0.1, turn_rate_noise=0.5)
    cv = ConstantVelocityKF(process_noise=1.0, measurement_noise=0.1)
    imm.init_state(_ct_truth(0.0, v, w), [v, 0.0])
    cv.init_state(_ct_truth(0.0, v, w), [v, 0.0])
    rng = np.random.default_rng(2)
    warm = 40
    for k in range(1, warm + 1):
        z = _ct_truth(k * dt, v, w) + rng.normal(0, 0.05, 2)
        imm.predict(dt)
        imm.update(z)
        cv.predict(dt)
        cv.update(z)
    horizon = 10  # ~1 s open-loop forecast, no measurements
    for _ in range(horizon):
        imm.predict(dt)
        cv.predict(dt)
    gt = _ct_truth((warm + horizon) * dt, v, w)
    imm_err = float(np.linalg.norm(imm.position - gt))
    cv_err = float(np.linalg.norm(cv.position - gt))
    assert imm_err < cv_err  # curved forecast beats the straight-line one


def test_imm_gating_distance_flags_outliers():
    imm = IMMFilter(measurement_noise=0.5)
    imm.init_state([0.0, 0.0], [1.0, 0.0])
    imm.predict(0.1)
    near = imm.gating_distance(imm.position)
    far = imm.gating_distance(imm.position + np.array([50.0, 50.0]))
    assert near < 1.0
    assert far > 100.0


def test_tracker_imm_estimates_turn_rate():
    cfg = TrackerConfig(motion_model="imm", min_hits=3, iou_threshold=0.01,
                        measurement_noise=0.2, process_noise=1.0, turn_rate_noise=0.5)
    tracker = MultiObjectTracker(cfg)
    v, w, dt = 6.0, 0.3, 0.1
    tracks = []
    for k in range(45):
        t = k * dt
        pos = _ct_truth(t, v, w)
        det = Detection(box=Box3D(pos[0], pos[1], 0, 4, 2, 1.5, w * t),
                        score=0.9, label=ObjectClass.CAR)
        tracks = tracker.update([det], t)
    assert len(tracks) == 1
    assert tracks[0].yaw_rate > 0.1


def test_gating_blocks_statistically_implausible_match():
    cfg = TrackerConfig(gating=True, gate_chi2=9.21, min_hits=1,
                        iou_threshold=0.1, measurement_noise=0.1, process_noise=0.3)
    tracker = MultiObjectTracker(cfg)
    for k in range(10):
        tracker.update([_det(0.0, 0.0)], k * 0.1)  # a still object -> tight covariance
    # a box 3 m away still overlaps in IoU (~0.14) but is implausibly far in Mahalanobis
    tracker.update([_det(3.0, 0.0)], 1.0)
    assert len(tracker.all_tracks) == 2  # original coasts, outlier spawns a new track


def test_no_gating_accepts_overlapping_match():
    cfg = TrackerConfig(gating=False, min_hits=1,
                        iou_threshold=0.1, measurement_noise=0.1, process_noise=0.3)
    tracker = MultiObjectTracker(cfg)
    for k in range(10):
        tracker.update([_det(0.0, 0.0)], k * 0.1)
    tracker.update([_det(3.0, 0.0)], 1.0)
    assert len(tracker.all_tracks) == 1  # IoU overlap is enough to associate
