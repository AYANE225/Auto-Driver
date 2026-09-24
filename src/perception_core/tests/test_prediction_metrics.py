import numpy as np
import pytest

from perception_core.common.types import Box3D, Track, TrajectoryPoint, Trajectory
from perception_core.eval.prediction_metrics import PredictionSample, evaluate_prediction
from perception_core.prediction.physics import MotionPredictor, PredictorConfig


def _straight_traj(vx, dt=0.5, n=6, confidence=1.0):
    pts = [TrajectoryPoint(t=(i + 1) * dt, x=vx * (i + 1) * dt, y=0.0) for i in range(n)]
    return Trajectory(points=pts, confidence=confidence, mode="constant_velocity")


def test_ade_fde_zero_for_perfect_prediction():
    traj = _straight_traj(2.0)
    m = evaluate_prediction([PredictionSample([traj], traj.as_array())])
    assert m.ade == 0.0
    assert m.fde == 0.0
    assert m.miss_rate == 0.0
    assert m.samples == 1
    assert m.horizon_steps == 6


def test_ade_fde_constant_offset():
    traj = _straight_traj(2.0, n=4)
    gt = traj.as_array() + np.array([0.0, 0.5])  # 0.5 m lateral error at every step
    m = evaluate_prediction([PredictionSample([traj], gt)])
    assert m.ade == pytest.approx(0.5)
    assert m.fde == pytest.approx(0.5)


def test_fde_uses_final_step_and_miss_rate():
    traj = _straight_traj(1.0, n=3)
    gt = traj.as_array().copy()
    gt[-1] += np.array([3.0, 4.0])  # only the final waypoint is 5 m off
    m = evaluate_prediction([PredictionSample([traj], gt)], miss_threshold=2.0)
    assert m.fde == pytest.approx(5.0)
    assert m.ade == pytest.approx(5.0 / 3.0, abs=1e-3)  # metrics are rounded to 4 dp
    assert m.miss_rate == 1.0  # 5 m > 2 m threshold


def test_min_ade_picks_best_of_multiple_modes():
    good = _straight_traj(2.0, n=4, confidence=0.3)
    bad = _straight_traj(2.0, n=4, confidence=0.7)  # higher confidence -> "best" mode
    for p in bad.points:
        p.y += 5.0
    m = evaluate_prediction([PredictionSample([bad, good], good.as_array())])
    assert m.ade == pytest.approx(5.0)      # best (most confident) mode is the wrong one
    assert m.min_ade == pytest.approx(0.0)  # min over modes recovers the good one
    assert m.modes == pytest.approx(2.0)


def test_empty_samples_return_zeros():
    m = evaluate_prediction([])
    assert m.samples == 0
    assert m.ade == 0.0
    assert m.fde == 0.0


def test_predictor_scores_well_on_constant_velocity():
    pred = MotionPredictor(PredictorConfig(horizon=2.0, step=0.5, mode="cv"))
    track = Track(track_id=0, box=Box3D(0, 0, 0, 4, 2, 1.5, 0.0), velocity=np.array([3.0, 0.0]))
    obj = pred.predict(track)
    gt = np.array([[3.0 * t, 0.0] for t in (0.5, 1.0, 1.5, 2.0)])
    m = evaluate_prediction([PredictionSample(obj.trajectories, gt)])
    assert m.fde < 1e-6
    assert m.ade < 1e-6
