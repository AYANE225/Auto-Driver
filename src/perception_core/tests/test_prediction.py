import numpy as np
import pytest

from perception_core.common.types import Box3D, Track
from perception_core.prediction.physics import MotionPredictor, PredictorConfig


def _track(vx, vy, yaw_rate=0.0, x=0.0, y=0.0):
    return Track(track_id=0, box=Box3D(x, y, 0, 4, 2, 1.5, np.arctan2(vy, vx)),
                 velocity=np.array([vx, vy]), yaw_rate=yaw_rate)


def test_cv_prediction_is_straight_line():
    pred = MotionPredictor(PredictorConfig(horizon=2.0, step=0.5, mode="cv"))
    out = pred.predict(_track(2.0, 0.0))
    traj = out.best
    assert len(traj.points) == 4
    assert traj.points[-1].x == pytest.approx(4.0)
    assert traj.points[-1].y == pytest.approx(0.0)


def test_ctrv_prediction_curves():
    pred = MotionPredictor(PredictorConfig(horizon=3.0, step=0.5, mode="ctrv"))
    out = pred.predict(_track(5.0, 0.0, yaw_rate=0.3))
    assert abs(out.best.points[-1].y) > 0.5  # trajectory bends away from the x-axis


def test_auto_mode_emits_multimodal_for_turning_object():
    pred = MotionPredictor(PredictorConfig(mode="auto"))
    out = pred.predict(_track(5.0, 0.0, yaw_rate=0.3))
    assert len(out.trajectories) == 2  # CTRV primary + CV fallback
    modes = {t.mode for t in out.trajectories}
    assert modes == {"ctrv", "constant_velocity"}


def test_auto_mode_single_hypothesis_when_straight():
    pred = MotionPredictor(PredictorConfig(mode="auto"))
    out = pred.predict(_track(5.0, 0.0, yaw_rate=0.0))
    assert len(out.trajectories) == 1
    assert out.best.mode == "constant_velocity"


def test_horizon_step_count():
    pred = MotionPredictor(PredictorConfig(horizon=4.0, step=0.4, mode="cv"))
    out = pred.predict(_track(1.0, 0.0))
    assert len(out.best.points) == 10
