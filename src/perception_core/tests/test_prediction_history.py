"""Forecast geometry and temporal isolation of the recorded-trajectory protocol."""
import importlib.util
from pathlib import Path

import numpy as np

from perception_core.prediction.history import (
    constant_acceleration, history_features, to_local, to_world,
)


def test_history_forecasts_recover_uniform_motion_in_world_coordinates():
    observed_t = np.arange(-10, 1) * 0.2
    future_t = np.arange(1, 7) * 0.5
    position = np.array([100.0, -200.0])
    velocity = np.array([3.0, -4.0])
    history = (position + observed_t[:, None] * velocity)[None]
    expected = (position + future_t[:, None] * velocity)[None]
    _, forecast, _, _ = history_features(history, 0.2, future_t)
    np.testing.assert_allclose(forecast, expected)
    np.testing.assert_allclose(constant_acceleration(history, 0.2, future_t), expected)


def test_local_features_ignore_global_translation_and_rotation():
    rng = np.random.default_rng(41)
    history = rng.normal(size=(5, 11, 2)).cumsum(axis=1)
    times = np.arange(1, 7) * 0.5
    angle = 0.8
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    shifted = history @ rotation.T + [300, -500]
    features, forecast, c, s = history_features(history, 0.2, times)
    other_features, other_forecast, _, _ = history_features(shifted, 0.2, times)
    np.testing.assert_allclose(features, other_features, atol=1e-11)
    np.testing.assert_allclose(other_forecast, forecast @ rotation.T + [300, -500])
    residual = rng.normal(size=(5, 6, 2))
    np.testing.assert_allclose(to_world(to_local(residual, c, s), c, s), residual)


def test_constant_acceleration_extrapolates_a_quadratic():
    past = np.arange(-10, 1) * 0.2
    future = np.arange(1, 7) * 0.5

    def trajectory(t):
        return np.array([20, -30]) + t[:, None] * [2, 1] + 0.5 * t[:, None]**2 * [1, -2]

    np.testing.assert_allclose(
        constant_acceleration(trajectory(past)[None], 0.2, future),
        trajectory(future)[None], atol=1e-10,
    )


def test_trajectory_windows_do_not_cross_actor_boundaries_or_missing_ticks(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location('eval_trajectories', repo / 'tools/eval_trajectories.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Actor 10 is continuous, actor 20 is missing tick 35 in every possible window.
    ticks = np.concatenate([np.arange(81), np.delete(np.arange(81), 35)])
    actors = np.concatenate([np.full(81, 10), np.full(80, 20)])
    x = ticks * 0.1 + actors * 100
    order = np.random.default_rng(5).permutation(len(ticks))
    path = tmp_path / 'town.npz'
    np.savez(path, dt=0.1, tick=ticks[order], actor_id=actors[order],
             x=x[order], y=np.zeros(len(ticks)), cls=np.zeros(len(ticks), dtype=int))
    history, future, _, actor_count = module.load_windows(path)
    assert actor_count == 1
    assert history.shape == (4, 11, 2)
    np.testing.assert_allclose(history[:, -1, 0], [1002, 1003, 1004, 1005])
    np.testing.assert_allclose(future[:, 0, 0] - history[:, -1, 0], 0.5)
    np.testing.assert_allclose(future[:, -1, 0] - history[:, -1, 0], 3.0)
