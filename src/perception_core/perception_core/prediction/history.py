"""Forecasting from observed XY histories in metres, without future inputs.

The local frame follows recent displacement (world axes for stationary actors).
Ridge features encode relative history; targets are residuals from constant
velocity. Model fitting and town splits live in tools/eval_trajectories.py.
"""
from __future__ import annotations

import numpy as np


def history_features(history: np.ndarray, dt: float, times: np.ndarray):
    """Return local XY features, CV forecast and local-to-world rotation components."""
    if dt <= 0 or history.ndim != 3 or history.shape[1] < 3 or history.shape[2] != 2:
        raise ValueError("expected (batch, history >= 3, 2) positions and positive dt")
    lag = min(history.shape[1] - 1, max(1, int(round(0.5 / dt))))
    velocity = (history[:, -1] - history[:, -1 - lag]) / (lag * dt)
    yaw = np.arctan2(velocity[:, 1], velocity[:, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    relative = history - history[:, -1:]
    local = np.stack([relative[..., 0] * c[:, None] + relative[..., 1] * s[:, None],
                      -relative[..., 0] * s[:, None] + relative[..., 1] * c[:, None]], -1)
    cv = history[:, -1:] + velocity[:, None] * times[None, :, None]
    return local.reshape(len(history), -1), cv, c, s


def to_local(displacement, c, s):
    return np.stack([displacement[..., 0] * c[:, None] + displacement[..., 1] * s[:, None],
                     -displacement[..., 0] * s[:, None] + displacement[..., 1] * c[:, None]], -1)


def to_world(displacement, c, s):
    return np.stack([displacement[..., 0] * c[:, None] - displacement[..., 1] * s[:, None],
                     displacement[..., 0] * s[:, None] + displacement[..., 1] * c[:, None]], -1)


def constant_acceleration(history, dt, times):
    """Fit a quadratic to the last second; anchor forecasts at the observed position."""
    count = min(history.shape[1], int(round(1 / dt)) + 1)
    t = np.arange(1 - count, 1) * dt
    design = np.column_stack([np.ones(count), t, 0.5 * t**2])
    coeff = np.einsum('ij,bjk->bik', np.linalg.pinv(design), history[:, -count:])
    return (history[:, -1:] + coeff[:, 1:2] * times[None, :, None]
            + 0.5 * coeff[:, 2:3] * times[None, :, None]**2)
