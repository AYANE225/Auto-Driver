"""A minimal constant-velocity Kalman filter over the ground-plane state.

State vector: ``[px, py, vx, vy]`` (position in m, velocity in m/s).
Measurement:  ``[px, py]`` (box centre projected to the ground plane).

The filter supports a variable time step so it copes with jittery sensor
timestamps, and uses a continuous white-noise-acceleration process model.
"""
from __future__ import annotations

import numpy as np


class ConstantVelocityKF:
    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 0.5) -> None:
        self.q = float(process_noise)          # acceleration spectral density
        self.r = float(measurement_noise)      # position measurement std (m)
        self.x = np.zeros(4)
        self.P = np.eye(4)
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.R = np.eye(2) * self.r**2

    def init_state(self, pos: np.ndarray, vel: np.ndarray = None) -> None:
        self.x[:2] = np.asarray(pos, dtype=float)[:2]
        self.x[2:] = np.zeros(2) if vel is None else np.asarray(vel, dtype=float)[:2]
        self.P = np.diag([self.r**2, self.r**2, 100.0, 100.0])

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        q, d = self.q, dt
        d2, d3 = d * d, d * d * d
        block = np.array([[d3 / 3.0, d2 / 2.0], [d2 / 2.0, d]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block
        Q[np.ix_([1, 3], [1, 3])] = block
        return Q

    def predict(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 1e-3)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.x.copy()

    def update(self, pos: np.ndarray) -> np.ndarray:
        z = np.asarray(pos, dtype=float)[:2]
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()
