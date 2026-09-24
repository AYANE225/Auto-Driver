"""Interacting Multiple Model (IMM) filter: constant-velocity + constant-turn.

The IMM runs a bank of filters -- a linear constant-velocity (CV) model and a
nonlinear constant-turn-rate (CT) model -- and blends their estimates through a
Markov mode-transition matrix. It follows manoeuvres (a turning vehicle) far
better than a single CV filter yet stays just as steady on straight driving, and
it is the multi-model tracker AV perception stacks reach for alongside
SORT / DeepSORT / UKF / JPDA.

State (shared by both models): ``[px, py, vx, vy, w]`` with ``w`` the yaw rate
(rad/s). Measurement: ``[px, py]``. The CV model advances position from velocity
only and keeps ``w`` near zero; the CT model rotates the velocity vector by ``w``.
It exposes the same ``init_state`` / ``predict`` / ``update`` / ``position`` /
``velocity`` interface as :class:`ConstantVelocityKF`, plus ``yaw_rate``,
``mode_probs`` and a Mahalanobis ``gating_distance`` for association gating.
"""
from __future__ import annotations

import numpy as np

_GAUSS_NORM_2D = 2.0 * np.pi  # (2*pi)^(d/2) with d = 2 measurement dims


class IMMFilter:
    def __init__(
        self,
        process_noise: float = 2.0,
        measurement_noise: float = 0.5,
        turn_rate_noise: float = 0.3,
        p_stay: float = 0.95,
    ) -> None:
        self.q = float(process_noise)          # accel spectral density (pos/vel)
        self.r = float(measurement_noise)      # position measurement std (m)
        self.qw = float(turn_rate_noise)       # yaw-rate spectral density (CT model)
        self.H = np.array([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]])
        self.R = np.eye(2) * self.r**2
        p = float(p_stay)
        self.pi = np.array([[p, 1.0 - p], [1.0 - p, p]])  # rows: from mode, cols: to mode
        self.mu = np.array([0.5, 0.5])
        self.x = np.zeros(5)
        self.P = np.eye(5)
        self.x_models = [self.x.copy(), self.x.copy()]
        self.P_models = [self.P.copy(), self.P.copy()]
        self._cbar = self.mu.copy()

    # -- setup -------------------------------------------------------------
    def init_state(self, pos, vel=None) -> None:
        pos = np.asarray(pos, dtype=float)[:2]
        vel = np.zeros(2) if vel is None else np.asarray(vel, dtype=float)[:2]
        x0 = np.array([pos[0], pos[1], vel[0], vel[1], 0.0])
        P0 = np.diag([self.r**2, self.r**2, 100.0, 100.0, 1.0])
        self.mu = np.array([0.5, 0.5])
        self.x_models = [x0.copy(), x0.copy()]
        self.P_models = [P0.copy(), P0.copy()]
        self.x, self.P = x0.copy(), P0.copy()

    # -- process models ----------------------------------------------------
    def _Q(self, dt: float, w_var: float) -> np.ndarray:
        d, d2, d3 = dt, dt * dt, dt * dt * dt
        block = self.q * np.array([[d3 / 3.0, d2 / 2.0], [d2 / 2.0, d]])
        Q = np.zeros((5, 5))
        for ip, iv in ((0, 2), (1, 3)):
            Q[ip, ip], Q[ip, iv] = block[0, 0], block[0, 1]
            Q[iv, ip], Q[iv, iv] = block[1, 0], block[1, 1]
        Q[4, 4] = w_var
        return Q

    def _cv_predict(self, x, P, dt):
        F = np.eye(5)
        F[0, 2] = F[1, 3] = dt
        return F @ x, F @ P @ F.T + self._Q(dt, 1e-6)

    def _ct_predict(self, x, P, dt):
        px, py, vx, vy, w = x
        w_var = (self.qw * dt) ** 2
        if abs(w) < 1e-4:                      # w->0 limit: straight line, but keep the
            F = np.eye(5)                      # first-order yaw-rate coupling so w stays
            F[0, 2] = F[1, 3] = dt             # observable (else it can never leave zero)
            F[0, 4] = -vy * dt * dt / 2.0
            F[1, 4] = vx * dt * dt / 2.0
            F[2, 4] = -vy * dt
            F[3, 4] = vx * dt
            return F @ x, F @ P @ F.T + self._Q(dt, w_var)
        s, c = np.sin(w * dt), np.cos(w * dt)
        xp = np.array([
            px + (vx * s - vy * (1 - c)) / w,
            py + (vx * (1 - c) + vy * s) / w,
            vx * c - vy * s,
            vx * s + vy * c,
            w,
        ])
        F = np.eye(5)
        F[0, 2], F[0, 3] = s / w, -(1 - c) / w
        F[1, 2], F[1, 3] = (1 - c) / w, s / w
        F[0, 4] = dt * (vx * c - vy * s) / w - (vx * s - vy * (1 - c)) / w**2
        F[1, 4] = dt * (vx * s + vy * c) / w - (vx * (1 - c) + vy * s) / w**2
        F[2, 2], F[2, 3], F[2, 4] = c, -s, -dt * (vx * s + vy * c)
        F[3, 2], F[3, 3], F[3, 4] = s, c, dt * (vx * c - vy * s)
        return xp, F @ P @ F.T + self._Q(dt, w_var)

    # -- IMM cycle ---------------------------------------------------------
    def _mix(self):
        self._cbar = self.pi.T @ self.mu                     # predicted mode probs
        mixed_x, mixed_P = [], []
        for j in range(2):
            wij = self.pi[:, j] * self.mu / max(self._cbar[j], 1e-12)
            xj = sum(wij[i] * self.x_models[i] for i in range(2))
            Pj = np.zeros((5, 5))
            for i in range(2):
                d = self.x_models[i] - xj
                Pj += wij[i] * (self.P_models[i] + np.outer(d, d))
            mixed_x.append(xj)
            mixed_P.append(Pj)
        return mixed_x, mixed_P

    def _combine(self):
        x = sum(self.mu[j] * self.x_models[j] for j in range(2))
        P = np.zeros((5, 5))
        for j in range(2):
            d = self.x_models[j] - x
            P += self.mu[j] * (self.P_models[j] + np.outer(d, d))
        return x, P

    def _update_model(self, x, P, z):
        nu = z - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        Sinv = np.linalg.inv(S)
        K = P @ self.H.T @ Sinv
        xu = x + K @ nu
        Pu = (np.eye(5) - K @ self.H) @ P
        det = max(float(np.linalg.det(S)), 1e-12)
        likelihood = float(np.exp(-0.5 * nu @ Sinv @ nu) / (_GAUSS_NORM_2D * np.sqrt(det)))
        return xu, Pu, likelihood

    def predict(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 1e-3)
        mx, mP = self._mix()
        x_cv, P_cv = self._cv_predict(mx[0], mP[0], dt)
        x_ct, P_ct = self._ct_predict(mx[1], mP[1], dt)
        self.x_models, self.P_models = [x_cv, x_ct], [P_cv, P_ct]
        self.x, self.P = self._combine()
        return self.x.copy()

    def update(self, pos) -> np.ndarray:
        z = np.asarray(pos, dtype=float)[:2]
        likelihoods = np.zeros(2)
        xs, ps = [], []
        for j in range(2):
            xu, pu, likelihood = self._update_model(self.x_models[j], self.P_models[j], z)
            xs.append(xu)
            ps.append(pu)
            likelihoods[j] = likelihood
        self.x_models, self.P_models = xs, ps
        weighted = self._cbar * likelihoods
        total = weighted.sum()
        self.mu = weighted / total if total > 1e-12 else np.array([0.5, 0.5])
        self.x, self.P = self._combine()
        return self.x.copy()

    # -- read-out ----------------------------------------------------------
    def gating_distance(self, pos) -> float:
        """Squared Mahalanobis distance of a measurement to the predicted state."""
        z = np.asarray(pos, dtype=float)[:2]
        nu = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(nu @ np.linalg.inv(S) @ nu)

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def yaw_rate(self) -> float:
        return float(self.x[4])

    @property
    def mode_probs(self) -> np.ndarray:
        return self.mu.copy()
