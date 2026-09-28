"""Steering and acceleration commands for a forward-only bicycle model."""

import numpy as np

from perception_core.planning.types import ControlCommand, VehicleConfig, VehicleState


class PathController:
    def __init__(self, vehicle=None):
        self.vehicle = vehicle or VehicleConfig()

    def command(self, state, plan, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive")
        cfg = self.vehicle
        emergency = plan.status in ("emergency_stop", "stale_input", "invalid_input")
        if (
            emergency
            or state.timestamp - plan.timestamp > 0.5
            or plan.timestamp > state.timestamp + 1e-6
        ):
            return ControlCommand(state.steering, -cfg.emergency_brake, True)
        path = plan.path_xy
        nearest = int(np.argmin(np.sum((path - [state.x, state.y]) ** 2, axis=1)))
        distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[nearest:], axis=0), axis=1))]
        lookahead = max(3.0, 0.75 * state.speed)
        target = path[min(len(path) - 1, nearest + int(np.searchsorted(distances, lookahead)))]
        delta = target - [state.x, state.y]
        lateral = -np.sin(state.yaw) * delta[0] + np.cos(state.yaw) * delta[1]
        steer = np.arctan2(2 * cfg.wheelbase * lateral, max(1.0, float(delta @ delta)))
        steer = np.clip(steer, -cfg.max_steer, cfg.max_steer)
        steer = np.clip(
            steer,
            state.steering - cfg.max_steer_rate * dt,
            state.steering + cfg.max_steer_rate * dt,
        )
        # Track the near-future speed with acceleration feed-forward.
        speed = float(np.interp(dt, plan.times, plan.speed))
        accel = float(np.interp(dt, plan.times, plan.acceleration)) + 1.5 * (speed - state.speed)
        accel = np.clip(accel, -cfg.comfortable_brake, cfg.max_accel)
        accel = np.clip(
            accel, state.acceleration - cfg.max_jerk * dt, state.acceleration + cfg.max_jerk * dt
        )
        if plan.status == "goal_reached":
            accel = -cfg.comfortable_brake
        return ControlCommand(float(steer), float(accel))


def bicycle_step(state, command, dt, vehicle=None):
    """Integrate one midpoint step; braking cannot make speed negative."""
    cfg = vehicle or VehicleConfig()
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive")
    if not np.isfinite([command.steering, command.acceleration]).all():
        raise ValueError("control must be finite")
    steering = float(np.clip(command.steering, -cfg.max_steer, cfg.max_steer))
    acceleration = float(np.clip(command.acceleration, -cfg.emergency_brake, cfg.max_accel))
    moving_dt = dt if acceleration >= 0 else min(dt, state.speed / max(1e-9, -acceleration))
    distance = max(0.0, state.speed * moving_dt + 0.5 * acceleration * moving_dt**2)
    dyaw = distance * np.tan(steering) / cfg.wheelbase
    yaw_mid = state.yaw + dyaw / 2
    speed = max(0.0, state.speed + acceleration * dt)
    return VehicleState(
        x=float(state.x + distance * np.cos(yaw_mid)),
        y=float(state.y + distance * np.sin(yaw_mid)),
        yaw=float((state.yaw + dyaw + np.pi) % (2 * np.pi) - np.pi),
        speed=speed,
        steering=steering,
        acceleration=(speed - state.speed) / dt,
        timestamp=state.timestamp + dt,
    )
