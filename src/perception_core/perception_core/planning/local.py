"""Receding-horizon lateral sampling with timed collision and speed checks.

Quintic lateral transitions are expressed along a supplied reference route.
Candidate speed profiles obey acceleration, braking and curvature limits.
This is a bounded sampling planner, not a complete search or an MPC solver.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .collision import rectangle_separation, sample_obstacles
from .types import MotionPlan, VehicleConfig


@dataclass
class PlannerConfig:
    cruise_speed: float = 8.0
    road_half_width: float = 5.5
    lateral_offsets: tuple = (-3.2, 0.0, 3.2)
    horizon: float = 4.0
    step: float = 0.1
    collision_margin: float = 0.4
    stop_buffer: float = 1.0
    max_observation_age: float = 0.35
    time_headway: float = 1.5
    standstill_gap: float = 2.0

    def __post_init__(self):
        values = [
            self.cruise_speed,
            self.road_half_width,
            self.horizon,
            self.step,
            self.max_observation_age,
            self.time_headway,
            self.standstill_gap,
        ]
        if not np.isfinite(values).all() or min(values) <= 0 or self.horizon < self.step:
            raise ValueError("planning time, speed and road limits must be positive")
        if (
            not np.isfinite([self.collision_margin, self.stop_buffer]).all()
            or min(self.collision_margin, self.stop_buffer) < 0
        ):
            raise ValueError("planning margins must be nonnegative")
        if not self.lateral_offsets or not np.isfinite(self.lateral_offsets).all():
            raise ValueError("at least one finite lateral offset is required")


class LocalPlanner:
    def __init__(self, route, config=None, vehicle=None):
        self.route = route
        self.cfg = config or PlannerConfig()
        self.vehicle = vehicle or VehicleConfig()
        self.previous_offset = 0.0
        self.times = np.linspace(
            0, self.cfg.horizon, int(np.ceil(self.cfg.horizon / self.cfg.step)) + 1
        )

    def _stop(self, state, status, obstacles=()):
        cfg, t = self.vehicle, self.times
        stop_t = np.minimum(t, state.speed / cfg.emergency_brake)
        distance = state.speed * stop_t - 0.5 * cfg.emergency_brake * stop_t**2
        curvature = np.tan(state.steering) / cfg.wheelbase
        yaw = state.yaw + distance * curvature
        if abs(curvature) < 1e-6:
            xy = [state.x, state.y] + distance[:, None] * [np.cos(state.yaw), np.sin(state.yaw)]
        else:
            xy = np.column_stack(
                [
                    state.x + (np.sin(yaw) - np.sin(state.yaw)) / curvature,
                    state.y - (np.cos(yaw) - np.cos(state.yaw)) / curvature,
                ]
            )
        speed = np.maximum(0, state.speed - cfg.emergency_brake * t)
        return MotionPlan(
            state.timestamp,
            t.copy(),
            xy,
            yaw,
            speed,
            np.full(len(t), -cfg.emergency_brake),
            xy,
            status,
            feasible=False,
        )

    def _collides(self, xy, yaw, speed, obstacles):
        cfg = self.vehicle
        center = xy + cfg.rear_to_center * np.column_stack([np.cos(yaw), np.sin(yaw)])
        for obstacle_xy, obstacle_yaw, length, width in obstacles:
            obstacle_speed = np.max(
                np.linalg.norm(np.diff(obstacle_xy, axis=0), axis=1) / np.diff(self.times)
            )
            # Inflate for unsampled translation between adjacent temporal samples.
            # Rotation/forecast error is covered only by the fixed margin; this
            # discrete check is not a continuous-time collision guarantee.
            margin = (
                self.cfg.collision_margin + self.cfg.step * (np.max(speed) + obstacle_speed) / 2
            )
            gap = rectangle_separation(
                center,
                yaw,
                cfg.length + 2 * margin,
                cfg.width + 2 * margin,
                obstacle_xy,
                obstacle_yaw,
                length,
                width,
            )
            if np.any(gap <= 0):
                return True
        return False

    def plan(self, state, perception, stop_s=None, speed_limit=None):
        cfg, vehicle, route = self.cfg, self.vehicle, self.route
        if (
            not np.isfinite(perception.timestamp)
            or not -1e-6 <= state.timestamp - perception.timestamp <= cfg.max_observation_age
        ):
            return self._stop(state, "stale_input")
        if (stop_s is not None and not np.isfinite(stop_s)) or (
            speed_limit is not None and (not np.isfinite(speed_limit) or speed_limit < 0)
        ):
            return self._stop(state, "invalid_input")
        try:
            obstacles = sample_obstacles(perception, self.times)
        except (ValueError, TypeError):
            return self._stop(state, "invalid_input")
        s0, d0 = route.project(state.x, state.y)
        remaining = route.length - s0
        goal_distance = np.linalg.norm(np.array([state.x, state.y]) - route.xy[-1])
        if remaining < 0.7 and goal_distance < 0.7 and state.speed < 0.2:
            result = self._stop(state, "goal_reached")
            result.feasible = True
            return result
        if abs(d0) + vehicle.width / 2 > cfg.road_half_width:
            return self._stop(state, "emergency_stop")
        cruise = min(cfg.cruise_speed, cfg.cruise_speed if speed_limit is None else speed_limit)
        path_length = min(max(35.0, state.speed * cfg.horizon + 10), max(remaining, 0.01))
        ds = np.linspace(0, path_length, max(3, int(path_length / 0.4) + 1))
        reference, reference_yaw, _ = route.sample(s0 + ds)
        heading_error = (state.yaw - reference_yaw[0] + np.pi) % (2 * np.pi) - np.pi
        if abs(heading_error) > 0.7:
            return self._stop(state, "emergency_stop")
        transition = max(18.0, 3.0 * state.speed)
        u = np.minimum(ds / transition, 1)
        slope = np.tan(heading_error) * transition
        stopping_distance = max(0, remaining - 0.5)
        stopping_for_signal = stop_s is not None and stop_s < route.length
        if stopping_for_signal:
            stopping_distance = min(
                stopping_distance,
                max(0, stop_s - s0 - vehicle.rear_to_center - vehicle.length / 2 - cfg.stop_buffer),
            )
        best, candidates, rejected = None, [], Counter()
        obstacle_velocities = [
            (xy[1:] - xy[:-1]) / np.diff(self.times)[:, None] for xy, _, _, _ in obstacles
        ]
        for offset in cfg.lateral_offsets:
            delta = offset - d0
            lateral = (
                d0
                + slope * u
                + (10 * delta - 6 * slope) * u**3
                + (-15 * delta + 8 * slope) * u**4
                + (6 * delta - 3 * slope) * u**5
            )
            path_xy = reference + lateral[:, None] * np.column_stack(
                [-np.sin(reference_yaw), np.cos(reference_yaw)]
            )
            path_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))]
            if np.any(np.diff(path_s) < 1e-7):
                rejected["geometry"] += 5
                continue
            tangent = np.gradient(path_xy, path_s, axis=0)
            path_yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
            curvature = np.gradient(path_yaw, path_s)
            # Include the full rectangular vehicle's transverse extent.
            relative_yaw = path_yaw - reference_yaw
            extent = vehicle.width / 2 * np.abs(np.cos(relative_yaw)) + (
                vehicle.length / 2 + abs(vehicle.rear_to_center)
            ) * np.abs(np.sin(relative_yaw))
            road_ok = bool(
                np.all(np.abs(lateral) + extent + cfg.collision_margin <= cfg.road_half_width)
            )
            curvature_ok = bool(
                np.max(np.abs(curvature)) <= np.tan(vehicle.max_steer) / vehicle.wheelbase
            )
            for fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
                speed = np.zeros(len(self.times))
                acceleration = np.zeros(len(self.times))
                travel = np.zeros(len(self.times))
                speed[0], acceleration[0] = state.speed, state.acceleration
                target_speed = cruise * fraction
                following = False
                for i, dt in enumerate(np.diff(self.times), 1):
                    kappa = abs(
                        float(np.interp(travel[i - 1] + max(2, speed[i - 1]), path_s, curvature))
                    )
                    distance_left = max(0, stopping_distance - travel[i - 1])
                    desired = min(
                        target_speed,
                        np.sqrt(vehicle.max_lateral_accel / max(0.001, kappa)),
                        np.sqrt(vehicle.comfortable_brake * distance_left),
                        distance_left,
                    )
                    position = np.array(
                        [np.interp(travel[i - 1], path_s, path_xy[:, k]) for k in range(2)]
                    )
                    heading = float(np.interp(travel[i - 1], path_s, path_yaw))
                    forward = np.array([np.cos(heading), np.sin(heading)])
                    left = np.array([-forward[1], forward[0]])
                    for (obstacle_xy, _, length, width), velocities in zip(
                        obstacles, obstacle_velocities
                    ):
                        relative = obstacle_xy[i - 1] - position - vehicle.rear_to_center * forward
                        lead_speed = float(velocities[i - 1] @ forward)
                        gap = float(relative @ forward) - (vehicle.length + length) / 2
                        if (
                            gap > 0
                            and lead_speed > 0.5
                            and abs(float(velocities[i - 1] @ left)) < 0.5
                            and abs(float(relative @ left))
                            < (vehicle.width + width) / 2 + cfg.collision_margin
                        ):
                            headway_speed = max(
                                0,
                                lead_speed
                                + 0.6
                                * (gap - cfg.standstill_gap - cfg.time_headway * speed[i - 1]),
                            )
                            if headway_speed < desired:
                                following = True
                                desired = headway_speed
                    accel = np.clip(
                        2.0 * (desired - speed[i - 1]),
                        -vehicle.comfortable_brake,
                        vehicle.max_accel,
                    )
                    acceleration[i] = np.clip(
                        accel,
                        acceleration[i - 1] - vehicle.max_jerk * dt,
                        acceleration[i - 1] + vehicle.max_jerk * dt,
                    )
                    speed[i] = max(0, speed[i - 1] + acceleration[i] * dt)
                    acceleration[i] = (speed[i] - speed[i - 1]) / dt
                    travel[i] = travel[i - 1] + (speed[i - 1] + speed[i]) * dt / 2
                xy = np.column_stack([np.interp(travel, path_s, path_xy[:, k]) for k in range(2)])
                yaw = np.interp(travel, path_s, path_yaw)
                reason = None
                if not road_ok:
                    reason = "road_boundary"
                elif not curvature_ok:
                    reason = "curvature"
                elif np.any(
                    speed**2 * np.abs(np.interp(travel, path_s, curvature))
                    > vehicle.max_lateral_accel + 0.05
                ):
                    reason = "lateral_acceleration"
                elif travel[-1] > stopping_distance + 0.4:
                    reason = "stop_position"
                elif self._collides(xy, yaw, speed, obstacles):
                    reason = "collision"
                score = (
                    3.5 * (cruise - float(np.mean(speed))) / max(cruise, 1)
                    + 0.025 * offset**2
                    + 0.02 * (offset - self.previous_offset) ** 2
                    + 0.5 * float(np.max(np.abs(curvature)))
                )
                candidates.append(
                    {
                        "xy": xy[::4].tolist(),
                        "offset": float(offset),
                        "target_speed": float(target_speed),
                        "feasible": reason is None,
                        "rejection": reason,
                        "score": float(score),
                    }
                )
                if reason:
                    rejected[reason] += 1
                    continue
                status = "avoidance" if abs(offset) > 0.1 else "cruise"
                if fraction < 1:
                    status = "yielding"
                elif following:
                    status = "following"
                if stopping_for_signal and stopping_distance < max(15, state.speed * cfg.horizon):
                    status = "stop_line"
                elif remaining < max(12, state.speed * cfg.horizon):
                    status = "goal_approach"
                if best is None or score < best[0]:
                    best = (
                        score,
                        MotionPlan(
                            state.timestamp,
                            self.times.copy(),
                            xy,
                            yaw,
                            speed,
                            acceleration,
                            path_xy,
                            status,
                            float(offset),
                        ),
                    )
        if best is None:
            result = self._stop(state, "emergency_stop", obstacles)
        else:
            result = best[1]
            self.previous_offset = result.target_offset
        result.candidate_count = len(candidates)
        result.rejected = dict(rejected)
        result.candidates = candidates
        return result
