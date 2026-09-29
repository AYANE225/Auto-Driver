"""Speed PI feedback and pedal allocation for the CARLA Model 3 at low speed.

The feed-forward terms approximate rolling/drivetrain resistance for this
simulated vehicle. They are adapter parameters, not a universal vehicle model.
Speeds are m/s, accelerations m/s², and pedal outputs are fractions in [0, 1].
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LongitudinalConfig:
    speed_kp: float = 0.25
    speed_ki: float = 0.08
    acceleration_feedforward: float = 0.10
    rolling_feedforward: float = 0.12
    speed_feedforward: float = 0.045
    integral_limit: float = 0.2
    max_throttle: float = 0.7
    throttle_rate: float = 0.8
    brake_rate: float = 1.5
    brake_enter: float = -0.06
    brake_exit: float = -0.02
    low_speed_brake_threshold: float = 2.3
    stop_speed: float = 0.05


class LongitudinalController:
    def __init__(self, config=None):
        self.cfg = config or LongitudinalConfig()
        self.integral = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.braking = False

    def command(self, speed, target_speed, acceleration, dt, *, emergency=False, hold=False):
        if not np.isfinite([speed, target_speed, acceleration, dt]).all() or dt <= 0:
            raise ValueError("longitudinal inputs must be finite and dt positive")
        if (
            emergency
            or hold
            or (speed < self.cfg.stop_speed and target_speed < self.cfg.stop_speed)
        ):
            self.integral = self.throttle = 0.0
            self.brake, self.braking = 1.0, True
            return self.throttle, self.brake
        cfg = self.cfg
        error = target_speed - speed
        feedforward = cfg.rolling_feedforward + cfg.speed_feedforward * speed
        effort = feedforward + cfg.acceleration_feedforward * acceleration + cfg.speed_kp * error
        integral = float(
            np.clip(
                self.integral + cfg.speed_ki * error * dt, -cfg.integral_limit, cfg.integral_limit
            )
        )
        # Conditional integration prevents windup at pedal saturation.
        if -1 < effort + integral < cfg.max_throttle or error * (effort + integral) < 0:
            self.integral = integral
        effort += self.integral
        if self.braking:
            self.braking = effort < cfg.brake_exit
        else:
            self.braking = effort < cfg.brake_enter
        target_throttle = 0.0 if self.braking else float(np.clip(effort, 0, cfg.max_throttle))
        target_brake = float(np.clip(-effort, 0, 1)) if self.braking else 0.0
        # PhysX brakes can lock this vehicle abruptly at walking speed. Coast
        # while still tracking a moving reference; the terminal stop and
        # emergency branches retain braking authority.
        if speed < cfg.low_speed_brake_threshold and target_speed >= cfg.stop_speed:
            target_brake = 0.0
        # Cut the opposite pedal immediately; never apply both together.
        self.throttle = (
            0.0
            if target_throttle == 0
            else float(
                np.clip(
                    target_throttle,
                    self.throttle - cfg.throttle_rate * dt,
                    self.throttle + cfg.throttle_rate * dt,
                )
            )
        )
        self.brake = (
            0.0
            if target_brake == 0
            else float(
                np.clip(
                    target_brake,
                    self.brake - cfg.brake_rate * dt,
                    self.brake + cfg.brake_rate * dt,
                )
            )
        )
        return self.throttle, self.brake
