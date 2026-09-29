"""Metrics computed from each applied control step, without display smoothing."""

import numpy as np


def longitudinal_metrics(samples, cruise_speed, comfortable_brake=3.0, max_accel=2.0):
    previous_pedal = previous_cruise_pedal = 0
    switches = cruise_switches = stops = restarts = violations = 0
    moved = stopped = was_cruise_stopped = False
    cruise_speeds = []
    for sample in samples:
        speed = sample["speed_mps"]
        pedal = 1 if sample["throttle"] > 0.02 else -1 if sample["brake"] > 0.02 else 0
        if pedal:
            switches += int(previous_pedal != 0 and pedal != previous_pedal)
            previous_pedal = pedal
        if moved and speed < 0.15:
            stopped = True
        if stopped and speed > 0.5:
            restarts += 1
            stopped = False
        moved |= speed > 2
        cruise_stopped = moved and sample["status"] == "cruise" and speed < 0.3
        stops += int(cruise_stopped and not was_cruise_stopped)
        was_cruise_stopped = cruise_stopped
        if sample["status"] == "cruise":
            if pedal:
                cruise_switches += int(
                    previous_cruise_pedal != 0 and pedal != previous_cruise_pedal
                )
                previous_cruise_pedal = pedal
            if speed >= 0.9 * cruise_speed or cruise_speeds:
                cruise_speeds.append(speed)
        else:
            previous_cruise_pedal = 0
        if not sample["emergency"]:
            acceleration = sample["command_acceleration_mps2"]
            violations += int(not -comfortable_brake - 1e-6 <= acceleration <= max_accel + 1e-6)
    return {
        "control_samples": len(samples),
        "pedal_reversals": switches,
        "cruise_pedal_reversals": cruise_switches,
        "unexpected_cruise_stops": stops,
        "restarts_after_stop": restarts,
        "normal_acceleration_limit_violations": violations,
        "cruise_speed_rmse_mps": float(
            np.sqrt(np.mean((np.array(cruise_speeds) - cruise_speed) ** 2))
        )
        if cruise_speeds
        else None,
        "max_speed_mps": max(s["speed_mps"] for s in samples),
    }
