"""Pedal switching, low-speed stopping and emergency override regressions."""

import numpy as np
import pytest

from perception_core.control.longitudinal import LongitudinalController


def test_cruise_acceleration_noise_does_not_alternate_pedals():
    controller = LongitudinalController()
    for acceleration in np.tile([-0.15, 0.15], 30):
        throttle, brake = controller.command(5, 5, acceleration, 0.1)
        assert 0 < throttle <= 0.7
        assert brake == 0


def test_pedals_are_exclusive_and_emergency_is_immediate():
    controller = LongitudinalController()
    for speed, target, acceleration in [(0, 5, 2)] * 30 + [(5, 0, -3)] * 30:
        throttle, brake = controller.command(speed, target, acceleration, 0.1)
        assert 0 <= throttle <= 0.7 and 0 <= brake <= 1
        assert throttle * brake == 0
    assert controller.command(5, 5, 2, 0.1, emergency=True) == (0, 1)
    assert controller.integral == 0
    assert controller.command(0, 0, 0, 0.1, hold=True) == (0, 1)
    throttle, brake = controller.command(0, 2, 2, 0.1)
    assert 0 < throttle <= 0.08 + 1e-9 and brake == 0


def test_low_speed_coasting_does_not_prevent_stop_or_emergency_braking():
    controller = LongitudinalController()
    assert controller.command(1, 0.4, -3, 0.1) == (0, 0)
    assert controller.command(1, 0, -3, 0.1)[1] > 0
    assert controller.command(0.02, 0, -1, 0.1) == (0, 1)
    assert controller.command(1, 0.4, -3, 0.1, emergency=True) == (0, 1)


@pytest.mark.parametrize("dt", [0, -1, np.nan])
def test_invalid_time_step_is_rejected(dt):
    with pytest.raises(ValueError):
        LongitudinalController().command(5, 5, 0, dt)
