from perception_core.eval.control_metrics import longitudinal_metrics


def sample(speed, throttle=0, brake=0, status="cruise", acceleration=0, emergency=False):
    return {
        "speed_mps": speed,
        "throttle": throttle,
        "brake": brake,
        "status": status,
        "command_acceleration_mps2": acceleration,
        "emergency": emergency,
    }


def test_coasting_cannot_hide_pedal_reversals_or_unexpected_stops():
    samples = [
        sample(0, 0.3),
        sample(5, 0.3),
        sample(4),
        sample(2, brake=0.4),
        sample(0),
        sample(1, 0.3),
        sample(5, 0.3),
    ]
    result = longitudinal_metrics(samples, 5)
    assert result["pedal_reversals"] == result["cruise_pedal_reversals"] == 2
    assert result["unexpected_cruise_stops"] == 1
    assert result["restarts_after_stop"] == 1


def test_terminal_stop_and_emergency_are_distinguished_from_control_errors():
    result = longitudinal_metrics(
        [
            sample(5, 0.3),
            sample(4, brake=1, status="emergency_stop", acceleration=-7, emergency=True),
            sample(0, brake=1, status="goal_reached", acceleration=-3),
        ],
        5,
    )
    assert result["unexpected_cruise_stops"] == result["restarts_after_stop"] == 0
    assert result["normal_acceleration_limit_violations"] == 0
    assert (
        longitudinal_metrics([sample(5, acceleration=-7)], 5)[
            "normal_acceleration_limit_violations"
        ]
        == 1
    )
