"""Physical limits, failure behavior and independent closed-loop outcomes."""

import numpy as np
import pytest

from perception_core.common.types import (
    Box3D,
    Detection,
    PerceptionOutput,
    PredictedObject,
    ObjectClass,
    Trajectory,
    TrajectoryPoint,
)
from perception_core.control import PathController, bicycle_step
from perception_core.planning import (
    LocalPlanner,
    PlannerConfig,
    ReferencePath,
    RoadGraph,
    VehicleConfig,
    VehicleState,
)
from perception_core.planning.collision import rectangle_separation, sample_obstacles
from perception_core.planning.types import ControlCommand
from perception_core.simulation.driving import observation, run_scenario, scenarios


def test_astar_obeys_direction_and_blocked_edges():
    graph = RoadGraph(
        {"a": [0, 0], "b": [2, 0], "c": [0, 3], "d": [4, 0]},
        [("a", "b"), ("b", "d"), ("a", "c"), ("c", "d")],
    )
    assert graph.search("a", "d") == ["a", "b", "d"]
    assert graph.search("a", "d", [("b", "d")]) == ["a", "c", "d"]
    assert graph.search("a", "a") == ["a"]
    with pytest.raises(ValueError, match="no route"):
        graph.search("d", "a")


def test_route_projection_and_sampling_in_world_frame():
    route = ReferencePath([[10, -5], [10, 15]])
    s, d = route.project(8, 3)
    assert s == pytest.approx(8)
    assert d == pytest.approx(2)  # left of northbound route is west
    xy, yaw, curvature = route.sample(s)
    np.testing.assert_allclose(xy, [10, 3])
    assert yaw == pytest.approx(np.pi / 2)
    assert curvature == pytest.approx(0)
    with pytest.raises(ValueError):
        ReferencePath([[0, 0], [0, 0]])


def test_sat_rotated_boxes_and_touching():
    assert rectangle_separation([0, 0], 0, 4, 2, [4, 0], 0, 4, 2) == pytest.approx(0)
    assert rectangle_separation([0, 0], 0, 4, 2, [0, 2.8], np.pi / 2, 4, 2) < 0
    assert rectangle_separation([0, 0], 0, 4, 2, [0, 3.1], np.pi / 2, 4, 2) > 0


@pytest.mark.parametrize("timestamp", [-1.0, 1.0, np.nan])
def test_stale_future_and_invalid_time_apply_full_brake(timestamp):
    state = VehicleState(0, 0, 0, speed=8)
    planner = LocalPlanner(ReferencePath([[0, 0], [80, 0]]))
    plan = planner.plan(state, PerceptionOutput(timestamp))
    command = PathController().command(state, plan, 0.1)
    assert plan.status == "stale_input"
    assert not plan.feasible
    assert command.emergency and command.acceleration == -7


def test_forecast_modes_and_unconfirmed_detections_are_obstacles():
    box = Box3D(10, 0, 0, 4, 2, 1)
    prediction = PredictedObject(
        1,
        ObjectClass.CAR,
        box,
        [
            Trajectory([TrajectoryPoint(1, 12, 0)]),
            Trajectory([TrajectoryPoint(1, 10, 2)]),
        ],
    )
    output = PerceptionOutput(
        0, predictions=[prediction], detections=[Detection(Box3D(30, 5, 0, 2, 1, 1))]
    )
    samples = sample_obstacles(output, np.array([0, 0.5, 1, 2]))
    assert len(samples) == 3
    np.testing.assert_allclose(samples[0][0][-1], [14, 0])
    np.testing.assert_allclose(samples[1][0][-1], [10, 4])
    np.testing.assert_allclose(samples[2][0][-1], [30, 5])


def test_imminent_collision_is_not_reported_as_feasible():
    planner = LocalPlanner(
        ReferencePath([[0, 0], [80, 0]]), PlannerConfig(road_half_width=2, lateral_offsets=(0,))
    )
    state = VehicleState(0, 0, 0, speed=10)
    output = PerceptionOutput(0, detections=[Detection(Box3D(7, 0, 0, 2, 4, 1))])
    plan = planner.plan(state, output)
    assert plan.status == "emergency_stop" and not plan.feasible
    assert plan.rejected["collision"] > 0
    assert np.all(plan.speed >= 0)


def test_stationary_forecast_does_not_rotate_the_obstacle_box():
    box = Box3D(10, 0, 0, 4, 2, 1, yaw=np.pi)
    forecast = PredictedObject(
        1, ObjectClass.CAR, box, [Trajectory([TrajectoryPoint(1, 10.01, 0, yaw=0.7)])]
    )
    output = PerceptionOutput(0, predictions=[forecast])
    samples = sample_obstacles(output, np.array([0, 0.5, 1]))
    np.testing.assert_allclose(samples[0][1], np.pi)
    forecast.trajectories[0].points[0] = TrajectoryPoint(1, 12, 0, yaw=0)
    samples = sample_obstacles(output, np.array([0, 0.5, 1]))
    np.testing.assert_allclose(samples[0][1], np.pi)


def test_invalid_obstacle_geometry_stops():
    planner = LocalPlanner(ReferencePath([[0, 0], [80, 0]]))
    output = PerceptionOutput(0, detections=[Detection(Box3D(np.nan, 0, 0, 2, 1, 1))])
    assert planner.plan(VehicleState(0, 0, 0), output).status == "invalid_input"


def test_stopping_cannot_reverse_and_steering_sign_is_left():
    stopped = bicycle_step(VehicleState(0, 0, 0, speed=1), ControlCommand(0, -7), 1)
    assert stopped.speed == 0
    assert stopped.x == pytest.approx(1 / 14)
    turning = bicycle_step(VehicleState(0, 0, 0, speed=5), ControlCommand(0.2, 0), 0.1)
    assert turning.y > 0 and turning.yaw > 0
    with pytest.raises(ValueError):
        VehicleConfig(width=-1)


def test_approaching_goal_from_rest_does_not_deadlock():
    route = ReferencePath([[0, 0], [10, 0]])
    planner, controller = LocalPlanner(route), PathController()
    state = VehicleState(8.5, 0, 0)
    for _ in range(100):
        plan = planner.plan(state, PerceptionOutput(state.timestamp))
        if plan.status == "goal_reached":
            break
        state = bicycle_step(state, controller.command(state, plan, 0.1), 0.1)
    assert plan.status == "goal_reached"
    assert abs(state.x - 10) < 0.7 and state.speed < 0.2


def test_lidar_observation_has_no_ground_truth():
    scenario = scenarios()["obstacle"]
    frame = observation(VehicleState(10, 2, 0.2), scenario.actors, "lidar", 0, 2026)
    assert frame.ground_truth is None
    assert len(frame.lidar) > 1200
    np.testing.assert_allclose(frame.ego_pose[:2, 3], [10, 2])


def test_lateral_offset_at_route_end_is_not_goal_arrival():
    planner = LocalPlanner(ReferencePath([[0, 0], [10, 0]]))
    result = planner.plan(VehicleState(10, 3, 0), PerceptionOutput(0))
    assert result.status != "goal_reached"


@pytest.mark.parametrize("name", list(scenarios()))
def test_closed_loop_acceptance(name):
    report = run_scenario(scenarios()[name], record_every=1000)
    metrics = report["metrics"]
    assert metrics["success"], metrics
    assert metrics["collision_samples"] == 0
    assert metrics["road_boundary_samples"] == 0
    assert metrics["red_light_violation_samples"] == 0
    assert metrics["max_abs_acceleration_mps2"] <= VehicleConfig().emergency_brake + 1e-6
