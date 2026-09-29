"""Closed-loop tests with a bicycle plant and current-time observations only.

The simulator knows actor trajectories for stepping/evaluation. The planner
receives only the perception pipeline's current output; it never sees future
ground truth. 'gt' is a detector-free planning baseline, explicitly labelled.
"""

from dataclasses import asdict, dataclass, field
from time import perf_counter

import numpy as np

from perception_core.common.geometry import transform_box
from perception_core.common.types import Detection, Frame, ObjectClass
from perception_core.control import PathController, bicycle_step
from perception_core.detection.lidar_cluster import LidarClusterConfig, LidarClusterDetector
from perception_core.detection.mock import GroundTruthDetector
from perception_core.io.synthetic import Actor, SyntheticSceneConfig, actor_box, generate_frames
from perception_core.pipeline import PerceptionPipeline, PipelineConfig
from perception_core.planning import (
    LocalPlanner,
    PlannerConfig,
    ReferencePath,
    RoadGraph,
    VehicleConfig,
    VehicleState,
)
from perception_core.planning.collision import rectangle_separation


@dataclass
class DrivingScenario:
    name: str
    title: str
    waypoints: list
    actors: list = field(default_factory=list)
    road_half_width: float = 5.5
    duration: float = 26.0
    cruise_speed: float = 8.0
    stop_s: float = None
    green_time: float = 0.0
    dropout: tuple = None
    obstacle_appears: float = None
    expected: str = "goal"
    route_search: dict = None
    events: list = field(default_factory=list)


@dataclass
class BrakingActor(Actor):
    brake_time: float = 5.0
    deceleration: float = 3.0

    def pose_at(self, t):
        braking = np.clip(t - self.brake_time, 0, self.speed / self.deceleration)
        distance = self.speed * min(t, self.brake_time)
        distance += self.speed * braking - 0.5 * self.deceleration * braking**2
        return self.x + distance * np.cos(self.yaw), self.y + distance * np.sin(self.yaw), self.yaw


@dataclass
class LaneChangeActor(Actor):
    change_time: float = 3.0
    change_duration: float = 2.5
    target_y: float = 0.0

    def pose_at(self, t):
        u = np.clip((t - self.change_time) / self.change_duration, 0, 1)
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        lateral_speed = (self.target_y - self.y) * 30 * u**2 * (1 - u) ** 2 / self.change_duration
        return (
            self.x + self.speed * t,
            self.y + (self.target_y - self.y) * blend,
            float(np.arctan2(lateral_speed, self.speed)),
        )


def scenarios():
    straight = [[0, 0], [25, 0], [50, 0], [75, 0]]

    def car(x, y=0, speed=0):
        return Actor(x, y, 0, speed, ObjectClass.CAR, 4.5, 1.9, 1.5)

    graph = RoadGraph(
        {
            "start": [0, 0],
            "a": [22, 0],
            "b": [44, 0],
            "goal": [80, 0],
            "c": [28, 12],
            "d": [52, 12],
            "e": [66, 0],
        },
        [
            ("start", "a"),
            ("a", "b"),
            ("b", "goal"),
            ("a", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "goal"),
        ],
    )
    blocked = [("a", "b")]
    nodes = graph.search("start", "goal", blocked)
    return {
        "cruise": DrivingScenario("cruise", "巡航与到达终点", straight),
        "curve": DrivingScenario(
            "curve",
            "弯道跟踪与限速",
            [[0, 0], [20, 0], [40, 8], [60, 15], [80, 15]],
            road_half_width=2.5,
        ),
        "obstacle": DrivingScenario("obstacle", "静态障碍绕行", straight, [car(33)]),
        "following": DrivingScenario(
            "following", "跟车减速", straight, [car(18, speed=3)], road_half_width=2.0, duration=32
        ),
        "crossing": DrivingScenario(
            "crossing",
            "横穿行人让行",
            straight,
            [Actor(29, -8, np.pi / 2, 1.4, ObjectClass.PEDESTRIAN, 0.7, 0.7, 1.75)],
            road_half_width=2.0,
        ),
        "traffic_light": DrivingScenario(
            "traffic_light",
            "红灯停车与绿灯起步",
            straight,
            road_half_width=2.0,
            stop_s=30,
            green_time=12,
            duration=32,
        ),
        "blocked": DrivingScenario(
            "blocked",
            "道路阻断停车",
            straight,
            [Actor(32, 0, 0, 0, ObjectClass.TRUCK, 3, 9, 2)],
            duration=14,
            expected="stopped",
        ),
        "emergency": DrivingScenario(
            "emergency",
            "突现障碍紧急制动",
            straight,
            [Actor(28, 0, 0, 0, ObjectClass.CAR, 2, 9, 1.5)],
            obstacle_appears=4.0,
            duration=12,
            expected="stopped",
        ),
        "dropout": DrivingScenario(
            "dropout", "感知超时制动与恢复", straight, dropout=(4.0, 8.0), duration=30
        ),
        "reroute": DrivingScenario(
            "reroute",
            "封路后的 A* 路线搜索",
            [graph.nodes[n].tolist() for n in nodes],
            cruise_speed=5,
            road_half_width=3.5,
            duration=35,
            route_search={
                "nodes": {k: v.tolist() for k, v in graph.nodes.items()},
                "edges": [[a, b] for a, nexts in graph.edges.items() for b in nexts],
                "blocked_edges": blocked,
                "selected_nodes": nodes,
            },
        ),
        "lead_braking": DrivingScenario(
            "lead_braking",
            "前车行驶后急刹",
            straight,
            [BrakingActor(20, 0, 0, 4, ObjectClass.CAR, 4.5, 1.9, 1.5)],
            road_half_width=2.0,
            duration=20,
            expected="stopped",
            events=[{"time_s": 5.0, "label": "前车开始以 3 m/s² 制动"}],
        ),
        "cut_in": DrivingScenario(
            "cut_in",
            "邻道车辆切入并跟车",
            straight,
            [LaneChangeActor(18, 3.4, 0, 3.5, ObjectClass.CAR, 4.5, 1.9, 1.5)],
            road_half_width=2.0,
            duration=36,
            events=[
                {"time_s": 3.0, "label": "邻道车辆开始切入"},
                {"time_s": 5.5, "label": "切入完成，继续低速行驶"},
            ],
        ),
        "signal_crossing": DrivingScenario(
            "signal_crossing",
            "绿灯起步后连续行人横穿",
            straight,
            [
                Actor(39, -17, np.pi / 2, 1.4, ObjectClass.PEDESTRIAN, 0.7, 0.7, 1.75),
                Actor(43, 22, -np.pi / 2, 1.4, ObjectClass.PEDESTRIAN, 0.7, 0.7, 1.75),
            ],
            road_half_width=2.0,
            stop_s=30,
            green_time=10,
            duration=38,
            events=[{"time_s": 10.0, "label": "绿灯亮起，仍需对横穿行人让行"}],
        ),
    }


def _current_actors(scenario, timestamp):
    if scenario.obstacle_appears is not None and timestamp < scenario.obstacle_appears - 1e-8:
        return []
    return scenario.actors


def observation(state, actors, detector, frame_id, seed):
    """Create current-time ego-frame observations from the simulated world."""
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    pose = np.array([[c, -s, 0, state.x], [s, c, 0, state.y], [0, 0, 1, 0], [0, 0, 0, 1]])
    inverse = np.linalg.inv(pose)
    visible = [
        a
        for a in actors
        if np.linalg.norm(actor_box(a, state.timestamp).center[:2] - [state.x, state.y]) < 55
    ]
    if detector == "gt":
        detections = [
            Detection(
                transform_box(actor_box(a, state.timestamp), inverse),
                label=a.label,
                source="ground_truth",
            )
            for a in visible
        ]
        return Frame(state.timestamp, frame_id=frame_id, ego_pose=pose, ground_truth=detections)
    # The existing surface sampler has no occlusion/ray casting. Each frame is
    # freshly generated at the actual controlled ego pose, not replayed.
    local_actors = []
    for actor in visible:
        box = transform_box(actor_box(actor, state.timestamp), inverse)
        local_actors.append(Actor(box.x, box.y, box.yaw, 0, actor.label, box.l, box.w, box.h))
    frame = generate_frames(
        local_actors,
        num_frames=1,
        seed=seed + frame_id,
        config=SyntheticSceneConfig(ground_points=1200, surface_density=10),
    )[0]
    frame.timestamp, frame.frame_id, frame.ego_pose = state.timestamp, frame_id, pose
    frame.ground_truth = None  # The LiDAR detector must not access true boxes.
    return frame


def run_scenario(scenario, detector="gt", seed=2026, dt=0.1, record_every=2):
    if detector not in ("gt", "lidar") or dt <= 0 or record_every < 1:
        raise ValueError("invalid simulation settings")
    route, vehicle = ReferencePath(scenario.waypoints), VehicleConfig()
    planner_cfg = PlannerConfig(
        cruise_speed=scenario.cruise_speed, road_half_width=scenario.road_half_width
    )
    planner = LocalPlanner(route, planner_cfg, vehicle)
    controller = PathController(vehicle)
    cfg = PipelineConfig()
    cfg.tracker.min_hits = 2
    cfg.predictor.horizon = planner_cfg.horizon
    cfg.predictor.step = 0.2
    cfg.lidar = LidarClusterConfig(voxel_size=0.2, ground_max_iter=20)
    pipeline = PerceptionPipeline(
        config=cfg,
        detector=GroundTruthDetector(seed=seed)
        if detector == "gt"
        else LidarClusterDetector(cfg.lidar),
    )
    state = VehicleState(*route.xy[0], float(route.yaw[0]))
    records, timings, lateral_errors, accelerations, states_seen = [], [], [], [], set()
    collision_steps, boundary_steps, red_violations = 0, 0, 0
    min_gap, output = np.inf, None
    achieved = False
    stop_duration = 0.0
    for frame_id in range(int(scenario.duration / dt)):
        actors = _current_actors(scenario, state.timestamp)
        stale = (
            scenario.dropout is not None
            and scenario.dropout[0] <= state.timestamp < scenario.dropout[1]
        )
        if not stale or output is None:
            output = pipeline.process(observation(state, actors, detector, frame_id, seed))
        stop_s = scenario.stop_s if state.timestamp < scenario.green_time else None
        begin = perf_counter()
        plan = planner.plan(state, output, stop_s=stop_s)
        command = controller.command(state, plan, dt)
        timings.append((perf_counter() - begin) * 1000)
        states_seen.add(plan.status)
        s, d = route.project(state.x, state.y)
        lateral_errors.append(abs(d))
        accelerations.append(command.acceleration)
        if frame_id % record_every == 0 or plan.status == "goal_reached":
            records.append(
                {
                    "time_s": round(state.timestamp, 4),
                    "ego": asdict(state),
                    "status": plan.status,
                    "feasible": plan.feasible,
                    "command": asdict(command),
                    "route_s": s,
                    "lateral_offset_m": d,
                    "obstacles": [
                        actor_box(a, state.timestamp).to_array().tolist() for a in actors
                    ],
                    "tracks": [t.box.to_array().tolist() for t in output.tracks],
                    "trajectory": plan.xy[::2].tolist(),
                    "speeds": plan.speed[::2].tolist(),
                    "candidates": plan.candidates,
                    "rejected": plan.rejected,
                    "signal": "red"
                    if stop_s is not None
                    else "green"
                    if scenario.stop_s
                    else "none",
                }
            )
        if plan.status == "goal_reached":
            achieved = True
            break
        # Independently inspect actual vehicle motion at 50 Hz, including
        # actor motion between planner updates. Collisions end the run.
        for _ in range(5):
            state = bicycle_step(state, command, dt / 5, vehicle)
            center = np.array([state.x, state.y]) + vehicle.rear_to_center * np.array(
                [np.cos(state.yaw), np.sin(state.yaw)]
            )
            collided = False
            for actor in _current_actors(scenario, state.timestamp):
                box = actor_box(actor, state.timestamp)
                gap = float(
                    rectangle_separation(
                        center,
                        state.yaw,
                        vehicle.length,
                        vehicle.width,
                        [box.x, box.y],
                        box.yaw,
                        box.l,
                        box.w,
                    )
                )
                min_gap = min(min_gap, gap)
                collided |= gap <= 0
            collision_steps += int(collided)
            _, offset = route.project(*center)
            _, yaw, _ = route.sample(route.project(*center)[0])
            width = vehicle.width / 2 * abs(np.cos(state.yaw - yaw)) + vehicle.length / 2 * abs(
                np.sin(state.yaw - yaw)
            )
            boundary_steps += int(abs(offset) + width > scenario.road_half_width + 0.05)
            if scenario.stop_s is not None and state.timestamp < scenario.green_time:
                front_s = route.project(
                    *(
                        center
                        + vehicle.length / 2 * np.array([np.cos(state.yaw), np.sin(state.yaw)])
                    )
                )[0]
                red_violations += int(front_s > scenario.stop_s)
            if collided:
                break
        stop_duration = stop_duration + dt if state.speed < 0.15 else 0
        if collision_steps:
            break
    stationary_success = scenario.expected == "stopped" and state.speed < 0.15 and stop_duration > 2
    behavior_ok = {
        "obstacle": "avoidance",
        "following": "following",
        "crossing": "yielding",
        "traffic_light": "stop_line",
        "emergency": "emergency_stop",
        "dropout": "stale_input",
        "lead_braking": "following",
        "cut_in": "following",
        "signal_crossing": "stop_line",
    }
    demonstrated = scenario.name not in behavior_ok or behavior_ok[scenario.name] in states_seen
    success = bool(
        (achieved or stationary_success)
        and demonstrated
        and collision_steps == 0
        and boundary_steps == 0
        and red_violations == 0
    )
    metrics = {
        "success": success,
        "goal_reached": achieved,
        "expected_outcome": scenario.expected,
        "behavior_demonstrated": demonstrated,
        "collision_samples": collision_steps,
        "road_boundary_samples": boundary_steps,
        "red_light_violation_samples": red_violations,
        "min_separating_axis_gap_m": None if np.isinf(min_gap) else min_gap,
        "max_lateral_offset_m": max(lateral_errors),
        "final_speed_mps": state.speed,
        "goal_distance_m": float(np.linalg.norm(np.array([state.x, state.y]) - route.xy[-1])),
        "elapsed_simulation_s": state.timestamp,
        "statuses": sorted(states_seen),
        "max_abs_acceleration_mps2": max(abs(a) for a in accelerations),
        "planning_control_ms": {
            "mean": float(np.mean(timings)),
            "p95": float(np.percentile(timings, 95)),
            "max": float(np.max(timings)),
        },
    }
    return {
        "schema_version": 1,
        "scenario": scenario.name,
        "title": scenario.title,
        "source": "closed_loop_kinematic_bicycle",
        "detector": detector,
        "seed": seed,
        "dt_s": dt,
        "evaluation_dt_s": dt / 5,
        "record_every": record_every,
        "vehicle": asdict(vehicle),
        "planner": asdict(planner_cfg),
        "road_half_width_m": scenario.road_half_width,
        "reference_path": route.xy[::4].tolist() + [route.xy[-1].tolist()],
        "stop_s": scenario.stop_s,
        "route_search": scenario.route_search,
        "events": scenario.events,
        "metrics": metrics,
        "frames": records,
    }
