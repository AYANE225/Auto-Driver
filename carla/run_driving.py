#!/usr/bin/env python3
"""Drive a CARLA vehicle using this project's planner/controller, no autopilot.

Use a dedicated server. The script restores world settings and destroys only
its own actors. GT observes current actor boxes; lidar uses actual sensor data.
Automatic route selection chooses a junction-free lane for a bounded test.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import queue
from time import perf_counter

import numpy as np

from perception_core.common.geometry import transform_box
from perception_core.common.types import Box3D, Detection, Frame, ObjectClass
from perception_core.control import PathController
from perception_core.control.longitudinal import LongitudinalController
from perception_core.detection.lidar_cluster import (
    LidarClusterConfig,
    LidarClusterDetector,
)
from perception_core.detection.mock import GroundTruthDetector
from perception_core.eval.control_metrics import longitudinal_metrics
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
from record_scenario import actor_to_box, transform_to_matrix


def select_route(world, requested_spawn):
    road_map = world.get_map()
    spawns = road_map.get_spawn_points()
    indices = range(len(spawns)) if requested_spawn is None else [requested_spawn]
    for index in indices:
        if not 0 <= index < len(spawns):
            raise ValueError("spawn index is outside this map")
        wp = road_map.get_waypoint(spawns[index].location)
        chain = [wp]
        for _ in range(35):
            options = wp.next(2)
            if len(options) != 1 or options[0].is_junction:
                break
            wp = options[0]
            chain.append(wp)
        if len(chain) == 36:
            nodes = {
                i: [w.transform.location.x, -w.transform.location.y]
                for i, w in enumerate(chain)
            }
            graph = RoadGraph(nodes, [(i, i + 1) for i in range(len(chain) - 1)])
            route = ReferencePath([nodes[i] for i in graph.search(0, len(chain) - 1)])
            if np.max(np.abs(route.curvature)) < 0.08:
                return index, spawns[index], chain, route
    raise ValueError("no suitable junction-free lane; choose a different town or spawn")


def sensor_frame(buffer, frame_id):
    while True:
        value = buffer.get(timeout=10)
        if value.frame == frame_id:
            return value
        if value.frame > frame_id:
            raise RuntimeError("sensor frame is ahead of the synchronous world")


def run(args):
    import carla
    from PIL import Image

    if args.out.exists():
        raise ValueError("--out must be a new directory")
    args.out.mkdir(parents=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(30)
    world = client.get_world()
    original = world.get_settings()
    owned, sensors, collisions, lane_events = [], [], [], []
    frames, timings, offsets, speeds, control_samples = [], [], [], [], []
    minimum_gap = np.inf
    try:
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.1
        settings.substepping, settings.max_substep_delta_time, settings.max_substeps = (
            True,
            0.01,
            10,
        )
        world.apply_settings(settings)
        index, spawn, chain, route = select_route(world, args.spawn)
        blueprints = world.get_blueprint_library()
        ego = world.spawn_actor(blueprints.find("vehicle.tesla.model3"), spawn)
        owned.append(ego)
        ego.set_autopilot(False)
        physics = ego.get_physics_control()
        wheel_positions = (
            np.array([[w.position.x, w.position.y] for w in physics.wheels]) / 100
        )
        rear, front = wheel_positions[2:].mean(axis=0), wheel_positions[:2].mean(axis=0)
        forward = np.array(
            [
                math.cos(math.radians(spawn.rotation.yaw)),
                math.sin(math.radians(spawn.rotation.yaw)),
            ]
        )
        origin_to_rear = float(
            (np.array([spawn.location.x, spawn.location.y]) - rear) @ forward
        )
        bounds = ego.bounding_box
        vehicle = VehicleConfig(
            wheelbase=float(np.linalg.norm(front - rear)),
            length=2 * bounds.extent.x,
            width=2 * bounds.extent.y,
            rear_to_center=origin_to_rear + bounds.location.x,
        )
        # The CARLA Python accessor returns a copy of the wheel list.
        wheels = physics.wheels
        for wheel in wheels[:2]:
            wheel.max_steer_angle = math.degrees(vehicle.max_steer)
        physics.wheels = wheels
        physics.steering_curve = [carla.Vector2D(0, 1), carla.Vector2D(100, 1)]
        ego.apply_physics_control(physics)
        lead = None
        if args.scenario in ("obstacle", "lead_braking"):
            transform = chain[20 if args.scenario == "obstacle" else 9].transform
            transform.location.z += 0.3
            obstacle = world.spawn_actor(blueprints.find("vehicle.audi.a2"), transform)
            owned.append(obstacle)
            obstacle.apply_control(carla.VehicleControl(brake=1, hand_brake=True))
            if args.scenario == "lead_braking":
                lead = obstacle
                lead_control = LongitudinalController()
        collision_sensor = world.spawn_actor(
            blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=ego
        )
        sensors.append(collision_sensor)
        collision_sensor.listen(
            lambda event: collisions.append(
                {"frame": event.frame, "other": event.other_actor.type_id}
            )
        )
        lane_sensor = world.spawn_actor(
            blueprints.find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=ego,
        )
        sensors.append(lane_sensor)
        lane_sensor.listen(lambda event: lane_events.append(event.frame))
        lidar_queue, camera_queue = queue.Queue(), queue.Queue()
        lidar = None
        if args.detector == "lidar":
            bp = blueprints.find("sensor.lidar.ray_cast")
            for key, value in {
                "channels": "64",
                "range": "55",
                "points_per_second": "320000",
                "rotation_frequency": "10",
                "upper_fov": "10",
                "lower_fov": "-30",
            }.items():
                bp.set_attribute(key, value)
            lidar = world.spawn_actor(
                bp, carla.Transform(carla.Location(z=2.4)), attach_to=ego
            )
            sensors.append(lidar)
            lidar.listen(lidar_queue.put)
        if args.images:
            bp = blueprints.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", "800")
            bp.set_attribute("image_size_y", "450")
            camera = world.spawn_actor(
                bp, carla.Transform(carla.Location(x=0.6, z=1.7)), attach_to=ego
            )
            sensors.append(camera)
            camera.listen(camera_queue.put)
            (args.out / "images").mkdir()
        cfg = PipelineConfig()
        cfg.tracker.min_hits = 2
        cfg.predictor.horizon = 4
        cfg.lidar = LidarClusterConfig(
            voxel_size=0.2, x_range=(-15, 55), y_range=(-12, 12), ground_max_iter=30
        )
        pipeline = PerceptionPipeline(
            config=cfg,
            detector=GroundTruthDetector(seed=2026)
            if args.detector == "gt"
            else LidarClusterDetector(cfg.lidar),
        )
        planner_cfg = PlannerConfig(
            cruise_speed=5.0,
            road_half_width=min(w.lane_width for w in chain) / 2,
            lateral_offsets=(0.0,),
            collision_margin=0.15,
            stop_speed_gain=0.4,
        )
        planner, controller = (
            LocalPlanner(route, planner_cfg, vehicle),
            PathController(vehicle),
        )
        longitudinal = LongitudinalController()
        filtered_acceleration = 0.0
        ego.apply_control(carla.VehicleControl(brake=1))
        for _ in range(15):
            world.tick()
        start_time = world.get_snapshot().timestamp.elapsed_seconds
        previous_steer, reached, held = 0.0, False, 0.0
        for step in range(int(args.duration / 0.1)):
            frame_id = world.tick()
            snapshot = world.get_snapshot()
            timestamp = snapshot.timestamp.elapsed_seconds - start_time
            if lead is not None:
                if timestamp >= 6:
                    lead.apply_control(carla.VehicleControl(brake=1))
                else:
                    transform = lead.get_transform()
                    location = transform.location
                    lead_s, _ = route.project(location.x, -location.y)
                    target, _, _ = route.sample(lead_s + 4)
                    lead_yaw = -math.radians(transform.rotation.yaw)
                    dx, dy = target - [location.x, -location.y]
                    lateral = -math.sin(lead_yaw) * dx + math.cos(lead_yaw) * dy
                    steer = math.atan2(2 * 2.6 * lateral, max(1, dx * dx + dy * dy))
                    velocity = lead.get_velocity()
                    lead_speed = math.hypot(velocity.x, velocity.y)
                    lead_throttle, lead_brake = lead_control.command(
                        lead_speed, 3.5, 0, 0.1
                    )
                    lead.apply_control(
                        carla.VehicleControl(
                            throttle=lead_throttle,
                            brake=lead_brake,
                            steer=float(np.clip(-steer / 1.0, -1, 1)),
                        )
                    )
            pose = transform_to_matrix(ego.get_transform())
            yaw = math.atan2(pose[1, 0], pose[0, 0])
            rear_xy = pose[:2, 3] - origin_to_rear * np.array(
                [math.cos(yaw), math.sin(yaw)]
            )
            velocity, acceleration = ego.get_velocity(), ego.get_acceleration()
            speed = math.hypot(velocity.x, velocity.y)
            longitudinal_acceleration = acceleration.x * math.cos(
                yaw
            ) - acceleration.y * math.sin(yaw)
            filtered_acceleration += 0.25 * (
                longitudinal_acceleration - filtered_acceleration
            )
            state = VehicleState(
                *rear_xy,
                yaw,
                speed,
                previous_steer,
                filtered_acceleration,
                timestamp,
            )
            actual_boxes = []
            for actor in world.get_actors().filter("vehicle.*"):
                if actor.id != ego.id:
                    array, label = actor_to_box(actor)
                    if np.linalg.norm(array[:2] - rear_xy) < 60:
                        actual_boxes.append(
                            (Box3D.from_array(array), ObjectClass.from_str(label))
                        )
            center = rear_xy + vehicle.rear_to_center * np.array(
                [math.cos(yaw), math.sin(yaw)]
            )
            for box, _ in actual_boxes:
                minimum_gap = min(
                    minimum_gap,
                    float(
                        rectangle_separation(
                            center,
                            yaw,
                            vehicle.length,
                            vehicle.width,
                            [box.x, box.y],
                            box.yaw,
                            box.l,
                            box.w,
                        )
                    ),
                )
            if lidar is not None:
                sample = sensor_frame(lidar_queue, frame_id)
                points = (
                    np.frombuffer(sample.raw_data, dtype=np.float32)
                    .reshape(-1, 4)
                    .copy()
                )
                points[:, 1] *= -1
                sensor_pose = transform_to_matrix(sample.transform)
                # Remove ego roof/body returns in the sensor's own frame.
                points = points[
                    (np.abs(points[:, 0]) > vehicle.length / 2 + 0.2)
                    | (np.abs(points[:, 1]) > vehicle.width / 2 + 0.2)
                ]
                frame = Frame(timestamp, frame_id, lidar=points, ego_pose=sensor_pose)
            else:
                inverse = np.linalg.inv(pose)
                frame = Frame(
                    timestamp,
                    frame_id,
                    ego_pose=pose,
                    ground_truth=[
                        Detection(transform_box(box, inverse), label=label)
                        for box, label in actual_boxes
                    ],
                )
            output = pipeline.process(frame)
            begin = perf_counter()
            plan = planner.plan(state, output)
            command = controller.command(state, plan, 0.1)
            timings.append((perf_counter() - begin) * 1000)
            previous_steer = command.steering
            target_speed = min(
                planner_cfg.cruise_speed, float(np.interp(0.5, plan.times, plan.speed))
            )
            throttle, brake = longitudinal.command(
                speed,
                target_speed,
                command.acceleration,
                0.1,
                emergency=command.emergency,
                hold=plan.status == "goal_reached",
            )
            # Right-handed positive-left angles map to CARLA negative steer.
            ego.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    brake=brake,
                    steer=-command.steering / vehicle.max_steer,
                )
            )
            image_file = None
            if args.images:
                picture = sensor_frame(camera_queue, frame_id)
                image_file = f"images/{step:06d}.jpg"
                rgb = np.frombuffer(picture.raw_data, dtype=np.uint8).reshape(
                    450, 800, 4
                )[:, :, :3][:, :, ::-1]
                Image.fromarray(rgb).save(args.out / image_file, quality=82)
            s, d = route.project(state.x, state.y)
            offsets.append(abs(d))
            speeds.append(speed)
            control_samples.append(
                {
                    "time_s": timestamp,
                    "speed_mps": speed,
                    "target_speed_mps": target_speed,
                    "measured_acceleration_mps2": longitudinal_acceleration,
                    "command_acceleration_mps2": command.acceleration,
                    "throttle": throttle,
                    "brake": brake,
                    "status": plan.status,
                    "emergency": command.emergency,
                }
            )
            if lead is not None:
                lead_velocity = lead.get_velocity()
                control_samples[-1]["lead_speed_mps"] = math.hypot(
                    lead_velocity.x, lead_velocity.y
                )
                control_samples[-1]["lead_brake"] = lead.get_control().brake
            frames.append(
                {
                    "time_s": timestamp,
                    "ego": asdict(state),
                    "status": plan.status,
                    "feasible": plan.feasible,
                    "trajectory": plan.xy[::2].tolist(),
                    "candidates": plan.candidates,
                    "rejected": plan.rejected,
                    "command": {
                        **asdict(command),
                        "throttle": throttle,
                        "brake": brake,
                    },
                    "target_speed_mps": target_speed,
                    "measured_acceleration_mps2": longitudinal_acceleration,
                    "obstacles": [b.to_array().tolist() for b, _ in actual_boxes],
                    "tracks": [t.box.to_array().tolist() for t in output.tracks],
                    "route_s": s,
                    "lateral_offset_m": d,
                    "image": image_file,
                }
            )
            held = held + 0.1 if speed < 0.15 else 0.0
            if plan.status == "goal_reached":
                reached = True
                if held >= 1.0:
                    break
            if collisions:
                break
            if args.scenario != "cruise" and timestamp > 12 and held > 3:
                break
        longitudinal_result = longitudinal_metrics(
            control_samples,
            planner_cfg.cruise_speed,
            vehicle.comfortable_brake,
            vehicle.max_accel,
        )
        control_ok = (
            longitudinal_result["unexpected_cruise_stops"] == 0
            and longitudinal_result["restarts_after_stop"] == 0
            and longitudinal_result["cruise_pedal_reversals"] <= 2
            and longitudinal_result["normal_acceleration_limit_violations"] == 0
            and longitudinal_result["max_speed_mps"] <= planner_cfg.cruise_speed + 0.75
        )
        metrics = {
            "goal_reached": reached,
            "collision_events": len(collisions),
            "lane_invasion_events": len(lane_events),
            "max_lateral_offset_m": max(offsets),
            "final_speed_mps": speeds[-1],
            "elapsed_simulation_s": timestamp,
            "distance_along_route_m": s,
            "stationary_duration_s": held,
            "min_separating_axis_gap_m": None if np.isinf(minimum_gap) else minimum_gap,
            "longitudinal": longitudinal_result,
            "control_acceptance": control_ok,
            "planning_control_ms": {
                "mean": float(np.mean(timings)),
                "p95": float(np.percentile(timings, 95)),
            },
            "success": not collisions
            and not lane_events
            and control_ok
            and (reached if args.scenario == "cruise" else held > 3 and s > 10),
        }
        if lead is not None:
            metrics["lead_max_speed_mps"] = max(
                v["lead_speed_mps"] for v in control_samples
            )
            metrics["success"] &= metrics["lead_max_speed_mps"] > 2 and timestamp > 9
        root = Path(__file__).resolve().parents[1]
        result = {
            "schema_version": 1,
            "source": "carla_closed_loop",
            "source_files_sha256": {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in [
                    Path(__file__).resolve(),
                    *sorted(
                        (root / "src/perception_core/perception_core").rglob("*.py")
                    ),
                ]
            },
            "scenario": args.scenario,
            "detector": args.detector,
            "town": world.get_map().name,
            "server_version": client.get_server_version(),
            "spawn_index": index,
            "autopilot": False,
            "vehicle": asdict(vehicle),
            "planner": asdict(planner_cfg),
            "longitudinal_controller": asdict(longitudinal.cfg),
            "control_samples": control_samples,
            "control_dt_s": 0.1,
            "camera_dt_s": 0.1,
            "events": [{"time_s": 6.0, "label": "前车开始全制动"}]
            if lead is not None
            else [],
            "reference_path": route.xy[::4].tolist(),
            "road_half_width_m": planner_cfg.road_half_width,
            "metrics": metrics,
            "collisions": collisions,
            "frames": frames,
            "scope": "Junction-free single-lane route; fixed ego steering curve; current actor boxes for GT or actual LiDAR. No traffic-light recognition.",
        }
        (args.out / "report.json").write_text(
            json.dumps(
                result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        )
        print(json.dumps(metrics), flush=True)
        return metrics["success"]
    finally:
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()
        for actor in reversed(owned):
            actor.destroy()
        world.apply_settings(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--scenario", choices=["cruise", "obstacle", "lead_braking"], default="cruise"
    )
    parser.add_argument("--detector", choices=["gt", "lidar"], default="gt")
    parser.add_argument("--spawn", type=int)
    parser.add_argument("--duration", type=float, default=35)
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/carla-driving"))
    parser.add_argument("--assert-success", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive")
    if not run(args) and args.assert_success:
        raise SystemExit("CARLA driving acceptance conditions were not met")


if __name__ == "__main__":
    main()
