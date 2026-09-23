#!/usr/bin/env python3
"""Record a CARLA scenario to an on-disk dataset replayable by :mod:`dataset`.

Runs under CARLA's own Python client (Python 3.8). It spawns an ego vehicle with
a LiDAR and one or more RGB cameras, populates the town with NPC traffic via the
Traffic Manager, then steps the simulator in synchronous mode and writes one
``.npz`` per frame plus a ``meta.json`` / ``calib.json``.

Coordinate convention (kept consistent across points, ego pose and GT boxes so
the downstream pipeline is correct):
  * CARLA is left-handed (x-forward, y-right, z-up); we convert to the
    right-handed (x-forward, y-left, z-up) frame used by ``perception_core`` by
    flipping the sign of y and negating yaw.
  * LiDAR points are stored in the sensor frame; ``ego_pose`` is the 4x4
    ``world <- lidar`` transform; ground-truth boxes are stored in the world
    frame (so they align with world-frame tracks after the pipeline lifts
    detections via ``ego_pose``).

    # in the CARLA Python 3.8 environment (see install_carla.sh)
    python record_scenario.py --scenario config/scenarios/urban.yaml --out data/urban
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
from typing import Dict, List, Tuple

import numpy as np
import yaml

# Coarse CARLA blueprint / base_type -> our taxonomy.
_CARLA_LABELS = {
    "car": "car", "truck": "truck", "bus": "bus", "van": "van",
    "bicycle": "bicycle", "motorcycle": "motorcycle", "pedestrian": "pedestrian",
}


def load_scenario(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("town", "Town03")
    cfg.setdefault("dt", 0.1)
    cfg.setdefault("num_frames", 400)
    cfg.setdefault("num_vehicles", 40)
    cfg.setdefault("num_walkers", 15)
    cfg.setdefault("weather", "ClearNoon")
    cfg.setdefault("seed", 2024)
    cfg.setdefault("lidar", {})
    cfg.setdefault("cameras", [])
    return cfg


# --------------------------------------------------------------------------- #
# Coordinate helpers (CARLA left-handed -> right-handed x-fwd/y-left/z-up)      #
# --------------------------------------------------------------------------- #
def transform_to_matrix(transform) -> np.ndarray:
    """CARLA ``Transform`` -> 4x4 homogeneous matrix in the right-handed frame."""
    r, loc = transform.rotation, transform.location
    cy, sy = math.cos(math.radians(-r.yaw)), math.sin(math.radians(-r.yaw))
    cp, sp = math.cos(math.radians(-r.pitch)), math.sin(math.radians(-r.pitch))
    cr, sr = math.cos(math.radians(r.roll)), math.sin(math.radians(r.roll))
    # yaw(z) * pitch(y) * roll(x)
    rot = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [loc.x, -loc.y, loc.z]  # flip y: left-handed -> right-handed
    return m


def actor_to_box(actor, ego=None) -> Tuple[np.ndarray, str]:
    """Return (``[x,y,z,l,w,h,yaw]`` in world frame, label) for a CARLA actor."""
    tf = actor.get_transform()
    ext = actor.bounding_box.extent  # half-sizes in metres
    label = _classify(actor)
    box = np.array([
        tf.location.x, -tf.location.y, tf.location.z + ext.z,
        2 * ext.x, 2 * ext.y, 2 * ext.z, math.radians(-tf.rotation.yaw),
    ], dtype=float)
    return box, label


def _classify(actor) -> str:
    tid = actor.type_id
    if tid.startswith("walker"):
        return "pedestrian"
    attrs = actor.attributes
    base = attrs.get("base_type", "")
    if base in _CARLA_LABELS:
        return _CARLA_LABELS[base]
    n = int(attrs.get("number_of_wheels", 4))
    if n == 2:
        return "motorcycle"
    return "car"


def lidar_to_numpy(measurement) -> np.ndarray:
    """CARLA raycast LiDAR measurement -> (N, 4) x,y,z,intensity (right-handed)."""
    pts = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4).copy()
    pts[:, 1] = -pts[:, 1]  # flip y
    return pts


# --------------------------------------------------------------------------- #
# Sensor + actor setup                                                          #
# --------------------------------------------------------------------------- #
def build_lidar_bp(world, cfg: dict):
    import carla
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    lidar = cfg.get("lidar", {})
    bp.set_attribute("channels", str(lidar.get("channels", 64)))
    bp.set_attribute("range", str(lidar.get("range", 100.0)))
    bp.set_attribute("points_per_second", str(lidar.get("points_per_second", 1_200_000)))
    bp.set_attribute("rotation_frequency", str(int(1.0 / cfg["dt"])))
    bp.set_attribute("upper_fov", str(lidar.get("upper_fov", 10.0)))
    bp.set_attribute("lower_fov", str(lidar.get("lower_fov", -30.0)))
    z = lidar.get("z", 1.8)
    return bp, carla.Transform(carla.Location(x=0.0, z=z))


def build_camera_bp(world, cam: dict):
    import carla
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(cam.get("width", 1280)))
    bp.set_attribute("image_size_y", str(cam.get("height", 720)))
    bp.set_attribute("fov", str(cam.get("fov", 90.0)))
    tf = carla.Transform(
        carla.Location(x=cam.get("x", 1.5), y=cam.get("y", 0.0), z=cam.get("z", 1.6)),
        carla.Rotation(yaw=cam.get("yaw", 0.0)),
    )
    return bp, tf


def camera_intrinsics(width: int, height: int, fov: float) -> np.ndarray:
    f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def set_weather(world, name: str) -> None:
    import carla
    preset = getattr(carla.WeatherParameters, name, None)
    if preset is not None:
        world.set_weather(preset)


def spawn_traffic(client, world, tm, cfg: dict) -> List[int]:
    """Spawn NPC vehicles via the Traffic Manager; returns their actor ids."""
    import carla
    bl = world.get_blueprint_library()
    vehicle_bps = [b for b in bl.filter("vehicle.*")]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    ids: List[int] = []
    batch = []
    for i in range(min(cfg["num_vehicles"], len(spawn_points) - 1)):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        batch.append(carla.command.SpawnActor(bp, spawn_points[i + 1])
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
    for res in client.apply_batch_sync(batch, True):
        if not res.error:
            ids.append(res.actor_id)
    return ids


# --------------------------------------------------------------------------- #
# Main recording loop                                                           #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", required=True, help="path to a scenario YAML")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--images", action="store_true", help="also save camera images")
    args = ap.parse_args()

    import carla  # only available inside the CARLA client environment

    cfg = load_scenario(args.scenario)
    random.seed(cfg["seed"])
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.load_world(cfg["town"])
    set_weather(world, cfg["weather"])

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg["dt"]
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(cfg["seed"])

    actors, sensors, queues = [], {}, {}
    try:
        # ego vehicle
        bl = world.get_blueprint_library()
        ego_bp = bl.filter(cfg.get("ego", "vehicle.tesla.model3"))[0]
        ego_bp.set_attribute("role_name", "ego")
        spawn = random.choice(world.get_map().get_spawn_points())
        ego = world.spawn_actor(ego_bp, spawn)
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)

        # LiDAR
        lidar_bp, lidar_tf = build_lidar_bp(world, cfg)
        lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=ego)
        q_lidar: "queue.Queue" = queue.Queue()
        lidar.listen(q_lidar.put)
        sensors["lidar"], queues["lidar"] = lidar, q_lidar
        actors.append(lidar)

        # cameras + calibration
        calib = {"lidar_to_cam": {}, "intrinsics": {}, "image_size": {}}
        cam_names: List[str] = []
        for cam in cfg["cameras"]:
            name = cam["name"]
            cam_bp, cam_tf = build_camera_bp(world, cam)
            sensor = world.spawn_actor(cam_bp, cam_tf, attach_to=ego)
            q_cam: "queue.Queue" = queue.Queue()
            sensor.listen(q_cam.put)
            sensors[name], queues[name] = sensor, q_cam
            actors.append(sensor)
            cam_names.append(name)
            w, h = cam.get("width", 1280), cam.get("height", 720)
            calib["intrinsics"][name] = camera_intrinsics(w, h, cam.get("fov", 90.0)).tolist()
            calib["image_size"][name] = [w, h]
            lidar_to_cam = np.linalg.inv(transform_to_matrix(cam_tf)) @ transform_to_matrix(lidar_tf)
            calib["lidar_to_cam"][name] = lidar_to_cam.tolist()

        traffic = spawn_traffic(client, world, tm, cfg)
        actors_for_gt_ids = traffic  # NPCs; ego excluded from GT

        _record_loop(world, ego, lidar_tf, sensors, queues, cam_names, cfg, args)

        with open(os.path.join(args.out, "calib.json"), "w") as fh:
            json.dump(calib, fh, indent=2)
        with open(os.path.join(args.out, "meta.json"), "w") as fh:
            json.dump({
                "scenario": os.path.basename(args.scenario), "town": cfg["town"],
                "dt": cfg["dt"], "num_frames": cfg["num_frames"], "cameras": cam_names,
                "carla_version": getattr(carla, "__version__", "unknown"),
            }, fh, indent=2)
        print(f"recorded {cfg['num_frames']} frames -> {args.out}")
    finally:
        for s in sensors.values():
            s.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        world.apply_settings(original)
        tm.set_synchronous_mode(False)


def _drain_until(q: "queue.Queue", frame: int, timeout: float = 2.0):
    """Return the sensor sample whose ``.frame`` matches the ticked world frame."""
    while True:
        data = q.get(timeout=timeout)
        if data.frame >= frame:
            return data


def _gather_ground_truth(world, ego, max_range: float) -> Tuple[np.ndarray, List[str]]:
    ego_loc = ego.get_location()
    boxes, labels = [], []
    for actor in world.get_actors():
        tid = actor.type_id
        if not (tid.startswith("vehicle") or tid.startswith("walker")):
            continue
        if actor.id == ego.id:
            continue
        if actor.get_location().distance(ego_loc) > max_range:
            continue
        box, label = actor_to_box(actor)
        boxes.append(box)
        labels.append(label)
    arr = np.array(boxes, dtype=float) if boxes else np.zeros((0, 7))
    return arr, labels


def _record_loop(world, ego, lidar_tf, sensors, queues, cam_names, cfg, args) -> None:
    warmup = int(cfg.get("warmup_frames", 30))
    max_range = float(cfg.get("lidar", {}).get("range", 100.0))
    for _ in range(warmup):  # let traffic disperse before recording
        world.tick()

    for i in range(cfg["num_frames"]):
        wframe = world.tick()
        lidar_data = _drain_until(queues["lidar"], wframe)
        points = lidar_to_numpy(lidar_data)
        ego_pose = transform_to_matrix(sensors["lidar"].get_transform())
        gt_boxes, gt_labels = _gather_ground_truth(world, ego, max_range)

        out = {
            "timestamp": np.float64(lidar_data.timestamp),
            "points": points,
            "ego_pose": ego_pose,
            "gt_boxes": gt_boxes,
            "gt_labels": np.array(gt_labels),
        }
        np.savez(os.path.join(args.out, "frames", f"{i:06d}.npz"), **out)

        if args.images:
            from PIL import Image
            for name in cam_names:
                img = _drain_until(queues[name], wframe)
                arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
                rgb = arr[:, :, [2, 1, 0]]  # BGRA -> RGB
                Image.fromarray(rgb).save(
                    os.path.join(args.out, "frames", f"{i:06d}_{name}.jpg"), quality=90)

        if i % 50 == 0:
            print(f"  frame {i}/{cfg['num_frames']}  points={len(points)}  gt={len(gt_boxes)}")


if __name__ == "__main__":
    main()

