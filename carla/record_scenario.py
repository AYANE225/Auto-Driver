#!/usr/bin/env python3
"""Record a CARLA scenario to an on-disk dataset replayable by perception_core.

Runs under CARLA's own Python client (see ``install_carla.sh``). It spawns an ego
vehicle with a LiDAR and one or more RGB cameras, populates the town with NPC
vehicles (Traffic Manager autopilot) and pedestrians (AI walker controllers),
then steps the simulator in synchronous mode and writes one ``.npz`` per frame
plus a ``meta.json`` / ``calib.json``.

Coordinate convention (kept consistent across points, ego pose and GT boxes so
the downstream pipeline is correct):
  * CARLA is left-handed (x-forward, y-right, z-up); we convert to the
    right-handed (x-forward, y-left, z-up) frame used by ``perception_core`` by
    flipping the sign of y and negating yaw.
  * LiDAR points are stored in the sensor frame; ``ego_pose`` is the 4x4
    ``world <- lidar`` transform; ground-truth boxes are stored in the world
    frame (so they align with world-frame tracks after the pipeline lifts
    detections via ``ego_pose``). Each box carries the CARLA actor id, so
    tracking ID switches can be scored against a stable identity.
  * ``calib.json`` maps LiDAR points into each camera's *optical* frame
    (z forward, x right, y down), the pinhole convention used for projection.

    # in the CARLA client environment (see install_carla.sh)
    python record_scenario.py --scenario config/scenarios/urban.yaml --out data/urban --images
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
from typing import List, Tuple

import numpy as np
import yaml

# CARLA blueprint ``base_type`` (lower-cased) -> our taxonomy.
_CARLA_LABELS = {
    "car": "car", "truck": "truck", "bus": "bus", "van": "van",
    "bicycle": "bicycle", "motorcycle": "motorcycle", "pedestrian": "pedestrian",
}

# Camera optical frame (z fwd, x right, y down) <- camera body frame (x fwd, y left, z up).
_OPTICAL_FROM_BODY = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def load_scenario(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("town", "Town10HD_Opt")
    cfg.setdefault("dt", 0.1)
    cfg.setdefault("num_frames", 400)
    cfg.setdefault("num_vehicles", 40)
    cfg.setdefault("num_walkers", 15)
    cfg.setdefault("weather", "ClearNoon")
    cfg.setdefault("seed", 2024)
    cfg.setdefault("unload_layers", [])
    cfg.setdefault("lidar", {})
    cfg.setdefault("cameras", [])
    return cfg


# --------------------------------------------------------------------------- #
# Coordinate helpers (CARLA left-handed -> right-handed x-fwd/y-left/z-up)      #
# --------------------------------------------------------------------------- #
def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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


def actor_to_box(actor) -> Tuple[np.ndarray, str]:
    """Return (``[x,y,z,l,w,h,yaw]`` in the right-handed world frame, label).

    The bounding-box centre is offset from the actor origin (vehicles have their
    origin at ground level, walkers at the body centre), so the box location is
    pushed through the actor transform rather than assumed.
    """
    tf = actor.get_transform()
    bb = actor.bounding_box
    centre = np.array(tf.get_matrix()) @ np.array([bb.location.x, bb.location.y, bb.location.z, 1.0])
    ext = bb.extent  # half-sizes in metres
    box = np.array([
        centre[0], -centre[1], centre[2],
        2 * ext.x, 2 * ext.y, 2 * ext.z,
        _wrap(math.radians(-(tf.rotation.yaw + bb.rotation.yaw))),
    ], dtype=float)
    return box, _classify(actor)


def _classify(actor) -> str:
    if actor.type_id.startswith("walker"):
        return "pedestrian"
    attrs = actor.attributes
    base = attrs.get("base_type", "").lower()
    if base in _CARLA_LABELS:
        return _CARLA_LABELS[base]
    return "motorcycle" if int(attrs.get("number_of_wheels", 4)) == 2 else "car"


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
    for key in ("dropoff_general_rate", "noise_stddev"):  # optional realism knobs
        if key in lidar:
            bp.set_attribute(key, str(lidar[key]))
    z = lidar.get("z", 1.8)
    return bp, carla.Transform(carla.Location(x=0.0, z=z))


def build_camera_bp(world, cam: dict):
    import carla
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(cam.get("width", 1280)))
    bp.set_attribute("image_size_y", str(cam.get("height", 720)))
    bp.set_attribute("fov", str(cam.get("fov", 90.0)))
    bp.set_attribute("motion_blur_intensity", str(cam.get("motion_blur", 0.0)))
    tf = carla.Transform(
        carla.Location(x=cam.get("x", 1.5), y=cam.get("y", 0.0), z=cam.get("z", 1.6)),
        carla.Rotation(pitch=cam.get("pitch", 0.0), yaw=cam.get("yaw", 0.0)),
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


def spawn_traffic(client, world, tm, cfg: dict, spawn_points) -> List[int]:
    """Spawn NPC vehicles on autopilot via the Traffic Manager; returns actor ids."""
    import carla
    vehicle_bps = list(world.get_blueprint_library().filter("vehicle.*"))
    batch = []
    for sp in spawn_points[: cfg["num_vehicles"]]:
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "autopilot")
        batch.append(carla.command.SpawnActor(bp, sp)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
    return [res.actor_id for res in client.apply_batch_sync(batch, True) if not res.error]


def spawn_walkers(client, world, cfg: dict) -> Tuple[List[int], List[int]]:
    """Spawn pedestrians driven by AI controllers; returns (walker ids, controller ids)."""
    import carla
    n = int(cfg.get("num_walkers", 0))
    if n <= 0:
        return [], []
    bl = world.get_blueprint_library()
    walker_bps = list(bl.filter("walker.pedestrian.*"))
    batch, speeds = [], []
    for _ in range(n):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = random.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        speed = 1.4
        if bp.has_attribute("speed"):  # recommended values: [idle, walk, run]
            speed = float(bp.get_attribute("speed").recommended_values[1])
        batch.append(carla.command.SpawnActor(bp, carla.Transform(loc)))
        speeds.append(speed)
    walkers, walker_speeds = [], []
    for res, speed in zip(client.apply_batch_sync(batch, True), speeds):
        if not res.error:
            walkers.append(res.actor_id)
            walker_speeds.append(speed)

    ctrl_bp = bl.find("controller.ai.walker")
    results = client.apply_batch_sync(
        [carla.command.SpawnActor(ctrl_bp, carla.Transform(), wid) for wid in walkers], True)
    pairs = [(res.actor_id, speed) for res, speed in zip(results, walker_speeds) if not res.error]
    world.tick()  # controllers must exist on the server before they can be started
    world.set_pedestrians_cross_factor(float(cfg.get("walker_cross_factor", 0.1)))
    for ctrl_id, speed in pairs:
        ctrl = world.get_actor(ctrl_id)
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(speed)
    return walkers, [c for c, _ in pairs]


# --------------------------------------------------------------------------- #
# Main recording loop                                                           #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, help="path to a scenario YAML")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--frames", type=int, default=None, help="override num_frames")
    ap.add_argument("--images", action="store_true", help="also save camera images")
    args = ap.parse_args()

    import carla  # only available inside the CARLA client environment

    cfg = load_scenario(args.scenario)
    if args.frames is not None:
        cfg["num_frames"] = args.frames
    random.seed(cfg["seed"])
    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)  # loading a town can take a while
    world = client.load_world(cfg["town"])
    for layer in cfg["unload_layers"]:  # e.g. ParkedVehicles: static props with no GT label
        world.unload_map_layer(getattr(carla.MapLayer, layer))
    set_weather(world, cfg["weather"])

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg["dt"]
    world.apply_settings(settings)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(cfg["seed"])
    world.set_pedestrians_seed(cfg["seed"])
    if "tm_speed_difference" in cfg:  # % below the speed limit (negative = faster)
        tm.global_percentage_speed_difference(float(cfg["tm_speed_difference"]))

    actors, npc_ids, sensors, queues = [], [], {}, {}
    try:
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # ego vehicle
        bl = world.get_blueprint_library()
        ego_bp = bl.filter(cfg.get("ego", "vehicle.tesla.model3"))[0]
        ego_bp.set_attribute("role_name", "hero")
        ego = world.spawn_actor(ego_bp, spawn_points[0])
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
            lidar_to_cam = (_OPTICAL_FROM_BODY @ np.linalg.inv(transform_to_matrix(cam_tf))
                            @ transform_to_matrix(lidar_tf))
            calib["lidar_to_cam"][name] = lidar_to_cam.tolist()

        vehicles = spawn_traffic(client, world, tm, cfg, spawn_points[1:])
        walkers, controllers = spawn_walkers(client, world, cfg)
        npc_ids += vehicles + controllers + walkers
        print(f"spawned {len(vehicles)} vehicles, {len(walkers)} walkers")

        _record_loop(world, ego, sensors, queues, cam_names, cfg, args)

        with open(os.path.join(args.out, "calib.json"), "w") as fh:
            json.dump(calib, fh, indent=2)
        with open(os.path.join(args.out, "meta.json"), "w") as fh:
            json.dump({
                "scenario": os.path.basename(args.scenario), "town": cfg["town"],
                "dt": cfg["dt"], "num_frames": cfg["num_frames"], "cameras": cam_names,
                "weather": cfg["weather"], "seed": cfg["seed"],
                "num_vehicles": cfg["num_vehicles"], "num_walkers": len(walkers),
                "lidar": cfg["lidar"], "carla_version": client.get_server_version(),
            }, fh, indent=2)
        print(f"recorded {cfg['num_frames']} frames -> {args.out}")
    finally:
        for s in sensors.values():
            s.stop()
        for ctrl_id in npc_ids:
            actor = world.get_actor(ctrl_id)
            if actor is not None and actor.type_id.startswith("controller"):
                actor.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in npc_ids])
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        world.apply_settings(original)
        tm.set_synchronous_mode(False)


def _drain_until(q: "queue.Queue", frame: int, timeout: float = 10.0):
    """Return the sensor sample whose ``.frame`` matches the ticked world frame."""
    while True:
        data = q.get(timeout=timeout)
        if data.frame >= frame:
            return data


def _gather_ground_truth(world, ego, max_range: float) -> Tuple[np.ndarray, List[str], List[int]]:
    ego_loc = ego.get_location()
    boxes, labels, ids = [], [], []
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
        ids.append(actor.id)
    arr = np.array(boxes, dtype=float) if boxes else np.zeros((0, 7))
    return arr, labels, ids


def _record_loop(world, ego, sensors, queues, cam_names, cfg, args) -> None:
    warmup = int(cfg.get("warmup_frames", 30))
    max_range = float(cfg.get("lidar", {}).get("range", 100.0))
    for _ in range(warmup):  # let traffic disperse before recording
        world.tick()
        for q in queues.values():  # discard warm-up sensor samples
            while not q.empty():
                q.get_nowait()

    for i in range(cfg["num_frames"]):
        wframe = world.tick()
        lidar_data = _drain_until(queues["lidar"], wframe)
        points = lidar_to_numpy(lidar_data)
        ego_pose = transform_to_matrix(sensors["lidar"].get_transform())
        gt_boxes, gt_labels, gt_ids = _gather_ground_truth(world, ego, max_range)

        np.savez(
            os.path.join(args.out, "frames", f"{i:06d}.npz"),
            timestamp=np.float64(lidar_data.timestamp),
            points=points,
            ego_pose=ego_pose,
            gt_boxes=gt_boxes,
            gt_labels=np.array(gt_labels),
            gt_ids=np.array(gt_ids, dtype=np.int64),
        )

        for name in cam_names:  # always drain so the queues never back up
            img = _drain_until(queues[name], wframe)
            if args.images:
                from PIL import Image
                arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
                Image.fromarray(arr[:, :, [2, 1, 0]]).save(  # BGRA -> RGB
                    os.path.join(args.out, "frames", f"{i:06d}_{name}.jpg"), quality=90)

        if i % 50 == 0:
            print(f"  frame {i}/{cfg['num_frames']}  points={len(points)}  gt={len(gt_boxes)}")


if __name__ == "__main__":
    main()
