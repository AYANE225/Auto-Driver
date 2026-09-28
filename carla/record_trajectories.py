#!/usr/bin/env python3
"""Record agent trajectories (no sensors) for training motion forecasters.

Runs under CARLA's Python client. For every requested town it spawns NPC
vehicles (Traffic Manager autopilot, with randomised speed offsets and lane
changes) and pedestrians (AI walker controllers), steps the world in
synchronous **no-rendering** mode — so it needs almost no GPU — and logs every
agent's pose at each tick into one compressed ``.npz`` per town::

    tick, actor_id, cls, x, y, yaw      (right-handed x-fwd / y-left / z-up)

``cls`` is 0 = vehicle, 1 = pedestrian, 2 = two-wheeler.

    python record_trajectories.py --towns Town01_Opt Town10HD_Opt --ticks 3000 \
        --out data/trajectories
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np

from record_scenario import spawn_walkers

VEHICLE, PEDESTRIAN, TWO_WHEELER = 0, 1, 2


def _agent_class(actor) -> int:
    if actor.type_id.startswith("walker"):
        return PEDESTRIAN
    return TWO_WHEELER if int(actor.attributes.get("number_of_wheels", 4)) == 2 else VEHICLE


def spawn_vehicles(client, world, tm, n: int, rng: random.Random):
    import carla
    bps = list(world.get_blueprint_library().filter("vehicle.*"))
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)
    batch = []
    for sp in spawn_points[:n]:
        bp = rng.choice(bps)
        bp.set_attribute("role_name", "autopilot")
        batch.append(carla.command.SpawnActor(bp, sp)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
    ids = [res.actor_id for res in client.apply_batch_sync(batch, True) if not res.error]
    for vid in ids:  # behavioural diversity: speed offsets and spontaneous lane changes
        v = world.get_actor(vid)
        tm.vehicle_percentage_speed_difference(v, rng.uniform(-25.0, 30.0))
        tm.random_left_lanechange_percentage(v, rng.uniform(0.0, 10.0))
        tm.random_right_lanechange_percentage(v, rng.uniform(0.0, 10.0))
        tm.distance_to_leading_vehicle(v, rng.uniform(1.5, 4.0))
    return ids


def record_town(client, town: str, args) -> str:
    import carla
    rng = random.Random(args.seed)
    random.seed(args.seed)  # spawn_walkers() draws from the module-level RNG
    world = client.load_world(town)
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = args.dt
    settings.no_rendering_mode = True
    world.apply_settings(settings)
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    world.set_pedestrians_seed(args.seed)

    vehicles = spawn_vehicles(client, world, tm, args.vehicles, rng)
    walkers, controllers = spawn_walkers(
        client, world, {"num_walkers": args.walkers, "walker_cross_factor": 0.3})
    agents = {aid: _agent_class(world.get_actor(aid)) for aid in vehicles + walkers}
    print(f"[{town}] {len(vehicles)} vehicles, {len(walkers)} walkers", flush=True)

    rows = []
    t0 = time.time()
    try:
        for _ in range(args.warmup):
            world.tick()
        for tick in range(args.ticks):
            world.tick()
            snapshot = world.get_snapshot()
            for aid, cls in agents.items():
                snap = snapshot.find(aid)
                if snap is None:
                    continue
                tf = snap.get_transform()
                rows.append((tick, aid, cls, tf.location.x, -tf.location.y,
                             math.radians(-tf.rotation.yaw)))
    finally:
        for cid in controllers:
            ctrl = world.get_actor(cid)
            if ctrl is not None:
                ctrl.stop()
        client.apply_batch([carla.command.DestroyActor(a) for a in vehicles + controllers + walkers])
        world.tick()
        world.apply_settings(original)
        tm.set_synchronous_mode(False)

    arr = np.array(rows, dtype=np.float64)
    out = os.path.join(args.out, f"{town}.npz")
    np.savez_compressed(
        out, tick=arr[:, 0].astype(np.int32), actor_id=arr[:, 1].astype(np.int64),
        cls=arr[:, 2].astype(np.int8), x=arr[:, 3].astype(np.float32),
        y=arr[:, 4].astype(np.float32), yaw=arr[:, 5].astype(np.float32),
        dt=np.float32(args.dt), town=town,
    )
    print(f"[{town}] {len(rows)} agent-states in {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--towns", nargs="+", required=True)
    ap.add_argument("--out", default="data/trajectories")
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--vehicles", type=int, default=100)
    ap.add_argument("--walkers", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    args = ap.parse_args()

    import carla
    os.makedirs(args.out, exist_ok=True)
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    for town in args.towns:
        record_town(client, town, args)


if __name__ == "__main__":
    main()
