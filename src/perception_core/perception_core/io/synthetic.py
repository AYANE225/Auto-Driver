"""Deterministic synthetic driving-scene generator (CARLA-free)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from perception_core.common.types import Box3D, Detection, Frame, ObjectClass


@dataclass
class Actor:
    """A rigid actor following a constant-turn-rate, constant-velocity motion."""

    x: float
    y: float
    yaw: float
    speed: float
    label: ObjectClass
    l: float
    w: float
    h: float
    yaw_rate: float = 0.0

    def pose_at(self, t: float) -> Tuple[float, float, float]:
        v, w, yaw0 = self.speed, self.yaw_rate, self.yaw
        if abs(w) < 1e-6:
            return self.x + v * np.cos(yaw0) * t, self.y + v * np.sin(yaw0) * t, yaw0
        yaw = yaw0 + w * t
        x = self.x + v / w * (np.sin(yaw) - np.sin(yaw0))
        y = self.y + v / w * (-np.cos(yaw) + np.cos(yaw0))
        return x, y, yaw


@dataclass
class SyntheticSceneConfig:
    x_range: Tuple[float, float] = (-50.0, 50.0)
    y_range: Tuple[float, float] = (-30.0, 30.0)
    ground_points: int = 4000
    surface_density: float = 12.0  # target LiDAR returns per m^2 of visible surface (at close range)
    min_actor_points: int = 40
    surface_noise: float = 0.03
    ground_noise: float = 0.02
    range_dropoff: float = 90.0  # actors farther than this shed points (visibility falloff)


def actor_box(actor: Actor, t: float) -> Box3D:
    x, y, yaw = actor.pose_at(t)
    return Box3D(x=x, y=y, z=actor.h / 2.0, l=actor.l, w=actor.w, h=actor.h, yaw=yaw)


def make_default_scene() -> List[Actor]:
    """A small mixed-traffic scene: oncoming/leading cars, a turning car, a pedestrian."""
    return [
        Actor(x=15.0, y=1.8, yaw=0.0, speed=8.0, label=ObjectClass.CAR, l=4.5, w=1.9, h=1.5),
        Actor(x=40.0, y=-1.8, yaw=np.pi, speed=10.0, label=ObjectClass.CAR, l=4.6, w=1.9, h=1.5),
        Actor(x=25.0, y=-6.0, yaw=0.2, speed=6.0, label=ObjectClass.CAR, l=4.4, w=1.9, h=1.5, yaw_rate=0.05),
        Actor(x=8.0, y=5.5, yaw=-0.3, speed=1.4, label=ObjectClass.PEDESTRIAN, l=0.7, w=0.7, h=1.75),
        Actor(x=30.0, y=8.0, yaw=np.pi, speed=12.0, label=ObjectClass.TRUCK, l=8.0, w=2.6, h=3.2),
    ]
def _sample_box_surface(box: Box3D, n: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    dx, dy, dz = box.l / 2.0, box.w / 2.0, box.h / 2.0
    faces = [
        ("x", dx), ("x", -dx), ("y", dy), ("y", -dy), ("z", dz), ("z", -dz),
    ]
    areas = np.array([box.w * box.h, box.w * box.h, box.l * box.h, box.l * box.h,
                      box.l * box.w, box.l * box.w])
    pick = rng.choice(len(faces), size=n, p=areas / areas.sum())
    local = np.zeros((n, 3))
    u = rng.uniform(-1, 1, n)
    v = rng.uniform(-1, 1, n)
    for i, (axis, val) in enumerate(faces):
        m = pick == i
        if not np.any(m):
            continue
        if axis == "x":
            local[m] = np.column_stack([np.full(m.sum(), val), u[m] * dy, v[m] * dz])
        elif axis == "y":
            local[m] = np.column_stack([u[m] * dx, np.full(m.sum(), val), v[m] * dz])
        else:
            local[m] = np.column_stack([u[m] * dx, v[m] * dy, np.full(m.sum(), val)])
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    world = local @ rot.T + box.center
    world += rng.normal(0, noise, world.shape)
    intensity = rng.uniform(0.3, 1.0, (n, 1))
    return np.hstack([world, intensity])


def _make_ground(cfg: SyntheticSceneConfig, rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(*cfg.x_range, cfg.ground_points)
    y = rng.uniform(*cfg.y_range, cfg.ground_points)
    z = rng.normal(0, cfg.ground_noise, cfg.ground_points)
    intensity = rng.uniform(0.0, 0.3, cfg.ground_points)
    return np.column_stack([x, y, z, intensity])


def generate_frames(
    actors: List[Actor],
    num_frames: int = 40,
    dt: float = 0.1,
    config: SyntheticSceneConfig = None,
    seed: int = 0,
) -> List[Frame]:
    """Rasterise ``actors`` into a sequence of LiDAR frames with ground truth."""
    cfg = config or SyntheticSceneConfig()
    rng = np.random.default_rng(seed)
    frames: List[Frame] = []
    for k in range(num_frames):
        t = k * dt
        clouds = [_make_ground(cfg, rng)]
        gt: List[Detection] = []
        for actor in actors:
            box = actor_box(actor, t)
            gt.append(Detection(box=box, score=1.0, label=actor.label, source="ground_truth"))
            rng_range = float(np.hypot(box.x, box.y))
            scale = max(0.2, 1.0 - rng_range / cfg.range_dropoff)
            area = 2.0 * (box.l * box.w + box.l * box.h + box.w * box.h)
            n = max(cfg.min_actor_points, int(area * cfg.surface_density * scale))
            clouds.append(_sample_box_surface(box, n, cfg.surface_noise, rng))
        lidar = np.vstack(clouds)
        rng.shuffle(lidar)
        frames.append(
            Frame(
                timestamp=t, frame_id=k, lidar=lidar,
                ego_pose=np.eye(4), ground_truth=gt,
            )
        )
    return frames

