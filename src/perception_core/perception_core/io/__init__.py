"""Frame readers and a synthetic scenario generator.

``synthetic`` builds a deterministic multi-actor driving scene and rasterises it
into LiDAR point clouds plus ground-truth boxes. It lets the whole pipeline run
end-to-end (and in CI) with no CARLA, no dataset download and no GPU.
``carla_dataset`` and ``kitti`` read recorded CARLA scenarios and real KITTI raw
drives into the *same* :class:`Frame` objects.
"""

from perception_core.io.carla_dataset import CarlaDataset
from perception_core.io.synthetic import (
    Actor,
    SyntheticSceneConfig,
    actor_box,
    generate_frames,
    make_default_scene,
)

__all__ = [
    "Actor",
    "CarlaDataset",
    "SyntheticSceneConfig",
    "actor_box",
    "generate_frames",
    "make_default_scene",
]
