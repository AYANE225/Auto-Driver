"""Backward-compatible shim: the CARLA dataset reader lives in perception_core.

    from perception_core.io.carla_dataset import CarlaDataset

It moved into the core library (next to the KITTI reader) so it is unit-tested
in CI and shared by the offline replay tool and the ROS 2 dataset player.
"""
from perception_core.io.carla_dataset import CarlaDataset

__all__ = ["CarlaDataset"]
