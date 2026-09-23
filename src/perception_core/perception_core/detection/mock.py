"""Ground-truth "detector" used for testing, CI and detector-free demos.

It replays the boxes stored on ``frame.ground_truth`` with optional localisation
noise and random dropout, which lets us exercise the tracking and prediction
stages deterministically without any learned model or GPU.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from perception_core.common.types import Box3D, Detection, Frame, ObjectClass
from perception_core.detection.base import Detector


class GroundTruthDetector(Detector):
    name = "ground_truth"

    def __init__(
        self,
        position_noise: float = 0.0,
        yaw_noise: float = 0.0,
        dropout: float = 0.0,
        false_positive_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.position_noise = float(position_noise)
        self.yaw_noise = float(yaw_noise)
        self.dropout = float(dropout)
        self.false_positive_rate = float(false_positive_rate)
        self.rng = np.random.default_rng(seed)

    def detect(self, frame: Frame) -> List[Detection]:
        gt = frame.ground_truth or []
        out: List[Detection] = []
        for det in gt:
            if self.dropout > 0 and self.rng.random() < self.dropout:
                continue
            b = det.box
            box = Box3D(
                x=b.x + self.rng.normal(0, self.position_noise),
                y=b.y + self.rng.normal(0, self.position_noise),
                z=b.z,
                l=b.l, w=b.w, h=b.h,
                yaw=b.yaw + self.rng.normal(0, self.yaw_noise),
            )
            out.append(Detection(box=box, score=det.score, label=det.label, source=self.name))

        if self.false_positive_rate > 0 and self.rng.random() < self.false_positive_rate:
            box = Box3D(
                x=self.rng.uniform(-30, 30), y=self.rng.uniform(-15, 15), z=0.0,
                l=4.0, w=2.0, h=1.6, yaw=self.rng.uniform(-np.pi, np.pi),
            )
            out.append(Detection(box=box, score=0.3, label=ObjectClass.UNKNOWN, source=self.name))
        return out
