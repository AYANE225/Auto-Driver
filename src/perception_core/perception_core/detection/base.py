"""Abstract detector interface shared by every backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from perception_core.common.types import Detection, Frame


class Detector(ABC):
    """Turns a :class:`Frame` into a list of single-frame :class:`Detection`.

    Implementations must be stateless with respect to time (no tracking); the
    :class:`~perception_core.tracking.mot.MultiObjectTracker` is responsible for
    temporal association.
    """

    #: Short identifier stamped onto every produced detection's ``source`` field.
    name: str = "detector"

    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        """Return detections for a single frame."""
        raise NotImplementedError

    def __call__(self, frame: Frame) -> List[Detection]:
        return self.detect(frame)
