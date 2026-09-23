"""Lightweight, dependency-free evaluation for detection and tracking.

Metrics follow the classic CLEAR-MOT definitions (MOTA / MOTP / ID switches)
plus per-frame precision and recall, matched by bird's-eye-view IoU. Ground
truth is matched to tracks with the Hungarian algorithm.
"""

from perception_core.eval.metrics import MotMetrics, evaluate_tracking

__all__ = ["MotMetrics", "evaluate_tracking"]
