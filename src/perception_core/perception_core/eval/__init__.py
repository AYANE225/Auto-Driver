"""Lightweight, dependency-free evaluation for detection, tracking and prediction.

Tracking metrics follow the classic CLEAR-MOT definitions (MOTA / MOTP / ID
switches) plus per-frame precision and recall, matched by bird's-eye-view IoU.
Prediction metrics follow the standard displacement measures used by motion
forecasting benchmarks (ADE / FDE / minADE / minFDE / miss-rate).
"""

from perception_core.eval.metrics import MotMetrics, evaluate_tracking
from perception_core.eval.prediction_metrics import (
    PredictionMetrics,
    PredictionSample,
    evaluate_prediction,
)

__all__ = [
    "MotMetrics",
    "evaluate_tracking",
    "PredictionMetrics",
    "PredictionSample",
    "evaluate_prediction",
]
