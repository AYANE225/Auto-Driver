"""Trajectory-forecasting metrics: ADE / FDE / minADE / minFDE / miss-rate.

Standard displacement metrics used by motion-prediction benchmarks (Argoverse,
nuScenes, Waymo). Given a set of predicted trajectories (optionally multi-modal)
and the object's realised future positions sampled at the same horizon times:

* ADE      -- mean L2 displacement over the horizon, most-likely (best) mode
* FDE      -- L2 displacement at the final horizon step, best mode
* minADE_k -- min mean displacement over the ``k`` predicted modes
* minFDE_k -- min final displacement over the ``k`` predicted modes
* miss-rate -- fraction of samples whose best-mode FDE exceeds a threshold

The metrics are pure NumPy so they unit-test in milliseconds and carry no ROS /
CARLA / deep-learning dependency, matching the rest of ``perception_core``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np

from perception_core.common.types import Trajectory


@dataclass
class PredictionSample:
    """One evaluation instance: predicted modes vs. the realised future."""

    trajectories: List[Trajectory]   # one or more predicted hypotheses (weighted)
    gt_future: np.ndarray            # (H, 2) realised future waypoints at the same times


@dataclass
class PredictionMetrics:
    samples: int
    horizon_steps: int
    ade: float
    fde: float
    min_ade: float
    min_fde: float
    miss_rate: float
    modes: float                     # mean number of predicted modes per sample

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _displacements(traj: Trajectory, gt: np.ndarray) -> np.ndarray:
    """Per-step L2 error between one predicted mode and the realised future."""
    pred = traj.as_array()
    n = min(len(pred), len(gt))
    if n == 0:
        return np.zeros(0)
    return np.linalg.norm(pred[:n] - gt[:n], axis=1)


def evaluate_prediction(
    samples: Sequence[PredictionSample], miss_threshold: float = 2.0
) -> PredictionMetrics:
    """Aggregate ADE / FDE / minADE / minFDE / miss-rate over a set of samples."""
    ades, fdes, min_ades, min_fdes, misses, mode_counts = [], [], [], [], [], []
    horizon = 0
    for s in samples:
        gt = np.asarray(s.gt_future, dtype=float).reshape(-1, 2)
        if gt.shape[0] == 0 or not s.trajectories:
            continue
        per_mode = [d for d in (_displacements(t, gt) for t in s.trajectories) if d.size]
        if not per_mode:
            continue
        best = max(s.trajectories, key=lambda t: t.confidence)
        best_d = _displacements(best, gt)
        if best_d.size == 0:
            continue
        horizon = max(horizon, best_d.size)
        ades.append(float(best_d.mean()))
        fdes.append(float(best_d[-1]))
        min_ades.append(min(float(d.mean()) for d in per_mode))
        min_fdes.append(min(float(d[-1]) for d in per_mode))
        misses.append(1.0 if best_d[-1] > miss_threshold else 0.0)
        mode_counts.append(len(s.trajectories))

    if not ades:
        return PredictionMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return PredictionMetrics(
        samples=len(ades),
        horizon_steps=horizon,
        ade=round(float(np.mean(ades)), 4),
        fde=round(float(np.mean(fdes)), 4),
        min_ade=round(float(np.mean(min_ades)), 4),
        min_fde=round(float(np.mean(min_fdes)), 4),
        miss_rate=round(float(np.mean(misses)), 4),
        modes=round(float(np.mean(mode_counts)), 3),
    )
