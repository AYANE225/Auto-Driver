"""Tracking-by-detection multi-object tracker (Kalman + Hungarian association)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from perception_core.common.geometry import bev_iou
from perception_core.common.types import Detection, ObjectClass, Track, TrackState
from perception_core.tracking.imm import IMMFilter
from perception_core.tracking.kalman import ConstantVelocityKF


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.1     # min BEV IoU to accept an association
    max_age: int = 5               # frames without a hit before a track is deleted
    min_hits: int = 3              # hits required to promote tentative -> confirmed
    process_noise: float = 2.0
    measurement_noise: float = 0.5
    yaw_smoothing: float = 0.5     # EMA weight on new heading measurement
    dim_smoothing: float = 0.3     # EMA weight on new size measurement
    max_history: int = 50
    motion_model: str = "cv"       # "cv" (constant velocity) | "imm" (CV + constant-turn IMM)
    gating: bool = False           # gate associations by Mahalanobis distance
    gate_chi2: float = 9.21        # chi-square gate, 2 DOF ~ 99% acceptance
    turn_rate_noise: float = 0.3   # IMM constant-turn yaw-rate spectral density
    transition_stay: float = 0.95  # IMM Markov self-transition probability


def _make_filter(cfg: TrackerConfig):
    if cfg.motion_model == "imm":
        return IMMFilter(cfg.process_noise, cfg.measurement_noise,
                         cfg.turn_rate_noise, cfg.transition_stay)
    return ConstantVelocityKF(cfg.process_noise, cfg.measurement_noise)


def _angle_ema(prev: float, meas: float, alpha: float) -> float:
    delta = (meas - prev + np.pi) % (2 * np.pi) - np.pi
    return float((prev + alpha * delta + np.pi) % (2 * np.pi) - np.pi)


class _TrackData:
    def __init__(self, det: Detection, track_id: int, cfg: TrackerConfig, t: float) -> None:
        self.id = track_id
        self.cfg = cfg
        self.kf = _make_filter(cfg)
        self.kf.init_state([det.box.x, det.box.y])
        self.box = det.box.copy()
        self.label_votes = {det.label: det.score}
        self.score = det.score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.last_time = t
        self._last_dt = 0.0
        self.heading: float = None
        self.yaw_rate_est = 0.0
        self.history: List[np.ndarray] = [self.kf.position]

    @property
    def label(self) -> ObjectClass:
        return max(self.label_votes, key=self.label_votes.get)

    def predict(self, t: float) -> None:
        dt = t - self.last_time
        self.kf.predict(dt)
        self._last_dt = max(dt, 1e-3)
        self.last_time = t
        self.box.x, self.box.y = self.kf.position
        self.age += 1
        self.time_since_update += 1

    def update(self, det: Detection) -> None:
        self.kf.update([det.box.x, det.box.y])
        self.box.x, self.box.y = self.kf.position
        a_dim = self.cfg.dim_smoothing
        self.box.z = (1 - a_dim) * self.box.z + a_dim * det.box.z
        self.box.l = (1 - a_dim) * self.box.l + a_dim * det.box.l
        self.box.w = (1 - a_dim) * self.box.w + a_dim * det.box.w
        self.box.h = (1 - a_dim) * self.box.h + a_dim * det.box.h
        self.box.yaw = _angle_ema(self.box.yaw, det.box.yaw, self.cfg.yaw_smoothing)
        self.label_votes[det.label] = self.label_votes.get(det.label, 0.0) + det.score
        self.score = 0.7 * self.score + 0.3 * det.score
        self.hits += 1
        self.time_since_update = 0
        self._estimate_turn_rate()
        imm_yaw_rate = getattr(self.kf, "yaw_rate", None)
        if imm_yaw_rate is not None:  # IMM estimates the turn rate directly
            self.yaw_rate_est = float(imm_yaw_rate)
        self.history.append(self.kf.position)
        if len(self.history) > self.cfg.max_history:
            self.history.pop(0)
        if self.hits >= self.cfg.min_hits:
            self.state = TrackState.CONFIRMED

    def _estimate_turn_rate(self) -> None:
        v = self.kf.velocity
        if float(np.linalg.norm(v)) < 0.5:  # heading ill-defined at low speed
            return
        head = float(np.arctan2(v[1], v[0]))
        if self.heading is not None and self._last_dt > 1e-3:
            dyaw = (head - self.heading + np.pi) % (2 * np.pi) - np.pi
            self.yaw_rate_est = 0.7 * self.yaw_rate_est + 0.3 * (dyaw / self._last_dt)
        self.heading = head

    def to_track(self) -> Track:
        return Track(
            track_id=self.id, box=self.box.copy(), label=self.label, score=float(self.score),
            velocity=self.kf.velocity, yaw_rate=float(self.yaw_rate_est), age=self.age,
            hits=self.hits, time_since_update=self.time_since_update, state=self.state,
            history=[h.copy() for h in self.history],
        )
def associate(
    tracks: List[_TrackData], detections: List[Detection], iou_threshold: float,
    gate_chi2: float = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Optimally match tracks to detections by BEV IoU (Hungarian algorithm).

    When ``gate_chi2`` is set, a track/detection pair is only eligible if the
    detection centre falls inside the track's Mahalanobis gate; gated-out pairs
    are given zero affinity so the assignment never picks them.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    iou = np.zeros((len(tracks), len(detections)))
    for i, trk in enumerate(tracks):
        for j, det in enumerate(detections):
            if gate_chi2 is not None and hasattr(trk.kf, "gating_distance"):
                if trk.kf.gating_distance([det.box.x, det.box.y]) > gate_chi2:
                    continue  # outside the gate -> leave affinity at 0 (disallowed)
            iou[i, j] = bev_iou(trk.box, det.box)

    row, col = linear_sum_assignment(-iou)
    matches, un_trk, un_det = [], [], []
    matched_t, matched_d = set(), set()
    for r, c in zip(row, col):
        if iou[r, c] >= iou_threshold:
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
    un_trk = [i for i in range(len(tracks)) if i not in matched_t]
    un_det = [j for j in range(len(detections)) if j not in matched_d]
    return matches, un_trk, un_det


class MultiObjectTracker:
    """Frame-to-frame tracker. Call :meth:`update` once per synchronised frame."""

    def __init__(self, config: TrackerConfig = None) -> None:
        self.cfg = config or TrackerConfig()
        self.tracks: List[_TrackData] = []
        self._next_id = 0

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 0

    def update(self, detections: List[Detection], timestamp: float) -> List[Track]:
        for trk in self.tracks:
            trk.predict(timestamp)

        matches, un_trk, un_det = associate(
            self.tracks, detections, self.cfg.iou_threshold,
            gate_chi2=self.cfg.gate_chi2 if self.cfg.gating else None,
        )
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])
        for d_idx in un_det:
            self.tracks.append(_TrackData(detections[d_idx], self._next_id, self.cfg, timestamp))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.cfg.max_age]
        return [t.to_track() for t in self.tracks if t.state is TrackState.CONFIRMED]

    @property
    def all_tracks(self) -> List[Track]:
        return [t.to_track() for t in self.tracks]

