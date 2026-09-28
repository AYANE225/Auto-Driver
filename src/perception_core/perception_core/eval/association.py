"""Association-aware tracking metrics: HOTA and IDF1.

CLEAR-MOT's MOTA is dominated by detection errors and counts an identity switch
as a single event, however long the wrong identity then persists. The two
metrics here weigh *association* quality properly and are the headline numbers
of current tracking benchmarks (KITTI / MOTChallenge report HOTA, IDF1 is the
classic identity metric):

* **HOTA** (Luiten et al., IJCV 2021) = sqrt(DetA * AssA), averaged over 19
  localisation thresholds alpha = 0.05 ... 0.95. It decomposes into detection
  accuracy (DetA), association accuracy (AssA) and localisation accuracy (LocA).
* **IDF1** (Ristani et al., ECCV-W 2016): F1 score of the best one-to-one
  mapping between ground-truth identities and track identities over the whole
  sequence (IDTP / IDFP / IDFN).

Both follow the reference TrackEval implementation, use bird's-eye-view IoU as
the similarity, and are pure NumPy/SciPy like the rest of ``perception_core``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from perception_core.common.geometry import bev_iou
from perception_core.common.types import Detection, Track

__all__ = ["HotaMetrics", "IdentityMetrics", "evaluate_hota", "evaluate_identity"]

_EPS = np.finfo(float).eps
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)  # 0.05, 0.10, ..., 0.95

_FrameData = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (gt idx, track idx, similarity)


@dataclass
class HotaMetrics:
    hota: float
    deta: float
    assa: float
    loca: float
    det_re: float
    det_pr: float
    ass_re: float
    ass_pr: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class IdentityMetrics:
    idf1: float
    idp: float
    idr: float
    idtp: int
    idfp: int
    idfn: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _index_frames(
    gt_frames: Sequence[List[Detection]],
    track_frames: Sequence[List[Track]],
    gt_id_frames: Optional[Sequence[List[int]]],
) -> Tuple[List[_FrameData], int, int]:
    """Map raw identities to dense indices and compute per-frame BEV-IoU matrices."""
    gt_map: Dict[int, int] = {}
    tr_map: Dict[int, int] = {}
    frames: List[_FrameData] = []
    for f, (gts, trks) in enumerate(zip(gt_frames, track_frames)):
        raw_ids = gt_id_frames[f] if gt_id_frames is not None else range(len(gts))
        g = np.array([gt_map.setdefault(i, len(gt_map)) for i in raw_ids], dtype=int)
        t = np.array([tr_map.setdefault(tr.track_id, len(tr_map)) for tr in trks], dtype=int)
        if len(gts) and len(trks):
            sim = np.array([[bev_iou(gd.box, tr.box) for tr in trks] for gd in gts])
        else:
            sim = np.zeros((len(gts), len(trks)))
        frames.append((g, t, sim))
    return frames, len(gt_map), len(tr_map)


def _id_counts(frames: List[_FrameData], n_gt: int, n_tr: int) -> Tuple[np.ndarray, np.ndarray]:
    gt_count, tr_count = np.zeros(n_gt), np.zeros(n_tr)
    for g, t, _ in frames:
        np.add.at(gt_count, g, 1)
        np.add.at(tr_count, t, 1)
    return gt_count, tr_count


def evaluate_hota(
    gt_frames: Sequence[List[Detection]],
    track_frames: Sequence[List[Track]],
    gt_id_frames: Optional[Sequence[List[int]]] = None,
) -> HotaMetrics:
    """HOTA / DetA / AssA / LocA averaged over the standard 19 alpha thresholds.

    ``gt_id_frames`` gives a stable identity per ground-truth object (as in
    :func:`~perception_core.eval.metrics.evaluate_tracking`); when omitted, the
    object's index within its frame is used.
    """
    frames, n_gt, n_tr = _index_frames(gt_frames, track_frames, gt_id_frames)
    if n_gt == 0 or n_tr == 0:
        return HotaMetrics(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    gt_count, tr_count = _id_counts(frames, n_gt, n_tr)

    # 1) Global alignment score between every (gt id, track id) pair.
    potential = np.zeros((n_gt, n_tr))
    for g, t, sim in frames:
        if sim.size:
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            sim_iou = np.where(denom > _EPS, sim / np.maximum(denom, _EPS), 0.0)
            potential[g[:, None], t[None, :]] += sim_iou
    global_align = potential / np.maximum(_EPS, gt_count[:, None] + tr_count[None, :] - potential)

    # 2) Per-frame matching that maximises alignment-weighted similarity.
    n_a = len(HOTA_ALPHAS)
    tp, fn, fp, loc = np.zeros(n_a), np.zeros(n_a), np.zeros(n_a), np.zeros(n_a)
    matches = np.zeros((n_a, n_gt, n_tr))
    for g, t, sim in frames:
        if len(g) == 0 or len(t) == 0:
            fn += len(g)
            fp += len(t)
            continue
        rows, cols = linear_sum_assignment(-(global_align[g[:, None], t[None, :]] * sim))
        for a, alpha in enumerate(HOTA_ALPHAS):
            ok = sim[rows, cols] >= alpha - _EPS
            r, c = rows[ok], cols[ok]
            tp[a] += len(r)
            fn[a] += len(g) - len(r)
            fp[a] += len(t) - len(r)
            if len(r):
                loc[a] += sim[r, c].sum()
                matches[a, g[r], t[c]] += 1

    # 3) Association scores from the matched (gt id, track id) pair counts.
    ass_a, ass_re, ass_pr = np.zeros(n_a), np.zeros(n_a), np.zeros(n_a)
    for a in range(n_a):
        mc = matches[a]
        denom = np.maximum(1.0, tp[a])
        ass_a[a] = (mc * mc / np.maximum(1.0, gt_count[:, None] + tr_count[None, :] - mc)).sum() / denom
        ass_re[a] = (mc * mc / np.maximum(1.0, gt_count[:, None])).sum() / denom
        ass_pr[a] = (mc * mc / np.maximum(1.0, tr_count[None, :])).sum() / denom

    det_a = tp / np.maximum(1.0, tp + fn + fp)
    det_re = tp / np.maximum(1.0, tp + fn)
    det_pr = tp / np.maximum(1.0, tp + fp)
    loc_a = np.maximum(1e-10, loc) / np.maximum(1e-10, tp)
    hota = np.sqrt(det_a * ass_a)

    def _m(x: np.ndarray) -> float:
        return round(float(np.mean(x)), 4)

    return HotaMetrics(hota=_m(hota), deta=_m(det_a), assa=_m(ass_a), loca=_m(loc_a),
                       det_re=_m(det_re), det_pr=_m(det_pr), ass_re=_m(ass_re),
                       ass_pr=_m(ass_pr))


def evaluate_identity(
    gt_frames: Sequence[List[Detection]],
    track_frames: Sequence[List[Track]],
    gt_id_frames: Optional[Sequence[List[int]]] = None,
    iou_threshold: float = 0.5,
) -> IdentityMetrics:
    """IDF1 / IDP / IDR from the optimal global identity assignment."""
    frames, n_gt, n_tr = _index_frames(gt_frames, track_frames, gt_id_frames)
    gt_count, tr_count = _id_counts(frames, n_gt, n_tr)
    if n_gt == 0 or n_tr == 0:
        idfn, idfp = int(gt_count.sum()), int(tr_count.sum())
        return IdentityMetrics(0.0, 0.0, 0.0, 0, idfp, idfn)

    potential = np.zeros((n_gt, n_tr))  # frames in which a gt id and a track id overlap
    for g, t, sim in frames:
        if sim.size:
            r, c = np.nonzero(sim >= iou_threshold)
            np.add.at(potential, (g[r], t[c]), 1)

    # Square cost matrix: real ids plus one dummy "unmatched" slot per id.
    n = n_gt + n_tr
    fn_mat, fp_mat = np.zeros((n, n)), np.zeros((n, n))
    fp_mat[n_gt:, :n_tr] = 1e10
    fn_mat[:n_gt, n_tr:] = 1e10
    for i in range(n_gt):
        fn_mat[i, :n_tr] = gt_count[i]
        fn_mat[i, n_tr + i] = gt_count[i]
    for j in range(n_tr):
        fp_mat[:n_gt, j] = tr_count[j]
        fp_mat[n_gt + j, j] = tr_count[j]
    fn_mat[:n_gt, :n_tr] -= potential
    fp_mat[:n_gt, :n_tr] -= potential

    rows, cols = linear_sum_assignment(fn_mat + fp_mat)
    idfn = int(round(fn_mat[rows, cols].sum()))
    idfp = int(round(fp_mat[rows, cols].sum()))
    idtp = int(round(gt_count.sum())) - idfn
    return IdentityMetrics(
        idf1=round(idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn), 4),
        idp=round(idtp / max(1.0, idtp + idfp), 4),
        idr=round(idtp / max(1.0, idtp + idfn), 4),
        idtp=idtp, idfp=idfp, idfn=idfn,
    )
