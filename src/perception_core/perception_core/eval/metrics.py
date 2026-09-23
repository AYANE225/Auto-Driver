"""CLEAR-MOT style tracking metrics matched by BEV IoU."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from perception_core.common.geometry import bev_iou
from perception_core.common.types import Detection, Track


@dataclass
class MotMetrics:
    frames: int
    gt_count: int
    tp: int
    fp: int
    fn: int
    id_switches: int
    precision: float
    recall: float
    mota: float
    motp: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_tracking(
    gt_frames: Sequence[List[Detection]],
    track_frames: Sequence[List[Track]],
    iou_threshold: float = 0.3,
    gt_id_frames: Optional[Sequence[List[int]]] = None,
) -> MotMetrics:
    """Compare per-frame tracks against ground truth.

    ``gt_id_frames`` supplies a stable identity per ground-truth object; when
    omitted, the object's index within its frame is used as identity (valid when
    the generator emits objects in a stable order, as the synthetic one does).
    """
    tp = fp = fn = idsw = 0
    matched_iou_sum = 0.0
    gt_total = 0
    last_match: Dict[int, int] = {}  # gt identity -> track_id from previous frame

    for f, (gts, trks) in enumerate(zip(gt_frames, track_frames)):
        gt_ids = gt_id_frames[f] if gt_id_frames is not None else list(range(len(gts)))
        gt_total += len(gts)

        if gts and trks:
            iou = np.array([[bev_iou(g.box, t.box) for t in trks] for g in gts])
            rows, cols = linear_sum_assignment(-iou)
            matched_g, matched_t = set(), set()
            for r, c in zip(rows, cols):
                if iou[r, c] >= iou_threshold:
                    tp += 1
                    matched_iou_sum += iou[r, c]
                    matched_g.add(r)
                    matched_t.add(c)
                    gid = gt_ids[r]
                    if gid in last_match and last_match[gid] != trks[c].track_id:
                        idsw += 1
                    last_match[gid] = trks[c].track_id
            fn += len(gts) - len(matched_g)
            fp += len(trks) - len(matched_t)
        else:
            fn += len(gts)
            fp += len(trks)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mota = 1.0 - (fn + fp + idsw) / gt_total if gt_total else 0.0
    motp = matched_iou_sum / tp if tp else 0.0
    return MotMetrics(
        frames=len(gt_frames), gt_count=gt_total, tp=tp, fp=fp, fn=fn,
        id_switches=idsw, precision=round(precision, 4), recall=round(recall, 4),
        mota=round(mota, 4), motp=round(motp, 4),
    )
