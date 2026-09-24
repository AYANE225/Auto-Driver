"""Unit tests for the CLEAR-MOT tracking metrics (:mod:`perception_core.eval.metrics`).

These lock down the MOTA / MOTP / ID-switch / precision / recall backbone that
every headline tracking number in the README rests on. Each scenario is built
with a hand-computable TP / FP / FN / ID-switch count, so the asserted metric
values are exact rather than circular.
"""
import pytest

from perception_core.common.types import Box3D, Detection, Track
from perception_core.eval.metrics import evaluate_tracking


def _det(x, y, l=4.0, w=2.0):
    return Detection(box=Box3D(x, y, 0.0, l, w, 1.5, 0.0))


def _trk(track_id, x, y, l=4.0, w=2.0):
    return Track(track_id=track_id, box=Box3D(x, y, 0.0, l, w, 1.5, 0.0))


def test_perfect_tracking_scores_unit_mota_and_motp():
    gt = [[_det(0, 0), _det(10, 0)], [_det(0, 0), _det(10, 0)]]
    trk = [[_trk(1, 0, 0), _trk(2, 10, 0)], [_trk(1, 0, 0), _trk(2, 10, 0)]]
    m = evaluate_tracking(gt, trk, gt_id_frames=[[1, 2], [1, 2]])
    assert (m.tp, m.fp, m.fn, m.id_switches) == (4, 0, 0, 0)
    assert m.precision == 1.0 and m.recall == 1.0
    assert m.mota == 1.0
    assert m.motp == 1.0
    assert m.frames == 2 and m.gt_count == 4


def test_non_overlapping_track_is_both_fp_and_fn():
    # A single gt and a single track 100 m away are matched by the Hungarian
    # step, but their IoU is below threshold -> one miss AND one false positive.
    m = evaluate_tracking([[_det(0, 0)]], [[_trk(1, 100, 0)]])
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)
    assert m.precision == 0.0 and m.recall == 0.0
    assert m.mota == -1.0  # 1 - (fn + fp + idsw) / gt_count = 1 - 2/1


def test_extra_track_counts_as_false_positive():
    gt = [[_det(0, 0)]]
    trk = [[_trk(1, 0, 0), _trk(2, 50, 0)]]  # one good match, one spurious track
    m = evaluate_tracking(gt, trk)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.precision == 0.5 and m.recall == 1.0
    assert m.mota == 0.0  # 1 - (0 + 1 + 0)/1


def test_missing_tracks_are_all_false_negatives():
    m = evaluate_tracking([[_det(0, 0), _det(10, 0)]], [[]])
    assert (m.tp, m.fp, m.fn) == (0, 0, 2)
    assert m.recall == 0.0 and m.precision == 0.0
    assert m.mota == 0.0  # 1 - 2/2


def test_identity_change_on_same_object_is_one_id_switch():
    # One ground-truth object (stable id 7) tracked across two frames; the track
    # id flips 1 -> 2 in the second frame -> exactly one ID switch, no FP/FN.
    gt = [[_det(0, 0)], [_det(0, 0)]]
    trk = [[_trk(1, 0, 0)], [_trk(2, 0, 0)]]
    m = evaluate_tracking(gt, trk, gt_id_frames=[[7], [7]])
    assert m.tp == 2 and m.id_switches == 1
    assert m.fp == 0 and m.fn == 0
    assert m.mota == 0.5  # 1 - (0 + 0 + 1)/2


def test_stable_identity_has_no_id_switch():
    gt = [[_det(0, 0)], [_det(0, 0)]]
    trk = [[_trk(5, 0, 0)], [_trk(5, 0, 0)]]
    m = evaluate_tracking(gt, trk, gt_id_frames=[[7], [7]])
    assert m.id_switches == 0
    assert m.mota == 1.0


def test_motp_is_mean_iou_over_matches():
    # Box A footprint spans x in [-2, 2], y in [-1, 1] (area 8); box B shifted
    # +2 m in x overlaps on x in [0, 2] -> intersection 4, union 12 -> IoU = 1/3.
    m = evaluate_tracking([[_det(0, 0)]], [[_trk(1, 2, 0)]], iou_threshold=0.3)
    assert m.tp == 1
    assert m.motp == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_iou_threshold_gates_a_weak_match():
    # The same 1/3-overlap pair, but a stricter threshold rejects the match.
    m = evaluate_tracking([[_det(0, 0)]], [[_trk(1, 2, 0)]], iou_threshold=0.5)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_empty_sequence_returns_zeroed_metrics():
    m = evaluate_tracking([], [])
    assert (m.tp, m.fp, m.fn, m.id_switches) == (0, 0, 0, 0)
    assert m.mota == 0.0 and m.motp == 0.0
    assert m.frames == 0 and m.gt_count == 0
