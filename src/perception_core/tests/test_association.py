import numpy as np
import pytest

from perception_core.common.types import Box3D, Detection, Track
from perception_core.eval import evaluate_hota, evaluate_identity, evaluate_tracking


def det(x=0):
    return Detection(Box3D(x, 0, 0, 4, 2, 1.5))


def trk(tid=10, x=0):
    return Track(tid, det(x).box)


def test_actor_exit_does_not_change_surviving_identity():
    gt = [[det(), det(20)], [det(20)]]
    tracks = [[trk(10), trk(20, 20)], [trk(20, 20)]]
    ids = [[100, 200], [200]]
    assert evaluate_tracking(gt, tracks, gt_id_frames=ids).id_switches == 0
    assert evaluate_hota(gt, tracks, ids).hota == 1
    assert evaluate_identity(gt, tracks, ids).idf1 == 1


def test_fragmented_track_reduces_association_with_perfect_detection():
    gt = [[det()] for _ in range(4)]
    tracks = [[trk(tid)] for tid in (10, 10, 20, 20)]
    ids = [[42]] * 4
    hota = evaluate_hota(gt, tracks, ids)
    identity = evaluate_identity(gt, tracks, ids)
    assert hota.deta == 1
    assert hota.assa == 0.5
    assert hota.hota == pytest.approx(np.sqrt(0.5), abs=1e-4)
    assert (identity.idtp, identity.idfp, identity.idfn) == (2, 2, 2)
    assert identity.idf1 == 0.5


def test_extra_track_reduces_detection_without_changing_association():
    gt, tracks = [[det()]], [[trk(), trk(20, 50)]]
    hota = evaluate_hota(gt, tracks)
    assert hota.assa == 1
    assert hota.deta == 0.5
    assert hota.hota == pytest.approx(np.sqrt(0.5), abs=1e-4)
    assert evaluate_identity(gt, tracks).idf1 == pytest.approx(2/3, abs=1e-4)


def test_empty_prediction_counts_identity_false_negatives():
    identity = evaluate_identity([[det()]], [[]], [[1]])
    assert (identity.idtp, identity.idfp, identity.idfn) == (0, 0, 1)
    assert evaluate_hota([[det()]], [[]], [[1]]).hota == 0
