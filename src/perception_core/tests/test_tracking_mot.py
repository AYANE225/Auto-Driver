import numpy as np

from perception_core.common.types import Box3D, Detection, ObjectClass, TrackState
from perception_core.tracking.mot import MultiObjectTracker, TrackerConfig


def _det(x, y, label=ObjectClass.CAR):
    return Detection(box=Box3D(x, y, 0, 4, 2, 1.5, 0.0), score=0.9, label=label)


def test_mot_stable_ids_for_two_objects():
    tracker = MultiObjectTracker(TrackerConfig(min_hits=2, iou_threshold=0.05, max_age=3))
    final = []
    for k in range(12):
        t = k * 0.1
        dets = [_det(1 + 2.0 * t, 0.0), _det(1 + 2.0 * t, 10.0)]
        final = tracker.update(dets, t)
    assert len(final) == 2
    assert {tr.track_id for tr in final} == {0, 1}


def test_mot_estimates_velocity():
    tracker = MultiObjectTracker(TrackerConfig(min_hits=2, iou_threshold=0.05))
    tracks = []
    for k in range(15):
        tracks = tracker.update([_det(1 + 3.0 * (k * 0.1), 0.0)], k * 0.1)
    assert len(tracks) == 1
    assert np.allclose(tracks[0].velocity, [3.0, 0.0], atol=0.4)


def test_mot_promotes_after_min_hits():
    tracker = MultiObjectTracker(TrackerConfig(min_hits=3, iou_threshold=0.05))
    assert tracker.update([_det(0, 0)], 0.0) == []          # tentative, not returned
    assert tracker.update([_det(0.1, 0)], 0.1) == []        # still tentative
    confirmed = tracker.update([_det(0.2, 0)], 0.2)         # 3rd hit -> confirmed
    assert len(confirmed) == 1
    assert confirmed[0].state is TrackState.CONFIRMED


def test_mot_coasts_through_short_occlusion():
    tracker = MultiObjectTracker(TrackerConfig(min_hits=2, iou_threshold=0.05, max_age=5))
    for k in range(4):
        tracker.update([_det(1 + 2.0 * (k * 0.1), 0.0)], k * 0.1)
    # object disappears for 3 frames (< max_age): the track must survive
    for k in range(4, 7):
        tracker.update([], k * 0.1)
    assert len(tracker.all_tracks) == 1


def test_mot_deletes_after_max_age():
    tracker = MultiObjectTracker(TrackerConfig(min_hits=2, iou_threshold=0.05, max_age=2))
    for k in range(4):
        tracker.update([_det(0, 0)], k * 0.1)
    for k in range(4, 10):
        tracker.update([], k * 0.1)
    assert len(tracker.all_tracks) == 0
