import json

import numpy as np
import pytest

from perception_core.common.geometry import make_se3, rot_z
from perception_core.common.types import Box3D, Frame, PerceptionOutput, Track
from perception_core.viz.replay import ReplayExporter, frame_payload


def test_export_transforms_world_tracks_without_mutation():
    pose = make_se3(rot_z(np.pi / 2), [100, 200, 0])
    track = Track(5, Box3D(100, 210, 1, 4, 2, 2, np.pi/2), velocity=np.array([0, 3.0]))
    frame = Frame(1.0, frame_id=9, ego_pose=pose, lidar=np.array([[1, 2, 0, .5]]))
    output = PerceptionOutput(1.0, tracks=[track])
    result = frame_payload(frame, output, 0.5)
    assert result['tracks'][0]['box'][:2] == [10.0, 0.0]
    assert result['tracks'][0]['velocity_mps'] == [3.0, 0.0]
    assert track.box.x == 100 and track.box.y == 210
    np.testing.assert_array_equal(track.velocity, [0, 3])
    assert result['points'] == [[1.0, 2.0, 0.0]]


def test_export_sampling_is_deterministic_and_bounded(tmp_path):
    points = np.random.default_rng(7).uniform(-2, 2, (100, 3))
    points[0, 0] = np.nan
    frame = Frame(0, frame_id=4, lidar=points)
    out = PerceptionOutput(0)
    a = frame_payload(frame, out, 0, point_limit=8)
    b = frame_payload(frame, out, 0, point_limit=8)
    assert a == b and len(a['points']) == 8 and a['roi_point_count'] == 99
    assert a['point_count'] == 100 and np.isnan(points[0, 0])
    exporter = ReplayExporter(tmp_path, point_limit=8)
    exporter.add_frame(frame, out, 0)
    exporter.finish({'processed_frames': 1})
    manifest = json.loads((tmp_path / 'index.json').read_text())
    assert json.loads((tmp_path / manifest['frames'][0]['file']).read_text()) == a
    with pytest.raises(FileExistsError):
        ReplayExporter(tmp_path)
