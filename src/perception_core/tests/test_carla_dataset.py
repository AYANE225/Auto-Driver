import json

import numpy as np

from perception_core.io.carla_dataset import CarlaDataset


def test_reader_preserves_actor_ids_and_counts_points_in_sensor_frame(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "meta.json").write_text(json.dumps({"dt": 0.1, "cameras": []}))
    pose = np.eye(4)
    pose[:3, 3] = [100, 20, 2]
    for fid, ids in [(2, [41, 99]), (10, [99])]:
        boxes = np.array([[105, 20, 2, 4, 2, 2, 0] for _ in ids])
        np.savez(tmp_path / "frames" / f"{fid:06d}.npz", timestamp=fid * 0.1,
                 points=np.array([[5, 0, 0, 0.8], [30, 0, 0, 0.4]], np.float32),
                 ego_pose=pose, gt_boxes=boxes, gt_ids=ids,
                 gt_labels=np.array(["car"] * len(ids)))
    ds = CarlaDataset(str(tmp_path))
    assert len(ds) == 2
    first, last = ds.read_frame(0), ds.read_frame(1)
    assert [first.frame_id, last.frame_id] == [2, 10]
    assert [d.attributes['gt_id'] for d in first.ground_truth] == [41, 99]
    assert last.ground_truth[0].attributes['gt_id'] == 99
    assert last.ground_truth[0].box.x == 105  # remains in world coordinates
    assert last.ground_truth[0].num_points == 1
    assert ds.calib is None
