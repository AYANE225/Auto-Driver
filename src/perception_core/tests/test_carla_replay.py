"""Exercise the real CLI on a moving-sensor recording, without CARLA or torch."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_carla_replay_reports_stable_ids_and_world_frame_gt(tmp_path):
    dataset = tmp_path / 'scene'
    (dataset / 'frames').mkdir(parents=True)
    (dataset / 'meta.json').write_text(json.dumps({'dt': 0.1, 'cameras': []}))
    for i in range(6):
        pose = np.eye(4)
        pose[:3, 3] = [100 + i, 0, 0]
        np.savez(dataset / 'frames' / f'{i:06d}.npz', timestamp=i * 0.1,
                 ego_pose=pose, points=np.empty((0, 4), dtype=np.float32),
                 gt_boxes=np.array([[110, 0, 1, 4, 2, 2, 0]]),
                 gt_labels=np.array(['car']), gt_ids=np.array([123]))
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(repo / 'src/perception_core'))
    report = tmp_path / 'report.json'
    result = subprocess.run([sys.executable, str(repo / 'carla/replay_demo.py'),
                             '--dataset', str(dataset), '--detector', 'gt',
                             '--no-video', '--report', str(report)], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text())
    assert payload['frames'] == 6
    assert payload['tp'] >= 3  # incorrect double world transform gives zero matches
    assert payload['id_switches'] == 0
    assert payload['identity']['idtp'] >= 3
    assert payload['hota']['hota'] > 0
    assert payload['visualization']['frames'] == 0
