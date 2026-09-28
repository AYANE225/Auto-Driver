"""Export sensor-frame replay samples for a static browser viewer.

All tracking frames must still be processed by the caller. This exporter only
subsamples visualization points; it never changes detection or evaluation.
"""
from pathlib import Path
import json

import numpy as np

from perception_core.common.ego_frame import detections_to_frame, output_to_frame
from perception_core.common.geometry import invert_se3


def _values(array):
    return np.round(np.asarray(array, dtype=float), 3).tolist()


def frame_payload(frame, output, elapsed_s, point_limit=2500):
    """Serialize one frame without modifying the pipeline's world-frame output."""
    if point_limit < 1:
        raise ValueError('point_limit must be positive')
    transform = invert_se3(frame.ego_pose) if frame.ego_pose is not None else np.eye(4)
    local = output_to_frame(output, transform)
    gt = detections_to_frame(frame.ground_truth or [], transform)
    points = np.empty((0, 3)) if frame.lidar is None else frame.lidar[:, :3]
    valid = np.isfinite(points).all(axis=1)
    valid &= (np.abs(points[:, 0]) <= 55) & (np.abs(points[:, 1]) <= 45)
    valid &= (points[:, 2] >= -3) & (points[:, 2] <= 4)
    points = points[valid]
    roi_count = len(points)
    if len(points) > point_limit:
        indices = np.random.default_rng(frame.frame_id).choice(len(points), point_limit, replace=False)
        points = points[np.sort(indices)]
    return {
        'frame_id': int(frame.frame_id), 'timestamp_s': frame.timestamp,
        'elapsed_s': round(elapsed_s, 3), 'point_count': frame.num_points,
        'roi_point_count': roi_count, 'points': _values(points),
        'ground_truth': [{'id': int(d.attributes.get('gt_id', i)), 'label': d.label.value,
                          'box': _values(d.box.to_array())} for i, d in enumerate(gt)],
        'tracks': [{'id': t.track_id, 'label': t.label.value,
                    'box': _values(t.box.to_array()), 'speed_mps': round(t.speed, 3),
                    'velocity_mps': _values(t.velocity), 'history': _values(t.history),
                    'missed_frames': t.time_since_update} for t in local.tracks],
        'predictions': [{'id': p.track_id,
                         'trajectories': [{'mode': tr.mode, 'xy': _values(tr.as_array()),
                                           'times_s': [tp.t for tp in tr.points]}
                                          for tr in p.trajectories]} for p in local.predictions],
    }


class ReplayExporter:
    """Write sampled JSON frames, optional RGB images, and a final manifest."""

    def __init__(self, directory, point_limit=2500, camera='front'):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / 'index.json').exists():
            raise FileExistsError('replay export exists; choose a new output directory')
        self.point_limit = point_limit
        self.camera = camera
        self.frames = []

    def add_frame(self, frame, output, elapsed_s):
        payload = frame_payload(frame, output, elapsed_s, self.point_limit)
        stem = f'{int(frame.frame_id):06d}'
        if self.camera in frame.images:
            from PIL import Image
            img = Image.fromarray(frame.images[self.camera])
            img.thumbnail((960, 540))
            img.save(self.directory / f'{stem}.jpg', quality=85)
            payload['image'] = f'{stem}.jpg'
        path = self.directory / f'{stem}.json'
        path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
        self.frames.append({'file': path.name, 'frame_id': payload['frame_id'],
                            'elapsed_s': payload['elapsed_s'], 'tracks': len(output.tracks)})

    def finish(self, metadata):
        manifest = {
            'schema_version': 1, 'coordinates': 'lidar: x forward, y left, z up; metres',
            'velocity': 'world velocity rotated into current lidar axes; no ego-velocity subtraction',
            'point_limit': self.point_limit, 'frames': self.frames, **metadata,
        }
        (self.directory / 'index.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
