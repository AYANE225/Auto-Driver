#!/usr/bin/env python3
"""Export denser display-only point samples for the website's 3D viewer.

Uses the existing replay manifest's frame IDs. Signed 16-bit little-endian XYZ
with 0.002 m per integer uses six bytes per point (at most 1 mm error per axis).
No detections, tracking outputs or evaluation reports are modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--points', type=int, default=20000)
    args = parser.parse_args()
    if args.points < 1:
        parser.error('--points must be positive')
    if args.out.exists():
        raise FileExistsError('choose a new output directory')
    replay = json.loads((args.replay / 'index.json').read_text())
    frames = []
    args.out.mkdir(parents=True)
    for entry in replay['frames']:
        fid = entry['frame_id']
        source = args.dataset / 'frames' / f'{fid:06d}.npz'
        existing = json.loads((args.replay / entry['file']).read_text())
        with np.load(source) as raw:
            points = raw['points'][:, :3]
            timestamp = float(raw['timestamp'])
        if abs(timestamp - existing['timestamp_s']) > 1e-7 or len(points) != existing['point_count']:
            raise ValueError(f'frame {fid} does not match the existing replay')
        mask = np.isfinite(points).all(axis=1)
        mask &= (np.abs(points[:, 0]) <= 55) & (np.abs(points[:, 1]) <= 45)
        mask &= (points[:, 2] >= -3) & (points[:, 2] <= 4)
        points = points[mask]
        if len(points) > args.points:
            indices = np.random.default_rng(fid).choice(len(points), args.points, replace=False)
            points = points[np.sort(indices)]
        encoded = np.rint(points.astype(float) / .002).astype('<i2')
        filename = f'{fid:06d}.bin'
        contents = encoded.tobytes()
        (args.out / filename).write_bytes(contents)
        frames.append({'frame_id': fid, 'timestamp_s': timestamp, 'file': filename,
                       'points': len(points), 'bytes': len(contents),
                       'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'sha256': hashlib.sha256(contents).hexdigest()})
    result = {'schema_version': 1, 'dataset': replay['dataset'],
              'coordinates': replay['coordinates'], 'dtype': 'int16_le_xyz', 'scale_m': .002,
              'point_limit': args.points, 'bounds_m': [[-55, 55], [-45, 45], [-3, 4]],
              'replay_manifest_sha256': hashlib.sha256((args.replay / 'index.json').read_bytes()).hexdigest(),
              'frames': frames}
    (args.out / 'index.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
