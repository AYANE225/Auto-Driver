#!/usr/bin/env python3
"""Compare tracking changes using fixed, world-frame CARLA detections.

Capture with --dataset and --cache (requires YOLO). Benchmark with --cache and
--report (CPU only, requires the C++ extension). Cached detections never use GT;
GT is only passed to the unchanged evaluation functions after tracking.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'carla'))
from replay_demo import build_pipeline, evaluation_view
from perception_core.common.types import Box3D, Detection, Frame, ObjectClass, PerceptionOutput, SensorCalibration
from perception_core.detection.lidar_cluster import LidarClusterConfig
from perception_core.eval import evaluate_hota, evaluate_identity, evaluate_tracking
from perception_core.io.carla_dataset import CarlaDataset
from perception_core.tracking.mot import MultiObjectTracker, TrackerConfig


def serialize_detection(d):
    return {'box': d.box.to_array().tolist(), 'score': d.score, 'label': d.label.value,
            'source': d.source, 'num_points': d.num_points, 'attributes': d.attributes}


def deserialize_detection(d):
    return Detection(Box3D.from_array(d['box']), d['score'], ObjectClass(d['label']),
                     d['source'], d['num_points'], d['attributes'])


def capture(dataset, cache, device):
    if cache.exists():
        raise FileExistsError('choose a new cache path')
    ds = CarlaDataset(str(dataset), load_images=True, count_points=False)
    args = SimpleNamespace(voxel_size=.2, tracker='imm', gating=False, detector='fusion',
                           weights='yolov8n.pt', device=device, gate_iou=.1,
                           iou_backend='python', box_yaw_period='2pi', matching_policy='post_filter')
    pipe = build_pipeline(args)
    frames, digest = [], hashlib.sha256()
    for index in range(len(ds)):
        frame = ds.read_frame(index)
        out = pipe.process(frame)
        for suffix in ['.npz', '_front.jpg']:
            digest.update((dataset / 'frames' / f'{frame.frame_id:06d}{suffix}').read_bytes())
        frames.append({
            'timestamp': frame.timestamp, 'frame_id': frame.frame_id, 'ego_pose': frame.ego_pose.tolist(),
            'detections': [serialize_detection(d) for d in out.detections],
            'ground_truth': [serialize_detection(d) for d in frame.ground_truth],
            'tracks': [{'id': t.track_id, 'box': t.box.to_array().tolist(),
                        'velocity': t.velocity.tolist()} for t in out.tracks],
        })
        if index % 100 == 0:
            print(f'capture {dataset.name} {index}/{len(ds)}', flush=True)
    payload = {'schema_version': 1, 'dataset': dataset.name, 'meta': ds.meta,
               'pipeline_config': asdict(pipe.cfg), 'calib': json.loads((dataset / 'calib.json').read_text()),
               'sensor_files_sha256': digest.hexdigest(),
               'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'frames': frames}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, 'wt') as stream:
        json.dump(payload, stream, separators=(',', ':'), allow_nan=False)


def benchmark(cache, repeats):
    with gzip.open(cache, 'rt') as stream:
        data = json.load(stream)
    raw = data['calib']
    calib = SensorCalibration(**{k: {name: np.array(value) if k != 'image_size' else tuple(value)
                                     for name, value in mapping.items()} for k, mapping in raw.items()})
    lidar = LidarClusterConfig(**data['pipeline_config']['lidar'])
    frames, detections = [], []
    for row in data['frames']:
        frames.append(Frame(row['timestamp'], frame_id=row['frame_id'],
                            ego_pose=np.array(row['ego_pose']), calib=calib,
                            ground_truth=[deserialize_detection(d) for d in row['ground_truth']]))
        detections.append([deserialize_detection(d) for d in row['detections']])
    variants = {
        'python_original': ('python', 2*np.pi, 'post_filter'),
        'cpp_original': ('cpp', 2*np.pi, 'post_filter'),
        'cpp_axis': ('cpp', np.pi, 'post_filter'),
        'cpp_threshold': ('cpp', 2*np.pi, 'thresholded'),
        'cpp_axis_threshold': ('cpp', np.pi, 'thresholded'),
    }
    samples = {name: [] for name in variants}
    result = {}
    for repeat in range(repeats):
        names = list(variants)
        names = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in names:
            backend, period, policy = variants[name]
            options = dict(data['pipeline_config']['tracker'])
            options.update(iou_backend=backend, box_yaw_period=period, matching_policy=policy)
            tracker = MultiObjectTracker(TrackerConfig(**options))
            gt, ids, tracks, times = [], [], [], []
            equivalent, max_error = True, 0.0
            for index, (frame, dets) in enumerate(zip(frames, detections)):
                begin = perf_counter()
                output = tracker.update(dets, frame.timestamp)
                elapsed = (perf_counter() - begin)*1000
                if index >= 5:
                    times.append(elapsed)
                if repeat == 0:
                    expected = data['frames'][index]['tracks']
                    if [t.track_id for t in output] != [t['id'] for t in expected]:
                        equivalent = False
                    elif expected:
                        error = max(float(np.max(np.abs(np.r_[t.box.to_array(), t.velocity]
                                                         - np.r_[e['box'], e['velocity']])))
                                    for t, e in zip(output, expected))
                        max_error = max(max_error, error)
                        equivalent &= error < 1e-7
                    selected_frame, selected = evaluation_view(
                        frame, PerceptionOutput(frame.timestamp, tracks=output), lidar, 'front')
                    gt.append(selected_frame.ground_truth)
                    ids.append([int(d.attributes['gt_id']) for d in selected_frame.ground_truth])
                    tracks.append(selected.tracks)
            samples[name].append({'mean_ms': float(np.mean(times)), 'p95_ms': float(np.percentile(times, 95))})
            if repeat == 0:
                mot = evaluate_tracking(gt, tracks, .3, gt_id_frames=ids).as_dict()
                result[name] = {'config': asdict(tracker.cfg), 'iou_backend': tracker.iou_backend,
                                'quality': {**mot, 'hota': evaluate_hota(gt, tracks, ids).as_dict(),
                                            'identity': evaluate_identity(gt, tracks, ids).as_dict()},
                                'matches_captured_states': bool(equivalent), 'max_state_error': max_error}
                if name in ('python_original', 'cpp_original') and not equivalent:
                    raise AssertionError(f'{name} changed captured tracking states')
            print(data['dataset'], repeat+1, name, samples[name][-1], flush=True)
    for name in variants:
        result[name]['timing_runs'] = samples[name]
        result[name]['mean_ms'] = float(np.mean([run['mean_ms'] for run in samples[name]]))
        result[name]['mean_p95_ms'] = float(np.mean([run['p95_ms'] for run in samples[name]]))
    return {'dataset': data['dataset'], 'frames': len(frames), 'repeats': repeats,
            'warmup_frames': 5, 'scope': 'tracker.update only; excludes detection, IO, projection and evaluation',
            'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
            'sensor_files_sha256': data['sensor_files_sha256'], 'baseline_commit': data['baseline_commit'],
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'platform': platform.platform(),
                            'threads': {k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']}},
            'results': result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, help='capture detections into a new cache, then exit')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--report', type=Path)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    if args.dataset:
        capture(args.dataset, args.cache, args.device)
        return
    if args.repeats < 1 or args.report is None:
        ap.error('benchmark needs --report and positive --repeats')
    report = benchmark(args.cache, args.repeats)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
