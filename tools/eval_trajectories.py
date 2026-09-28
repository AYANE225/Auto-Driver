#!/usr/bin/env python3
"""Town-disjoint CARLA forecasting evaluation using recorded actor histories.

Train: Town01/02/03; validation: Town04; test: Town05/10HD. All models receive
2 s of GT history and predict the same six positions through 3 s. Ridge learns
CV residuals; alpha is chosen using validation ADE only. No sensor detections
or tracking errors are represented in this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from perception_core.prediction.history import (
    constant_acceleration, history_features, to_local, to_world,
)

SPLITS = {'train': ['Town01_Opt', 'Town02_Opt', 'Town03_Opt'],
          'validation': ['Town04_Opt'], 'test': ['Town05_Opt', 'Town10HD_Opt']}


def load_windows(path):
    with np.load(path, allow_pickle=False) as data:
        dt = float(data['dt'])
        if not np.isclose(dt, 0.1):
            raise ValueError('this protocol requires 10 Hz recordings')
        order = np.lexsort((data['tick'], data['actor_id']))
        ids, ticks = data['actor_id'][order], data['tick'][order]
        xy = np.column_stack([data['x'][order], data['y'][order]]).astype(float)
        classes = data['cls'][order]
    bounds = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
    histories, futures, labels, actors = [], [], [], []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        # Global tick grid: one forecast origin per second. Reject missing ticks.
        for origin in range(begin + 20, end - 30):
            if ticks[origin] % 10 or not np.all(np.diff(ticks[origin-20:origin+31]) == 1):
                continue
            history = xy[origin-20:origin+1:2]  # 11 observations, 0.2 s spacing
            future = xy[origin+5:origin+31:5]
            if not np.isfinite(history).all() or not np.isfinite(future).all():
                continue
            histories.append(history)
            futures.append(future)
            labels.append(classes[origin])
            actors.append(ids[origin])
    if not histories:
        raise ValueError(f'no complete windows: {path}')
    return np.array(histories), np.array(futures), np.array(labels), len(set(actors))


def score(forecast, future, mask=None):
    errors = np.linalg.norm(forecast - future, axis=-1)
    if mask is not None:
        errors = errors[mask]
    if not len(errors):
        return {'windows': 0, 'ade_m': None, 'fde_m': None, 'miss_rate_2m': None}
    return {'windows': len(errors), 'ade_m': float(errors.mean()),
            'fde_m': float(errors[:, -1].mean()),
            'miss_rate_2m': float(np.mean(errors[:, -1] > 2))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('carla/data/trajectories'))
    ap.add_argument('--out', type=Path, default=Path('docs/benchmarks/forecasting'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    times = np.arange(1, 7) * 0.5
    data, provenance = {}, {}
    for town in sum(SPLITS.values(), []):
        path = args.root / (town + '.npz')
        history, future, labels, actors = load_windows(path)
        features, cv, c, s = history_features(history, 0.2, times)
        data[town] = (history, future, labels, features, cv, c, s)
        provenance[town] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'windows': len(history), 'actors': actors}
        print(town, provenance[town], flush=True)
    train = [data[town] for town in SPLITS['train']]
    x = np.concatenate([row[3] for row in train])
    y = np.concatenate([to_local(row[1] - row[4], row[5], row[6]).reshape(len(row[0]), -1)
                        for row in train])
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    val = data[SPLITS['validation'][0]]
    candidates = []
    models = {}
    for alpha in [1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=alpha).fit(x, y)
        local = model.predict(scaler.transform(val[3])).reshape(-1, 6, 2)
        metrics = score(val[4] + to_world(local, val[5], val[6]), val[1])
        candidates.append({'alpha': alpha, **metrics})
        models[alpha] = model
    selected = min(candidates, key=lambda row: row['ade_m'])['alpha']
    model = models[selected]
    results, predictions = {}, {}
    for town in SPLITS['validation'] + SPLITS['test']:
        history, future, labels, features, cv, c, s = data[town]
        residual = model.predict(scaler.transform(features)).reshape(-1, 6, 2)
        forecasts = {'cv': cv, 'ca': constant_acceleration(history, 0.2, times),
                     'ridge': cv + to_world(residual, c, s)}
        moving = np.linalg.norm(history[:, -1] - history[:, -6], axis=1) >= 0.5
        masks = {'all': np.ones(len(history), dtype=bool), 'moving': moving,
                 'vehicle': labels == 0, 'pedestrian': labels == 1, 'two_wheeler': labels == 2}
        results[town] = {name: {group: score(pred, future, mask) for group, mask in masks.items()}
                         for name, pred in forecasts.items()}
        if town in SPLITS['test']:
            predictions[town] = (future, moving, forecasts)
    results['test_pooled'] = {
        name: {group: score(np.concatenate([v[2][name] for v in predictions.values()]),
                            np.concatenate([v[0] for v in predictions.values()]),
                            np.concatenate([v[1] for v in predictions.values()])
                            if group == 'moving' else None)
               for group in ['all', 'moving']} for name in ['cv', 'ca', 'ridge']}
    report = {'protocol': {'history_s': 2, 'history_step_s': 0.2, 'horizon_s': 3,
                           'forecast_step_s': 0.5, 'origin_stride_s': 1,
                           'input': 'ground-truth actor XY histories, not perception output',
                           'moving': 'past 1 s displacement >= 0.5 m',
                           'aggregation': 'window-weighted; overlapping windows are correlated',
                           'selection': 'validation all-window ADE only; no test fitting',
                           'splits': SPLITS}, 'data': provenance,
              'validation_candidates': candidates, 'selected_alpha': selected,
              'results': results, 'environment': {'python': platform.python_version(),
                                                 'numpy': np.__version__}}
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.out / 'ridge.npz', coef=model.coef_, intercept=model.intercept_,
                        feature_mean=scaler.mean_, feature_scale=scaler.scale_, alpha=selected,
                        history_step_s=0.2, times=times)
    print(json.dumps(results['test_pooled'], indent=2), flush=True)


if __name__ == '__main__':
    main()
