#!/usr/bin/env python3
"""Export evenly spaced moving test windows for browser forecast comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_trajectories import load_windows
from perception_core.prediction.history import constant_acceleration, history_features, to_world


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('carla/data/trajectories'))
    ap.add_argument('--model', type=Path, default=Path('docs/benchmarks/forecasting/ridge.npz'))
    ap.add_argument('--out', type=Path, default=Path('docs/assets/forecast_examples.json'))
    args = ap.parse_args()
    examples, sources = [], {}
    with np.load(args.model, allow_pickle=False) as saved:
        model = {key: saved[key] for key in saved.files}
    for town in ['Town05_Opt', 'Town10HD_Opt']:
        path = args.root / (town + '.npz')
        sources[town] = hashlib.sha256(path.read_bytes()).hexdigest()
        history, future, labels, _ = load_windows(path)
        moving = np.flatnonzero(np.linalg.norm(history[:, -1] - history[:, -6], axis=1) >= 0.5)
        selected = moving[np.linspace(0, len(moving) - 1, 12, dtype=int)]
        # Translation makes browser values compact; the model is translation invariant.
        origins = history[selected, -1:]
        h, f = history[selected] - origins, future[selected] - origins
        features, cv, c, s = history_features(h, 0.2, model['times'])
        residual = ((features - model['feature_mean']) / model['feature_scale']) @ model['coef'].T
        residual += model['intercept']
        forecasts = {'cv': cv, 'ca': constant_acceleration(h, 0.2, model['times']),
                     'ridge': cv + to_world(residual.reshape(-1, 6, 2), c, s)}
        for row, index in enumerate(selected):
            examples.append({
                'town': town, 'window_index': int(index), 'class_id': int(labels[index]),
                'history': np.round(h[row], 4).tolist(), 'future': np.round(f[row], 4).tolist(),
                'forecasts': {name: np.round(pred[row], 4).tolist() for name, pred in forecasts.items()},
            })
    payload = {'history_step_s': 0.2, 'future_times_s': model['times'].tolist(),
               'coordinates': 'world XY in metres, translated to the last observed position',
               'selection': '12 evenly spaced indices within moving windows of each test town',
               'sources_sha256': sources, 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
               'examples': examples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'{len(examples)} examples -> {args.out}')


if __name__ == '__main__':
    main()
