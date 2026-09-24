#!/usr/bin/env python3
"""Prediction-accuracy evaluation (ADE / FDE) on an analytic scenario bank.

Rolls the physics :class:`MotionPredictor` forward on a small, self-contained
bank of ground-truth manoeuvres and scores it with the standard displacement
metrics (ADE / FDE / minADE / minFDE / miss-rate). The bank deliberately mixes
cases the constant-velocity / constant-turn-rate models fit exactly (straight,
steady turn) with cases they provably cannot (acceleration, lane change), so the
report shows *honest* non-zero errors instead of a rigged perfect score.

    python tools/eval_prediction.py
    python tools/eval_prediction.py --mode cv --report pred_metrics.json
    python tools/eval_prediction.py --horizon 4 --step 0.5 --miss-threshold 2.0
"""
from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np

from perception_core.common.types import Box3D, ObjectClass, Track, TrackState
from perception_core.eval.prediction_metrics import PredictionSample, evaluate_prediction
from perception_core.prediction.physics import MotionPredictor, PredictorConfig

Scenario = Tuple[str, Track, np.ndarray, str]  # (name, track, gt_future (H,2), note)


def _timesteps(horizon: float, step: float) -> np.ndarray:
    n = max(1, int(round(horizon / step)))
    return np.arange(1, n + 1) * step


def _track(x: float, y: float, vx: float, vy: float, yaw_rate: float) -> Track:
    return Track(
        track_id=0, box=Box3D(x, y, 0.0, 4.5, 1.9, 1.5, float(np.arctan2(vy, vx))),
        velocity=np.array([vx, vy], dtype=float), yaw_rate=yaw_rate,
        label=ObjectClass.CAR, state=TrackState.CONFIRMED,
    )


def build_scenarios(horizon: float, step: float) -> List[Scenario]:
    """A mix of manoeuvres the physics models fit exactly and ones they can't."""
    t = _timesteps(horizon, step)
    scenarios: List[Scenario] = []

    # 1. Straight, constant velocity -- CV fits exactly.
    v = 8.0
    gt = np.column_stack([v * t, np.zeros_like(t)])
    scenarios.append(("straight", _track(0, 0, v, 0, 0.0), gt, "CV exact"))

    # 2. Steady turn, constant turn rate -- CTRV fits exactly.
    v, w = 8.0, 0.25
    yaw = w * t
    gt = np.column_stack([v / w * np.sin(yaw), v / w * (1.0 - np.cos(yaw))])
    scenarios.append(("steady_turn", _track(0, 0, v, 0, w), gt, "CTRV exact"))

    # 3. Constant acceleration -- CV holds v0, GT keeps speeding up (unmodelled).
    v0, a = 6.0, 2.5
    gt = np.column_stack([v0 * t + 0.5 * a * t ** 2, np.zeros_like(t)])
    scenarios.append(("accelerate", _track(0, 0, v0, 0, 0.0), gt,
                      f"unmodelled a={a} m/s^2"))

    # 4. Lane change -- lateral half-cosine shift; heading rate ~0 at t=0 so the
    #    physics models see a straight line and miss the manoeuvre entirely.
    v, lane, dur = 12.0, 3.5, min(3.0, horizon)
    prog = np.clip(t / dur, 0.0, 1.0)
    gt = np.column_stack([v * t, lane * 0.5 * (1.0 - np.cos(np.pi * prog))])
    scenarios.append(("lane_change", _track(0, 0, v, 0, 0.0), gt,
                      f"unmodelled {lane} m lateral shift"))

    return scenarios


def evaluate(scenarios: List[Scenario], cfg: PredictorConfig,
             miss_threshold: float) -> Tuple[List[dict], dict]:
    predictor = MotionPredictor(cfg)
    rows: List[dict] = []
    samples: List[PredictionSample] = []
    for name, track, gt, note in scenarios:
        obj = predictor.predict(track)
        sample = PredictionSample(obj.trajectories, gt)
        m = evaluate_prediction([sample], miss_threshold=miss_threshold)
        rows.append({"scenario": name, "note": note, **m.as_dict()})
        samples.append(sample)
    overall = evaluate_prediction(samples, miss_threshold=miss_threshold).as_dict()
    return rows, overall


def format_report(rows: List[dict], overall: dict, cfg: PredictorConfig) -> str:
    lines = [
        "Motion-prediction accuracy (physics baseline)",
        f"  mode={cfg.mode}  horizon={cfg.horizon}s  step={cfg.step}s  "
        f"({overall['horizon_steps']} waypoints)",
        "",
        f"  {'scenario':<14}{'ADE':>8}{'FDE':>8}{'minADE':>8}{'minFDE':>8}"
        f"{'miss':>7}{'modes':>7}   note",
        "  " + "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"  {r['scenario']:<14}{r['ade']:>8.3f}{r['fde']:>8.3f}"
            f"{r['min_ade']:>8.3f}{r['min_fde']:>8.3f}{r['miss_rate']:>7.2f}"
            f"{r['modes']:>7.1f}   {r['note']}"
        )
    lines += [
        "  " + "-" * 78,
        f"  {'OVERALL':<14}{overall['ade']:>8.3f}{overall['fde']:>8.3f}"
        f"{overall['min_ade']:>8.3f}{overall['min_fde']:>8.3f}"
        f"{overall['miss_rate']:>7.2f}{overall['modes']:>7.1f}"
        f"   {overall['samples']} scenarios",
        "",
        "  ADE/FDE in metres (lower is better). Straight/steady_turn ~0 confirm the",
        "  models are exact when their assumptions hold; accelerate/lane_change carry",
        "  honest error where constant-velocity/turn-rate physics cannot comply.",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["cv", "ctrv", "auto"], default="auto",
                   help="predictor motion model (default: auto picks CV/CTRV per track)")
    p.add_argument("--horizon", type=float, default=3.0, help="forecast horizon (s)")
    p.add_argument("--step", type=float, default=0.5, help="forecast resolution (s)")
    p.add_argument("--miss-threshold", type=float, default=2.0,
                   help="best-mode FDE (m) above which a sample counts as a miss")
    p.add_argument("--report", metavar="PATH", help="write the result as JSON to PATH")
    args = p.parse_args()

    cfg = PredictorConfig(horizon=args.horizon, step=args.step, mode=args.mode)
    scenarios = build_scenarios(args.horizon, args.step)
    rows, overall = evaluate(scenarios, cfg, args.miss_threshold)
    print(format_report(rows, overall, cfg))

    if args.report:
        payload = {
            "config": {"mode": cfg.mode, "horizon": cfg.horizon, "step": cfg.step,
                       "miss_threshold": args.miss_threshold},
            "per_scenario": rows, "overall": overall,
        }
        with open(args.report, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
