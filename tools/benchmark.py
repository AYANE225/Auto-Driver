#!/usr/bin/env python3
"""Latency / real-time benchmark for the perception pipeline.

Runs the full ``detect -> fuse -> track -> predict`` pipeline over a synthetic
sequence and reports per-stage latency (mean / median / p95, in ms), end-to-end
latency and throughput (Hz). Optionally asserts a real-time budget so it can be
wired into CI as a performance gate.

    python tools/benchmark.py --frames 200
    python tools/benchmark.py --frames 200 --actors 20 --tracker imm
    python tools/benchmark.py --frames 100 --detector gt --budget-ms 100
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

from perception_core.detection.mock import GroundTruthDetector
from perception_core.io.synthetic import Actor, generate_frames, make_default_scene
from perception_core.pipeline import PerceptionPipeline, PipelineConfig
from perception_core.tracking.mot import TrackerConfig

_STAGES = ["detect_ms", "fuse_ms", "track_ms", "predict_ms", "total_ms"]


def make_scene(n_actors: int) -> List[Actor]:
    """Tile the default scene up to ``n_actors`` by spatially offsetting copies."""
    base = make_default_scene()
    actors: List[Actor] = []
    i = 0
    while len(actors) < n_actors:
        proto = base[i % len(base)]
        row = i // len(base)
        actors.append(Actor(
            x=proto.x + 6.0 * (i % len(base)) - 12.0, y=proto.y + 9.0 * row - 9.0,
            yaw=proto.yaw, speed=proto.speed, label=proto.label,
            l=proto.l, w=proto.w, h=proto.h, yaw_rate=proto.yaw_rate,
        ))
        i += 1
    return actors[:n_actors]


def build_pipeline(detector: str, tracker: str) -> PerceptionPipeline:
    cfg = PipelineConfig(tracker=TrackerConfig(motion_model=tracker))
    det = GroundTruthDetector(position_noise=0.1, seed=0) if detector == "gt" else None
    return PerceptionPipeline(detector=det, config=cfg)


def summarise(samples: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for stage, vals in samples.items():
        arr = np.asarray(vals)
        out[stage] = {
            "mean": round(float(arr.mean()), 3),
            "median": round(float(np.median(arr)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "max": round(float(arr.max()), 3),
        }
    return out


def run_benchmark(frames: int, actors: int, detector: str, tracker: str,
                  warmup: int, dt: float, seed: int) -> Dict:
    scene = make_scene(actors)
    seq = generate_frames(scene, num_frames=frames + warmup, dt=dt, seed=seed)
    pipe = build_pipeline(detector, tracker)

    samples: Dict[str, List[float]] = {s: [] for s in _STAGES}
    for k, frame in enumerate(seq):
        pipe.process(frame)
        if k < warmup:            # exclude cold-start frames (imports, JIT, cache warmup)
            continue
        for stage in _STAGES:
            samples[stage].append(pipe.last_timings[stage])

    stats = summarise(samples)
    total = stats["total_ms"]
    return {
        "config": {
            "frames": frames, "warmup": warmup, "actors": actors,
            "detector": detector, "tracker": tracker, "dt": dt,
        },
        "latency_ms": stats,
        "throughput_hz": round(1000.0 / total["mean"], 1) if total["mean"] > 0 else 0.0,
        "realtime_factor": round(dt * 1000.0 / total["mean"], 1) if total["mean"] > 0 else 0.0,
    }


def format_report(result: Dict) -> str:
    cfg, lat = result["config"], result["latency_ms"]
    lines = [
        "Perception pipeline latency benchmark",
        f"  detector={cfg['detector']}  tracker={cfg['tracker']}  "
        f"actors={cfg['actors']}  frames={cfg['frames']} (+{cfg['warmup']} warmup)",
        "",
        f"  {'stage':<12}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}   (ms)",
        "  " + "-" * 56,
    ]
    for stage in _STAGES:
        s = lat[stage]
        name = stage[:-3]  # strip "_ms"
        lines.append(f"  {name:<12}{s['mean']:>9.3f}{s['median']:>9.3f}"
                     f"{s['p95']:>9.3f}{s['max']:>9.3f}")
    lines += [
        "  " + "-" * 56,
        f"  throughput   : {result['throughput_hz']:.1f} Hz (mean)",
        f"  realtime x   : {result['realtime_factor']:.1f}x  "
        f"(sim step dt={cfg['dt']}s; >1 means faster than real time)",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=200, help="measured frames (excl. warmup)")
    p.add_argument("--warmup", type=int, default=5, help="cold-start frames to discard")
    p.add_argument("--actors", type=int, default=5, help="number of actors in the scene")
    p.add_argument("--detector", choices=["lidar", "gt"], default="lidar",
                   help="'lidar' = DBSCAN clustering (default), 'gt' = noisy ground truth")
    p.add_argument("--tracker", choices=["cv", "imm"], default="cv",
                   help="motion model: 'cv' constant-velocity or 'imm' CV+CT bank")
    p.add_argument("--dt", type=float, default=0.1, help="sim time step (s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-ms", type=float, default=None,
                   help="assert p95 end-to-end latency <= this (CI perf gate); exit 1 if exceeded")
    p.add_argument("--report", metavar="PATH", help="write the result as JSON to PATH")
    args = p.parse_args()

    result = run_benchmark(args.frames, args.actors, args.detector, args.tracker,
                           args.warmup, args.dt, args.seed)
    print(format_report(result))

    if args.report:
        with open(args.report, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.report}")

    if args.budget_ms is not None:
        p95 = result["latency_ms"]["total_ms"]["p95"]
        ok = p95 <= args.budget_ms
        print(f"\nbudget: p95 total {p95:.3f} ms {'<=' if ok else '>'} "
              f"{args.budget_ms:.3f} ms  ->  {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
