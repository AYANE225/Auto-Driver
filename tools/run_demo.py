#!/usr/bin/env python3
"""Offline end-to-end demo: run the perception pipeline on a synthetic driving
scene, render a bird's-eye-view animation and report tracking metrics.

Runs anywhere perception_core + matplotlib are installed (no ROS / CARLA / GPU).

    python tools/run_demo.py --frames 60 --out outputs --gif outputs/demo.gif
    python tools/run_demo.py --frames 20 --no-video --report report.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from perception_core.detection.mock import GroundTruthDetector
from perception_core.eval.metrics import evaluate_tracking
from perception_core.io.synthetic import generate_frames, make_default_scene
from perception_core.pipeline import PerceptionPipeline


def build_pipeline(detector: str) -> PerceptionPipeline:
    if detector == "gt":
        return PerceptionPipeline(detector=GroundTruthDetector(position_noise=0.1, seed=0))
    return PerceptionPipeline()  # default: classical LiDAR clustering


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--detector", choices=["lidar", "gt"], default="lidar")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--gif", default=None, help="path for the animated GIF (default <out>/demo.gif)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--no-video", action="store_true", help="skip rendering (metrics only)")
    ap.add_argument("--report", default=None, help="write the metrics report to this JSON path")
    args = ap.parse_args()

    frames = generate_frames(make_default_scene(), num_frames=args.frames, dt=args.dt, seed=args.seed)
    pipe = build_pipeline(args.detector)

    gt_frames, track_frames, images = [], [], []
    renderer = None
    if not args.no_video:
        from perception_core.viz.bev import BevRenderer  # lazy: matplotlib only if rendering
        renderer = BevRenderer()

    for frame in frames:
        out = pipe.process(frame)
        gt_frames.append(frame.ground_truth or [])
        track_frames.append(out.tracks)
        if renderer is not None:
            images.append(renderer.draw(frame, out))

    metrics = evaluate_tracking(gt_frames, track_frames, iou_threshold=0.3)
    report = {"detector": args.detector, "frames": args.frames, "dt": args.dt, **metrics.as_dict()}
    print("=== tracking metrics ===")
    for k, v in report.items():
        print(f"  {k:12s}: {v}")

    if renderer is not None:
        from perception_core.viz.bev import save_gif

        os.makedirs(args.out, exist_ok=True)
        gif_path = args.gif or os.path.join(args.out, "demo.gif")
        save_gif(images, gif_path, fps=args.fps)
        renderer.close()
        print(f"wrote {len(images)} frames -> {gif_path}")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote report -> {args.report}")


if __name__ == "__main__":
    main()
