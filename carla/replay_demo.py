#!/usr/bin/env python3
"""Replay a recorded CARLA scenario through the perception pipeline offline.

Reuses the exact same :class:`~perception_core.pipeline.PerceptionPipeline` as
the synthetic demo — only the data source changes — then renders a BEV GIF and
reports CLEAR-MOT metrics against the CARLA ground truth. Runs with NO CARLA and
NO ROS installed (needs only perception_core + the recorded dataset).

    python replay_demo.py --dataset data/urban --gif urban.gif --report urban.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from the repo without installing this folder as a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CarlaDataset  # noqa: E402

from perception_core.eval.metrics import evaluate_tracking  # noqa: E402
from perception_core.pipeline import PerceptionPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="recorded scenario directory")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--gif", default=None, help="path for the BEV animation")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--report", default=None, help="write metrics JSON here")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    ds = CarlaDataset(args.dataset)
    pipe = PerceptionPipeline()

    renderer = None
    if not args.no_video:
        from perception_core.viz.bev import BevRenderer
        renderer = BevRenderer()

    gt_frames, track_frames, images = [], [], []
    n = len(ds) if args.max_frames is None else min(len(ds), args.max_frames)
    for i in range(n):
        frame = ds.read_frame(i)
        out = pipe.process(frame)
        gt_frames.append(frame.ground_truth or [])
        track_frames.append(out.tracks)
        if renderer is not None:
            images.append(renderer.draw(frame, out))

    metrics = evaluate_tracking(gt_frames, track_frames, iou_threshold=0.3)
    report = {"dataset": args.dataset, "frames": n, **metrics.as_dict()}
    print("=== tracking metrics (CARLA replay) ===")
    for k, v in report.items():
        print(f"  {k:12s}: {v}")

    if renderer is not None and images:
        from perception_core.viz.bev import save_gif
        gif_path = args.gif or os.path.join(args.dataset, "replay.gif")
        save_gif(images, gif_path, fps=args.fps)
        renderer.close()
        print(f"wrote {len(images)} frames -> {gif_path}")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote report -> {args.report}")


if __name__ == "__main__":
    main()
