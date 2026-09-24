#!/usr/bin/env python3
"""Camera-only demo: public VGGT pseudo-LiDAR -> perception pipeline -> BEV GIF.

Feeds a folder of images through the **public** VGGT model to synthesise a
pseudo-LiDAR sweep per frame, then runs the *identical* detect -> track ->
predict stack used for real LiDAR. This showcases the pluggable detector
front-end: swap the sensor, keep the pipeline.

Requires the 'vggt' extra + (ideally) a CUDA GPU -- it is intentionally NOT part
of CI, which stays torch-free:

    pip install 'src/perception_core[vggt,viz]'
    python tools/run_vggt_demo.py --images path/to/frames \
        --gif outputs/vggt_demo.gif --tracker imm --scale 1.0
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np

from perception_core.common.types import Frame
from perception_core.detection.vggt import VggtConfig, VggtLidarDetector
from perception_core.pipeline import PerceptionPipeline, PipelineConfig
from perception_core.tracking.mot import TrackerConfig

_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def load_images(folder: str, max_frames: int) -> List[np.ndarray]:
    from PIL import Image  # lazy: only needed here

    paths: List[str] = []
    for ext in _EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(paths)[:max_frames] if max_frames > 0 else sorted(paths)
    if not paths:
        raise FileNotFoundError(f"no images ({'/'.join(_EXTS)}) found under {folder!r}")
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of sequential camera frames")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.1, help="assumed inter-frame time (s)")
    ap.add_argument("--tracker", choices=["cv", "imm"], default="cv")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="metric scale for the (up-to-scale) monocular geometry")
    ap.add_argument("--conf-percentile", type=float, default=50.0,
                    help="drop VGGT points below this confidence percentile")
    ap.add_argument("--checkpoint", default="facebook/VGGT-1B")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--gif", default=None, help="output GIF path (default <out>/vggt_demo.gif)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--no-video", action="store_true", help="run the pipeline without rendering")
    args = ap.parse_args()

    images = load_images(args.images, args.max_frames)
    detector = VggtLidarDetector(config=VggtConfig(
        checkpoint=args.checkpoint, scale=args.scale, conf_percentile=args.conf_percentile,
    ))
    pipe = PerceptionPipeline(
        detector=detector,
        config=PipelineConfig(tracker=TrackerConfig(motion_model=args.tracker)),
    )

    renderer = None
    if not args.no_video:
        from perception_core.viz.bev import BevRenderer  # lazy: matplotlib only if rendering
        renderer = BevRenderer()

    rendered, n_tracks = [], 0
    for k, image in enumerate(images):
        frame = Frame(timestamp=k * args.dt, frame_id=k, images={"cam_front": image})
        out = pipe.process(frame)
        n_tracks = max(n_tracks, len(out.tracks))
        print(f"frame {k:3d}: {len(out.detections)} det -> {len(out.tracks)} tracks "
              f"| {pipe.last_timings['total_ms']:.1f} ms")
        if renderer is not None:
            rendered.append(renderer.draw(frame, out, show_gt=False))

    print(f"processed {len(images)} frames, peak {n_tracks} concurrent tracks")
    if renderer is not None:
        from perception_core.viz.bev import save_gif

        os.makedirs(args.out, exist_ok=True)
        gif_path = args.gif or os.path.join(args.out, "vggt_demo.gif")
        save_gif(rendered, gif_path, fps=args.fps)
        renderer.close()
        print(f"wrote {len(rendered)} frames -> {gif_path}")


if __name__ == "__main__":
    main()
