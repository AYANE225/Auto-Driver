#!/usr/bin/env python3
"""Replay CARLA recordings with camera/BEV views and tracking metrics.

    python carla/replay_demo.py --dataset carla/data/urban --detector fusion \
        --tracker imm --fov-eval --view both --gif docs/screenshots/carla_urban.gif

Requires perception_core; YOLO and rendering dependencies are optional. Sensor
recordings are replayed offline, with no running CARLA server or ROS required.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import os
from time import perf_counter

import numpy as np

from perception_core.common.ego_frame import detections_to_frame, output_to_frame
from perception_core.common.geometry import invert_se3, transform_box
from perception_core.detection.mock import GroundTruthDetector
from perception_core.detection.lidar_cluster import LidarClusterConfig
from perception_core.eval import evaluate_hota, evaluate_identity, evaluate_tracking
from perception_core.fusion.late_fusion import LateFusionConfig, project_box_to_image
from perception_core.io.carla_dataset import CarlaDataset
from perception_core.pipeline import PerceptionPipeline, PipelineConfig
from perception_core.tracking.mot import TrackerConfig


def build_pipeline(args):
    cfg = PipelineConfig(lidar=LidarClusterConfig(voxel_size=args.voxel_size),
                         tracker=TrackerConfig(motion_model=args.tracker, gating=args.gating))
    camera_detector = None
    detector = None
    if args.detector == "gt":
        detector = GroundTruthDetector(position_noise=0.1, seed=0)
        cfg.track_in_world = False  # recorded GT boxes are already in world coordinates
    elif args.detector == "fusion":
        from perception_core.detection.yolo import YoloCameraDetector, YoloConfig
        camera_detector = YoloCameraDetector(YoloConfig(model=args.weights, device=args.device))
        cfg.fusion = LateFusionConfig(iou_threshold=args.gate_iou, require_camera=True)
    return PerceptionPipeline(detector=detector, camera_detector=camera_detector, config=cfg)


def evaluation_view(frame, output, lidar_config, camera=None):
    """Select GT and tracks by the same sensor-frame XY region and optional FOV.

    Includes occluded GT within that region; no ground-truth visibility filter
    or label-dependent track suppression is applied. Evaluation is class-agnostic.
    """
    to_sensor = invert_se3(frame.ego_pose) if frame.ego_pose is not None else np.eye(4)

    def keep(box):
        local = transform_box(box, to_sensor)
        if not (lidar_config.x_range[0] <= local.x <= lidar_config.x_range[1]
                and lidar_config.y_range[0] <= local.y <= lidar_config.y_range[1]):
            return False
        if camera is None:
            return True
        return project_box_to_image(local, frame.calib.lidar_to_cam[camera],
                                    frame.calib.intrinsics[camera],
                                    frame.calib.image_size.get(camera)) is not None

    gt = [d for d in (frame.ground_truth or []) if keep(d.box)]
    tracks = [t for t in output.tracks if keep(t.box)]
    track_ids = {t.track_id for t in tracks}
    selected = replace(output, tracks=tracks,
                       predictions=[p for p in output.predictions if p.track_id in track_ids])
    return replace(frame, ground_truth=gt), selected


def render_frame(frame, output, renderer, overlay, args, elapsed):
    from perception_core.viz.camera import side_by_side
    from PIL import Image, ImageDraw

    to_sensor = invert_se3(frame.ego_pose) if frame.ego_pose is not None else np.eye(4)
    local = output_to_frame(output, to_sensor)
    local_frame = replace(frame, ground_truth=detections_to_frame(frame.ground_truth, to_sensor))
    panels = []
    if args.view in ("camera", "both"):
        panels.append(overlay.draw(frame.images[args.camera], local))
    if args.view in ("bev", "both"):
        panels.append(renderer.draw(local_frame, local, all_modes=True, title="LiDAR / tracks / forecast"))
    panel = side_by_side(panels, height=args.height, gap=8)
    # Label the actual detector and playback time; playback rate is not throughput.
    canvas = Image.new("RGB", (panel.shape[1], panel.shape[0] + 48), (16, 20, 24))
    canvas.paste(Image.fromarray(panel), (0, 48))
    draw = ImageDraw.Draw(canvas)
    from perception_core.viz.camera import _load_font
    font = _load_font(16)
    title = (f"CARLA {args.dataset.name} | {args.detector.upper()} + {args.tracker.upper()} | "
             f"t={elapsed:.1f}s | frame {frame.frame_id} | {len(local.tracks)} tracks")
    draw.text((14, 6), title, font=font, fill=(239, 244, 250))
    draw.text((14, 27), "Matching colors: track IDs   |   Lines: motion forecasts   |   Offline replay",
              font=_load_font(13), fill=(184, 195, 208))
    return np.asarray(canvas)


def run_replay(args):
    render = not args.no_video
    needs_images = args.detector == "fusion" or (render and args.view in ("camera", "both"))
    ds = CarlaDataset(str(args.dataset), load_images=needs_images, count_points=False)
    n = len(ds) - args.start_frame
    if args.max_frames is not None:
        n = min(n, args.max_frames)
    if n <= 0:
        raise ValueError("no frames in the requested interval")
    if needs_images and args.camera not in ds.cameras:
        raise ValueError(f"dataset has no camera '{args.camera}'")
    if args.fov_eval or (render and args.view in ("camera", "both")):
        if ds.calib is None or not ds.calib.has_camera(args.camera):
            raise ValueError(f"camera '{args.camera}' needs calibration")

    pipe = build_pipeline(args)
    renderer = overlay = None
    if render:
        if args.view in ("bev", "both"):
            from perception_core.viz.bev import BevRenderer
            renderer = BevRenderer(xlim=(-15, 55), ylim=(-30, 30), figsize=(7, 6))
        if args.view in ("camera", "both"):
            from perception_core.viz.camera import CameraOverlay
            overlay = CameraOverlay(ds.calib, args.camera)

    gt_frames, gt_ids, track_frames, images, timings = [], [], [], [], []
    raw_gt_count = 0
    first_time = None
    t0 = perf_counter()
    try:
        for offset in range(n):
            frame = ds.read_frame(args.start_frame + offset)
            if needs_images and args.camera not in frame.images:
                raise ValueError(f"missing {args.camera} image for frame {frame.frame_id}")
            if first_time is None:
                first_time = frame.timestamp
            out = pipe.process(frame)
            timings.append(pipe.last_timings.copy())
            raw_gt_count += len(frame.ground_truth or [])
            selected_frame, selected_out = evaluation_view(
                frame, out, pipe.cfg.lidar, args.camera if args.fov_eval else None)
            gt_frames.append(selected_frame.ground_truth)
            gt_ids.append([int(d.attributes["gt_id"]) for d in selected_frame.ground_truth])
            track_frames.append(selected_out.tracks)
            if (render and offset % args.render_every == 0
                    and (args.gif_frames == 0 or len(images) < args.gif_frames)):
                images.append(render_frame(selected_frame, selected_out, renderer, overlay,
                                           args, frame.timestamp - first_time))
            if offset % 50 == 0 or offset == n - 1:
                print(f"frame {offset + 1}/{n}: {len(out.detections)} detections, "
                      f"{len(selected_out.tracks)} tracks in evaluation region, "
                      f"{pipe.last_timings['total_ms']:.1f} ms", flush=True)
    finally:
        if renderer is not None:
            renderer.close()
    replay_seconds = perf_counter() - t0

    # The same GT identities and selected tracks feed all three metric families.
    mot = evaluate_tracking(gt_frames, track_frames, args.iou, gt_id_frames=gt_ids)
    hota = evaluate_hota(gt_frames, track_frames, gt_id_frames=gt_ids)
    identity = evaluate_identity(gt_frames, track_frames, gt_id_frames=gt_ids,
                                 iou_threshold=args.id_iou)
    warmup = min(args.warmup, n - 1)
    measured = timings[warmup:]
    latency = {key: {"mean": round(float(np.mean([v[key] for v in measured])), 3),
                     "p95": round(float(np.percentile([v[key] for v in measured], 95)), 3)}
               for key in measured[0]}
    report = {
        "dataset": args.dataset.name, "town": ds.meta.get("town"),
        "carla_version": ds.meta.get("carla_version"),
        "detector": args.detector, "tracker": args.tracker,
        "start_frame": args.start_frame, "dt": ds.dt,
        **mot.as_dict(), "hota": hota.as_dict(), "identity": identity.as_dict(),
        "evaluation": {"similarity": "BEV IoU", "class_agnostic": True,
                       "mot_iou": args.iou, "id_iou": args.id_iou,
                       "hota_alphas": "0.05:0.05:0.95", "fov_camera": args.camera if args.fov_eval else None,
                       "x_range": pipe.cfg.lidar.x_range, "y_range": pipe.cfg.lidar.y_range,
                       "visibility_filter": False, "raw_gt_count": raw_gt_count},
        "pipeline_config": asdict(pipe.cfg),
        "camera_detector": asdict(pipe.camera_detector.cfg) if pipe.camera_detector else None,
        "latency_ms": latency, "latency_warmup_frames": warmup,
        "latency_scope": "pipeline only; excludes IO, evaluation and rendering",
        "replay_seconds": round(replay_seconds, 3),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(),
                        "threads": {key: os.environ.get(key) for key in
                                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}},
        "visualization": {"view": args.view if render else None, "frames": len(images),
                          "render_every": args.render_every,
                          "playback_fps": args.fps or 1 / (ds.dt * args.render_every)},
    }
    print(json.dumps({k: report[k] for k in ("dataset", "frames", "precision", "recall",
                                            "mota", "hota", "identity", "latency_ms")}, indent=2))
    if images:
        from perception_core.viz.bev import save_gif
        path = args.gif or args.dataset / "replay.gif"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_gif(images, str(path), fps=args.fps or 1 / (ds.dt * args.render_every))
        print(f"wrote {len(images)} frames -> {path}", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote report -> {args.report}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--detector", choices=["lidar", "fusion", "gt"], default="lidar")
    ap.add_argument("--tracker", choices=["cv", "imm"], default="imm")
    ap.add_argument("--gating", action="store_true", help="enable Mahalanobis association gating")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO checkpoint for fusion")
    ap.add_argument("--device", default="", help="YOLO device, e.g. cpu or cuda:0")
    ap.add_argument("--gate-iou", type=float, default=0.1)
    ap.add_argument("--voxel-size", type=float, default=0.0,
                    help="weighted DBSCAN voxel size in metres; 0 uses raw points")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--fov-eval", action="store_true", help="score and display only the selected camera FOV")
    ap.add_argument("--iou", type=float, default=0.3, help="MOT matching BEV IoU")
    ap.add_argument("--id-iou", type=float, default=0.5, help="IDF1 matching BEV IoU")
    ap.add_argument("--warmup", type=int, default=5, help="frames excluded from latency statistics only")
    ap.add_argument("--view", choices=["bev", "camera", "both"], default="both")
    ap.add_argument("--height", type=int, default=400, help="rendered panel height in pixels")
    ap.add_argument("--gif", type=Path)
    ap.add_argument("--gif-frames", type=int, default=120, help="maximum rendered frames (0 = all)")
    ap.add_argument("--render-every", type=int, default=1, help="render every Nth frame; track every frame")
    ap.add_argument("--fps", type=float, help="playback FPS; defaults to the recording rate / render-every")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    if (args.start_frame < 0 or args.warmup < 0 or args.gif_frames < 0 or args.render_every < 1
            or args.height < 1 or (args.max_frames is not None and args.max_frames < 1)
            or (args.fps is not None and args.fps <= 0)):
        ap.error("frame counts, image height and FPS must be in their valid positive ranges")
    if not all(0 < value <= 1 for value in (args.iou, args.id_iou, args.gate_iou)):
        ap.error("IoU thresholds must be in (0, 1]")
    if not np.isfinite(args.voxel_size) or args.voxel_size < 0:
        ap.error("voxel size must be finite and nonnegative")
    run_replay(args)


if __name__ == "__main__":
    main()
