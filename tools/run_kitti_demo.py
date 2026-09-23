#!/usr/bin/env python3
"""Offline end-to-end demo on the KITTI raw dataset (real Velodyne + tracklets).

Runs the SAME detection -> tracking -> prediction pipeline used on synthetic and
CARLA data over a real KITTI raw drive, renders a bird's-eye-view animation in
the ego frame and reports honest CLEAR-MOT metrics against the tracklet labels.

    python tools/run_kitti_demo.py --root data/kitti --date 2011_09_26 --drive 14 \
        --frames 150 --gif docs/screenshots/demo_kitti.gif --report metrics_kitti.json
"""
from __future__ import annotations

import argparse
import json
import os
from copy import copy
from typing import List

import numpy as np

from perception_core.common.geometry import invert_se3, transform_box
from perception_core.common.types import (
    Detection,
    ObjectClass,
    PerceptionOutput,
    Trajectory,
    TrajectoryPoint,
)
from perception_core.detection.lidar_cluster import LidarClusterConfig
from perception_core.detection.mock import GroundTruthDetector
from perception_core.eval.metrics import evaluate_tracking
from perception_core.fusion.late_fusion import (
    LateFusion,
    LateFusionConfig,
    project_box_to_image,
)
from perception_core.io.kitti import KittiRawReader
from perception_core.pipeline import PerceptionPipeline, PipelineConfig

# Road users a stock COCO-trained YOLO can recognise (drives the fusion gate).
_VEHICLE_CLASSES = [
    ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS,
    ObjectClass.MOTORCYCLE, ObjectClass.BICYCLE, ObjectClass.PEDESTRIAN,
]


def kitti_pipeline(detector: str, gate_iou: float = 0.1) -> PerceptionPipeline:
    """Pipeline for a real 64-beam Velodyne (ground ~-1.73 m, dense, clutter).

    ``lidar``: the classical geometric detector — honest but flooded by urban
    clutter (buildings/vegetation). ``fusion``: gate those LiDAR clusters with a
    stock YOLO camera detector (drop in-image boxes no camera confirms) — a
    *learned* detector rescuing precision on real data. ``gt``: replay tracklets
    with localisation noise to isolate the tracking + prediction stages.
    """
    lidar = LidarClusterConfig(
        x_range=(-45.0, 45.0), y_range=(-30.0, 30.0), z_range=(-2.4, 1.0),
        ground_dist_thresh=0.25, dbscan_eps=0.6, dbscan_min_samples=15,
        min_points=15, min_extent=0.3, max_extent=8.0, max_height=4.0,
    )
    cfg = PipelineConfig(lidar=lidar)
    if detector == "gt":
        return PerceptionPipeline(
            detector=GroundTruthDetector(position_noise=0.2, yaw_noise=0.03,
                                         dropout=0.05, seed=0),
            config=cfg,
        )
    if detector == "fusion":
        from perception_core.detection.yolo import YoloCameraDetector, YoloConfig
        cfg.fusion = LateFusionConfig(iou_threshold=gate_iou, require_camera=True)
        yolo = YoloCameraDetector(YoloConfig(conf=0.25, keep=_VEHICLE_CLASSES))
        return PerceptionPipeline(camera_detector=yolo, config=cfg)
    return PerceptionPipeline(config=cfg)


def _detections_to_local(dets: List[Detection], T: np.ndarray) -> List[Detection]:
    return [
        Detection(box=transform_box(d.box, T), score=d.score, label=d.label,
                  source=d.source, attributes=d.attributes)
        for d in dets
    ]


def _output_to_local(out: PerceptionOutput, T: np.ndarray) -> PerceptionOutput:
    """Express world-frame tracks/forecasts in the current ego frame for display."""
    R = T[:3, :3]
    tracks = []
    for tr in out.tracks:
        t2 = copy(tr)
        t2.box = transform_box(tr.box, T)
        t2.velocity = R[:2, :2] @ np.asarray(tr.velocity, dtype=float)
        if tr.history:
            h = np.array([[p[0], p[1], 0.0] for p in tr.history])
            t2.history = [row[:2] for row in (h @ R.T + T[:3, 3])]
        tracks.append(t2)
    preds = []
    for p in out.predictions:
        p2 = copy(p)
        p2.current_box = transform_box(p.current_box, T)
        p2.trajectories = []
        for traj in p.trajectories:
            pts = []
            for tp in traj.points:
                q = R @ np.array([tp.x, tp.y, 0.0]) + T[:3, 3]
                pts.append(TrajectoryPoint(t=tp.t, x=float(q[0]), y=float(q[1])))
            p2.trajectories.append(Trajectory(points=pts, confidence=traj.confidence, mode=traj.mode))
        preds.append(p2)
    return PerceptionOutput(timestamp=out.timestamp, frame_id=out.frame_id,
                            detections=[], tracks=tracks, predictions=preds)


def _fov_mask(boxes, ego: np.ndarray, calib, cam: str = "cam2") -> List[bool]:
    """True for world-frame boxes that project into the camera image.

    KITTI's own object benchmark only annotates objects visible in the camera,
    so restricting evaluation to this frustum is the standard, fair comparison
    for a camera-gated detector (rather than penalising it for the 360deg LiDAR
    field of view the forward camera cannot see).
    """
    if not calib.has_camera(cam):
        return [True] * len(boxes)
    T_inv = invert_se3(ego)
    mask = []
    for b in boxes:
        local = transform_box(b, T_inv)
        box2d = project_box_to_image(local, calib.lidar_to_cam[cam],
                                     calib.intrinsics[cam], calib.image_size.get(cam))
        mask.append(box2d is not None)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/kitti", help="dir holding <date>/")
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drive", type=int, default=14)
    ap.add_argument("--frames", type=int, default=-1, help="limit frames (-1 = all)")
    ap.add_argument("--detector", choices=["lidar", "fusion", "gt"], default="lidar",
                    help="lidar = classical geometric; fusion = YOLO camera-LiDAR "
                         "gate (learned); gt = replay tracklets (isolate tracking)")
    ap.add_argument("--min-range", type=float, default=3.0, help="drop returns closer than this")
    ap.add_argument("--iou", type=float, default=0.3, help="BEV IoU match threshold")
    ap.add_argument("--gate-iou", dest="gate_iou", type=float, default=0.1,
                    help="min camera<->LiDAR 2D IoU for the fusion gate to keep a box")
    ap.add_argument("--fov-eval", dest="fov_eval", action="store_true",
                    help="score only inside the camera frustum (KITTI convention; "
                         "auto-on for --detector fusion)")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    fov_eval = args.fov_eval or args.detector == "fusion"
    reader = KittiRawReader(args.root, args.date, args.drive,
                            load_images=args.detector == "fusion")
    n = len(reader) if args.frames < 0 else min(args.frames, len(reader))
    pipe = kitti_pipeline(args.detector, gate_iou=args.gate_iou)

    renderer = None
    if not args.no_video:
        from perception_core.viz.bev import BevRenderer
        renderer = BevRenderer(xlim=(-30.0, 50.0), ylim=(-30.0, 30.0))

    gt_frames, gt_id_frames, track_frames, images = [], [], [], []
    for i in range(n):
        frame = reader.read_frame(i)
        ego = frame.ego_pose
        if args.min_range > 0:
            r = np.hypot(frame.lidar[:, 0], frame.lidar[:, 1])
            frame.lidar = frame.lidar[r > args.min_range]

        gt = frame.ground_truth or []
        gt_ids = [int(d.attributes.get("gt_id", -1)) for d in gt]
        if args.detector == "gt":
            # Ground truth is already in the world frame; stop the pipeline from
            # lifting the replayed detections a second time.
            frame.ego_pose = None
        out = pipe.process(frame)
        tracks = list(out.tracks)

        if fov_eval:
            # Restrict scoring to the camera frustum (KITTI's annotation region).
            gm = _fov_mask([d.box for d in gt], ego, frame.calib)
            gt = [d for d, k in zip(gt, gm) if k]
            gt_ids = [i for i, k in zip(gt_ids, gm) if k]
            tm = _fov_mask([t.box for t in tracks], ego, frame.calib)
            tracks = [t for t, k in zip(tracks, tm) if k]

        gt_frames.append(gt)
        gt_id_frames.append(gt_ids)
        track_frames.append(tracks)

        if renderer is not None:
            T_inv = invert_se3(ego)
            frame.ground_truth = _detections_to_local(gt, T_inv)
            images.append(renderer.draw(frame, _output_to_local(out, T_inv)))

    metrics = evaluate_tracking(gt_frames, track_frames, iou_threshold=args.iou,
                                gt_id_frames=gt_id_frames)
    report = {"dataset": "kitti_raw", "date": args.date, "drive": args.drive,
              "detector": args.detector, "fov_eval": fov_eval,
              "frames": n, **metrics.as_dict()}
    print("=== KITTI tracking metrics (real data) ===")
    for k, v in report.items():
        print(f"  {k:12s}: {v}")

    if renderer is not None and args.gif:
        from perception_core.viz.bev import save_gif
        os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
        save_gif(images, args.gif, fps=args.fps)
        renderer.close()
        print(f"wrote {len(images)} frames -> {args.gif}")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote report -> {args.report}")


if __name__ == "__main__":
    main()
