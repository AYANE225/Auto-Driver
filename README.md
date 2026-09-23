# Auto-Driver — Autonomous-Driving Perception, Tracking & Prediction

![CI](https://github.com/AYANE225/Auto-Driver/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![ROS 2](https://img.shields.io/badge/ROS%202-Humble-22314E)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end **perception → sensor fusion → multi-object tracking → motion
prediction** stack for autonomous driving, packaged as **ROS 2 Humble** nodes and
fed by **CARLA** simulation data. The emphasis is *engineering & systems
integration*: a clean, framework-agnostic core, swappable detector backends,
unit tests, CI, and a one-command visual demo.

<p align="center">
  <img src="docs/screenshots/demo_lidar.gif" width="640" alt="Bird's-eye-view perception demo"/>
</p>

> Bird's-eye view of the offline demo: gray LiDAR sweep, green dashed
> ground-truth boxes, colored confirmed tracks with IDs / velocity arrows /
> history trails, and dotted multi-modal trajectory forecasts.

## Design goals

- **Framework-agnostic core.** `perception_core` depends only on
  NumPy / SciPy / scikit-learn — **no** hard ROS, CARLA, or deep-learning
  dependency — so the algorithms are unit-tested on CPU in seconds and reused
  unchanged from a ROS node, a CARLA client, or an offline batch script.
- **Swappable backends (Strategy pattern).** LiDAR clustering, an optional YOLO
  camera detector, and a ground-truth replay all implement one `Detector`
  interface; the pipeline never changes.
- **Fail-safe fusion.** Camera-LiDAR fusion degrades gracefully to LiDAR-only
  when no calibrated camera is present.
- **Honest evaluation.** Built-in CLEAR-MOT metrics (MOTA / MOTP / ID-switches /
  precision / recall) so results are measured, not asserted.

## Architecture

```
                        ┌──────────────────────── perception_core (pure Python) ────────────────────────┐
 CARLA / rosbag2        │                                                                                │
  LiDAR + cameras  ──►  │  Detection ──► (Camera-LiDAR Fusion) ──► Tracking ──► Prediction ──► Output    │
  (sensor_msgs)         │  RANSAC+DBSCAN     2D IoU label          KF + Hungarian   CV / CTRV             │
                        │  +PCA box / YOLO   transfer              AB3DMOT lifecycle multi-modal          │
                        └───────────────────────────────────┬────────────────────────────────────────────┘
                                                             │
              ROS 2 Humble nodes (rclpy)  ◄──────────────────┘   RViz2 / Foxglove MarkerArray viz
```

Detections are lifted into the world frame via the ego pose before tracking, so
track states and forecasts live in a stable global frame even while the ego
vehicle moves.

## Results (offline demo, LiDAR detector)

60-frame synthetic sequence, 5 actors (cars, a turning car, a truck, a
pedestrian), classical LiDAR detector — matched to ground truth by BEV IoU:

| Metric      | Value | | Metric        | Value |
|-------------|-------|-|---------------|-------|
| Precision   | 0.95  | | MOTA          | 0.777 |
| Recall      | 0.823 | | MOTP (IoU)    | 0.772 |
| ID switches | 1     | | Frames        | 60    |

Reproduce: `python tools/run_demo.py --frames 60 --detector lidar --report metrics.json`

## Real-data validation (KITTI raw)

The same pipeline runs **unchanged** on the real KITTI raw dataset (64-beam
Velodyne + camera + OXTS ego-motion). A `KittiRawReader` turns a drive into the
identical `Frame` objects, so nothing in detection / fusion / tracking /
prediction changes — only the data source. Numbers below are on drive
`2011_09_26_0014` (314 frames, 1142 tracklet poses, dense city traffic).

**1. Tracking & prediction quality (GT-replay).** Replaying the tracklet labels
as noisy detections isolates the tracking + prediction stages on *real* ego
motion — the world-frame MOT stays locked through the car's actual turns:

| Metric      | Value | | Metric           | Value    |
|-------------|-------|-|------------------|----------|
| MOTA        | 0.641 | | Precision        | 0.795    |
| MOTP (IoU)  | 0.822 | | Recall           | 0.864    |
| ID switches | **1** | | Frames / objects | 314/1142 |

<p align="center">
  <img src="docs/screenshots/demo_kitti.gif" width="640" alt="KITTI raw BEV perception demo"/>
</p>

**2. Detection on real clutter — why fusion matters.** The classical geometric
detector, tuned on clean synthetic scenes, is *flooded* by urban clutter
(buildings, vegetation and poles all form car-sized clusters). Wiring in a stock
**YOLO camera detector** as a camera-LiDAR **fusion gate** — drop any LiDAR
cluster inside the image that no camera detection confirms — is a *learned*
detector rescuing precision on real data. Scored inside the camera frustum
(KITTI's standard annotation region):

| Detector (camera FOV)          | Precision | Recall | False positives | MOTA   |
|--------------------------------|-----------|--------|-----------------|--------|
| Classical LiDAR only           | 0.051     | 0.251  | 5198            | −4.40  |
| **+ YOLO camera-LiDAR gate**   | **0.533** | 0.181  | **177**         | **+0.02** |

The camera gate cuts false positives by **97 %** and lifts precision **10×**,
flipping MOTA positive. Recall drops — the gate can only *reject*, never add, and
is capped by the geometric clusterer's own recall. That ceiling is precisely why
production stacks reach for a learned 3D detector; the pluggable `Detector`
interface exists so one drops straight in.

Reproduce:

```bash
# tracking/prediction on real data (GT-replay) + BEV GIF
python tools/run_kitti_demo.py --root data/kitti --drive 14 --detector gt \
    --gif docs/screenshots/demo_kitti.gif --report metrics_kitti.json
# classical detector vs. YOLO camera-LiDAR fusion gate (needs the [yolo] extra + a torch env)
python tools/run_kitti_demo.py --root data/kitti --drive 14 --detector lidar --fov-eval
python tools/run_kitti_demo.py --root data/kitti --drive 14 --detector fusion
```

## Component summary

| Stage      | Default implementation                                    | Key deps            |
|------------|-----------------------------------------------------------|---------------------|
| Detection  | `LidarClusterDetector` — RANSAC ground + DBSCAN + PCA box | scikit-learn        |
| Detection  | `YoloCameraDetector` — optional 2D camera detector        | ultralytics *(opt)* |
| Detection  | `GroundTruthDetector` — replay for tests / CI / demo      | –                   |
| Fusion     | `LateFusion` — project 3D→image, 2D-IoU label transfer    | –                   |
| Tracking   | `MultiObjectTracker` — constant-velocity KF + Hungarian   | scipy               |
| Prediction | `MotionPredictor` — CV / CTRV, multi-modal                | –                   |

## Repository layout

```
carla_av_perception/
├── src/perception_core/     # framework-agnostic core library (+ unit tests)
│   └── perception_core/     #   common · detection · fusion · tracking · prediction · io · eval · viz
├── src/av_perception_msgs/  # ROS 2 custom interfaces (tracked / predicted objects)
├── src/av_perception/       # ROS 2 rclpy nodes wrapping perception_core
├── src/av_bringup/          # ROS 2 launch files, params, RViz config
├── tools/run_demo.py        # offline end-to-end demo + BEV GIF + metrics
├── tools/run_kitti_demo.py  # same pipeline on real KITTI raw + honest CLEAR-MOT
├── docs/                    # screenshots, architecture notes
└── .github/workflows/ci.yml # test (py3.9–3.11) · lint · smoke matrix
```

## Status

| Layer | Description | Status |
|-------|-------------|--------|
| `perception_core` | Detection / fusion / tracking / prediction + 41 unit tests | ✅ done |
| Offline demo + CI | BEV renderer, GIF, CLEAR-MOT report, GitHub Actions | ✅ done |
| ROS 2 layer | Custom msgs, rclpy nodes, launch + RViz visualization | ✅ done |
| CARLA layer | Sensor bridge, NPC traffic, record → replay dataset | ✅ done |
| Real-data (KITTI) | `KittiRawReader`, GT-replay tracking, YOLO camera-LiDAR fusion gate | ✅ done |

## Quickstart

```bash
# 1. Install the core library
pip install -e "src/perception_core[viz]"

# 2. Run the offline end-to-end demo (writes a BEV GIF + a metrics report)
python tools/run_demo.py --frames 60 --detector lidar \
    --gif docs/screenshots/demo_lidar.gif --report metrics.json

# 3. Run the unit tests
pip install -e "src/perception_core[test]"
pytest -q src/perception_core
```

> Inside a **sourced ROS 2 environment**, disable the incompatible ROS pytest
> plugins first: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`.

### Run the ROS 2 graph

```bash
# Build the workspace (ROS 2 Humble sourced)
colcon build --packages-select av_perception_msgs av_perception av_bringup
source install/setup.bash

# Launch: synthetic LiDAR publisher + perception node + RViz (no CARLA needed)
ros2 launch av_bringup perception.launch.py            # add rviz:=false for headless
```

The graph publishes `~/tracks` (`TrackedObjectArray`), `~/predictions`
(`PredictedObjectArray`) and `~/markers` (`MarkerArray` for RViz). Point the
`perception_node` at a real source by remapping `/lidar/points`.

### Feed it CARLA data

```bash
# See carla/README.md — record a scenario from a CARLA server, then either
# replay it offline through the same pipeline ...
python carla/replay_demo.py --dataset carla/data/urban --gif docs/screenshots/demo_carla.gif
# ... or republish it into the ROS 2 graph as /lidar/points.
```


## Tech stack

Python · NumPy · SciPy · scikit-learn · ROS 2 Humble (rclpy) · RViz2 ·
rosbag2 · CARLA · Matplotlib · pytest · GitHub Actions. Optional: ultralytics
(YOLO), Open3D.

## License

MIT — see [LICENSE](LICENSE).

