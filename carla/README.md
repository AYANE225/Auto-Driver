# CARLA data layer

Record synchronized LiDAR, RGB images, sensor poses and actor ground truth, then
replay them through `perception_core` or ROS 2 without a running CARLA server.
The published urban and highway demos use **CARLA 0.9.16**, 400 frames each at
0.1 s intervals. Recording used a Python 3.12 client; offline replay used Python
3.11. Use a Python client package that matches your server version. The ROS
interpreter version depends on your ROS installation, not on the recording format.

## Contents

| File | Purpose |
|------|---------|
| `record_scenario.py` | Spawn ego, LiDAR, cameras, vehicles and walkers; save synchronized frames |
| `record_trajectories.py` | Record actor states in multiple towns without sensor rendering |
| `dataset.py` | Compatibility import for `perception_core.io.CarlaDataset` |
| `replay_demo.py` | LiDAR / YOLO fusion / GT replay, CV / IMM tracking, camera + BEV GIF and metrics |
| `config/scenarios/*.yaml` | Town, traffic, sensor settings and random seeds |
| `install_carla.sh` | Legacy CARLA 0.9.15 installer requiring Python 3.8; not the environment used for these demos |

## Record a scenario

Run these commands from the repository root, with a matching CARLA server and
client already installed. The demo server was 0.9.16; the legacy installer above
does not recreate that environment.

```bash
# In the CARLA client environment
pip install carla==0.9.16 numpy pyyaml pillow
/path/to/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen -quality-level=Epic

# In another terminal using the same client environment
python carla/record_scenario.py --scenario carla/config/scenarios/urban.yaml \
    --out carla/data/urban --images
python carla/record_scenario.py --scenario carla/config/scenarios/highway.yaml \
    --out carla/data/highway --images
```

Each recording contains `meta.json`, `calib.json`, `frames/000000.npz`, and
optional `frames/000000_front.jpg` images. NPZ fields include `points`, `ego_pose`,
`timestamp`, `gt_boxes`, `gt_labels`, and `gt_ids`. Traffic configuration, spawned
walker count and CARLA version are recorded in metadata. Data and model weights are ignored
by Git; GIFs and metric reports are published in `docs/screenshots/`.

## Replay and evaluate

```bash
pip install -e 'src/perception_core[yolo,viz]'
python carla/replay_demo.py --dataset carla/data/urban --detector fusion \
    --tracker imm --device cuda:0 --fov-eval --render-every 2 --gif-frames 120 \
    --gif docs/screenshots/demo_carla_urban.gif \
    --report docs/screenshots/metrics_carla_urban.json

# CPU-only GT replay, without image/rendering dependencies
python carla/replay_demo.py --dataset carla/data/urban --detector gt \
    --tracker imm --no-video --report outputs/carla_gt.json
```

Use `--device cpu` for YOLO without CUDA. Fusion requires recorded images and
calibration; `--view bev` enables LiDAR-only visualization without camera images.
`--start-frame` and `--max-frames` select the evaluation interval.
`--render-every` and `--gif-frames` limit rendering only: every selected input
frame still updates the tracker and metrics. Default playback follows the sensor
interval and rendering stride, independently of measured processing throughput.

The report includes MOTA/MOTP, HOTA, IDF1, pipeline configuration, evaluation
region and pipeline latency. All metrics use stable CARLA actor IDs and
class-agnostic BEV IoU. GT and tracks share the same XY region and optional camera
FOV. Occluded GT remains included; no LiDAR-point visibility filter is applied.
Use `--voxel-size 0.2` for optional point-count-weighted voxel clustering;
the default `0` clusters raw points. Boxes are fitted to the original points.
See the [comparison protocol](../docs/benchmarks/README.md) for speed and quality
on both recordings, and the [project page](https://ayane225.github.io/Auto-Driver/)
for playable previews.

## ROS 2 replay

After building and sourcing the workspace in a ROS 2 environment:

```bash
colcon build --packages-select av_perception_msgs av_perception av_bringup
source install/setup.bash
ros2 launch av_bringup perception.launch.py source:=carla \
    dataset:=/absolute/path/to/carla/data/highway replay_rate:=2.0 rviz:=false
python tools/check_ros_replay.py --dataset carla/data/highway --frames 8
```

Run the smoke check separately from the launch to avoid duplicate publishers.
The replay node publishes `/lidar/points` in `lidar` and a matching `map <- lidar`
TF at each recorded timestamp. The perception node waits briefly for TF and
publishes tracks, forecasts and markers in `map`. Sensor timestamps are retained
even when wall-clock playback is slower. ROS currently runs the classical LiDAR
detector and IMM tracker; camera fusion is available in the offline replay.

## Coordinate convention

CARLA uses left-handed x-forward, y-right, z-up coordinates. Recording converts
to right-handed x-forward, y-left, z-up by flipping y and negating yaw. LiDAR
points stay in the sensor frame; `ego_pose` is `world <- lidar`; ground-truth
boxes are in world coordinates. Camera calibration maps sensor points to optical
coordinates (x-right, y-down, z-forward). Images use RGB in `Frame`; the YOLO
adapter converts to BGR for Ultralytics NumPy input.

Tracking and forecasting run in world coordinates. Rendering transforms output
copies into the sensor frame, preserving the original track states and using
the same ID colors in camera, BEV and ROS markers.

## Actor trajectories

```bash
python carla/record_trajectories.py --towns Town01_Opt Town10HD_Opt \
    --ticks 3000 --out carla/data/trajectories
```

Each town produces an NPZ with `tick`, `actor_id`, `cls`, `x`, `y`, `yaw`, `dt`
and `town`. Positions use world coordinates in metres; yaw is in radians;
classes are 0 vehicle, 1 pedestrian, 2 two-wheeler. Actor IDs are scoped to each
recording. Existing local recordings cover six towns with 3,000 ticks each.
The separate `tools/eval_trajectories.py` evaluator trains a Ridge residual
baseline on Town01/02/03, selects its alpha on Town04 and tests on Town05/10HD.
It compares CV, constant acceleration and Ridge with identical GT histories;
moving actors are also reported separately. See the
[forecasting protocol](../docs/benchmarks/README.md#forecasting-town-disjoint-actor-histories).
The runtime predictor continues to use CV/CTRV.
