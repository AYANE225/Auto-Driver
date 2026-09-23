# CARLA data layer

Self-collected simulation data for the perception stack. CARLA provides
photorealistic sensor streams (LiDAR + cameras) with perfect ground truth, which
we record to a compact on-disk dataset and then replay through
`perception_core` / ROS 2.

## Why a record-then-replay design?

CARLA's Python client targets **Python 3.7/3.8**, while ROS 2 Humble runs on
**Python 3.11**. Rather than fight that mismatch, the two are decoupled:

```
 CARLA server (GPU)                 record_scenario.py (py3.8 client)
  Town + traffic     ── sensors ──►  spawn ego + LiDAR/cameras + NPC traffic
                                     step in sync mode, save frames
                                              │
                                              ▼
                                     data/<scenario>/  (framework-neutral .npz)
                                              │
                         ┌────────────────────┴─────────────────────┐
                         ▼                                           ▼
        replay_demo.py (py3.11, perception_core)      ROS 2 replay node (av_perception)
        offline BEV GIF + CLEAR-MOT metrics           publishes PointCloud2 → live graph
```

The recorded dataset is plain NumPy + JSON, so it is portable, versionable, and
replayable with **no CARLA and no ROS installed**.

## Contents

| File | Runs under | Purpose |
|------|-----------|---------|
| `install_carla.sh` | bash | Download CARLA 0.9.15 server + create a py3.8 client venv |
| `record_scenario.py` | CARLA py3.8 | Spawn ego + sensors + NPC traffic, record a scenario |
| `dataset.py` | py3.11 | Read a recorded scenario back into `perception_core.Frame`s |
| `replay_demo.py` | py3.11 | Run the pipeline over a scenario → BEV GIF + metrics |
| `config/scenarios/*.yaml` | – | Reproducible scenario definitions (town, traffic, sensors) |

## Coordinate convention

CARLA is left-handed (x-forward, **y-right**, z-up). We convert to the
right-handed frame used by `perception_core` (x-forward, **y-left**, z-up) by
flipping the sign of `y` and negating yaw. LiDAR points are stored in the sensor
frame; `ego_pose` is the `world <- lidar` transform; ground-truth boxes are
stored in the world frame, so they align with world-frame tracks once the
pipeline lifts detections through `ego_pose`.

## Quickstart

```bash
# 1. Install (one-time; downloads several GB of CARLA server)
bash install_carla.sh

# 2. Start the server (headless, needs a GPU)
$HOME/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen

# 3. Record a scenario (CARLA client env)
source $HOME/.venvs/carla-client/bin/activate
python record_scenario.py --scenario config/scenarios/urban.yaml --out data/urban --images

# 4. Replay through the pipeline (needs only perception_core)
python replay_demo.py --dataset data/urban --gif ../docs/screenshots/demo_carla.gif \
    --report ../docs/screenshots/metrics_carla.json
```

> Recording needs a running CARLA server with a GPU; the recorded dataset does
> not. `dataset.py` and `replay_demo.py` are exercised by the repo's normal
> Python tooling and require no CARLA install.
