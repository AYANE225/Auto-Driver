# perception_core

Framework-agnostic autonomous-driving perception core: **detection → fusion →
tracking → prediction**, plus route search, local motion planning and vehicle
control with closed-loop simulation. No hard dependency on ROS, CARLA, or a deep-learning
framework — the whole stack runs on NumPy / SciPy / scikit-learn and is unit
tested on CPU.

```python
from perception_core.pipeline import PerceptionPipeline
from perception_core.io.synthetic import generate_frames, make_default_scene

pipe = PerceptionPipeline()                      # LiDAR clustering + KF/Hungarian MOT + CTRV
for frame in generate_frames(make_default_scene(), num_frames=40):
    out = pipe.process(frame)
    for tr in out.tracks:
        print(tr.track_id, tr.label.value, tr.box.center, tr.velocity)
```

## Components

| Stage      | Default implementation                             | Key deps            |
|------------|----------------------------------------------------|---------------------|
| Detection  | `LidarClusterDetector` (RANSAC ground + DBSCAN + PCA box) | scikit-learn  |
| Detection  | `GroundTruthDetector` (replay, for tests/CI/demo)  | –                   |
| Detection  | `YoloCameraDetector` (optional 2D camera)          | ultralytics *(opt)* |
| Fusion     | `LateFusion` (project 3D→image, IoU label transfer)| –                   |
| Tracking   | `MultiObjectTracker` (CV KF or IMM CV/CT + Hungarian; optional Mahalanobis gating) | scipy |
| Prediction | `MotionPredictor` (CV / CTRV, multi-modal)         | –                   |
| Planning   | `RoadGraph`, `ReferencePath`, `LocalPlanner` (A*, lateral/speed sampling, collision checks) | numpy / scipy |
| Control    | `PathController` (Pure Pursuit + speed feedback), `bicycle_step` | numpy |
| Evaluation | CLEAR-MOT, HOTA, IDF1, ADE/FDE                    | numpy / scipy       |
| Data       | `CarlaDataset`, `KittiRawReader`, synthetic scenes | numpy              |
| Rendering  | Camera overlays and BEV with shared track colors  | pillow / matplotlib *(opt)* |

All detectors implement the same `Detector` interface, so backends are
swappable without touching the pipeline.

Planning consumes world-frame `PerceptionOutput` and rear-axle `VehicleState`.
Run `tools/run_driving.py` from the repository root for ten deterministic
closed-loop scenarios. `carla/run_driving.py` applies commands to actual CARLA
vehicles without ego autopilot. See `docs/planning_control_zh.md` for coordinate
conventions, acceptance conditions, reproduction and the bounded test scope.

## Native tracking geometry

Normal installation builds the optional `perception_core._geometry` extension
with pybind11 and a C++14 compiler. Tracking uses its batch rotated BEV IoU:
corners are cached per box, disjoint axis-aligned bounds are rejected, and
overlaps use Sutherland–Hodgman polygon clipping in double precision. The Python
reference and all evaluators remain available. Native execution releases the GIL.

```bash
pip install .
python -c 'from perception_core.common.iou import resolve_iou_backend; print(resolve_iou_backend())'
# Explicitly disable native compilation on a fresh source checkout:
PERCEPTION_CORE_NO_NATIVE=1 pip install .
```

`TrackerConfig(iou_backend="auto")` selects C++ when available and Python
otherwise. `"python"` selects the scalar reference; `"cpp"` requires a successful
native build and raises an error if unavailable. Compilation failure permits a
Python installation. The build environment needs pybind11 in both cases; there
is no pybind11 runtime dependency. CARLA replay reports record the actual backend.

Association defaults to `matching_policy="thresholded"`: only IoUs at or above
the threshold contribute to assignment, maximizing their sum with unmatched
objects allowed. This does not prioritize match count. `"post_filter"` reproduces
the older solve-then-reject policy. Box smoothing keeps `box_yaw_period=2*pi` by
default; optional `pi` smoothing respects the sign ambiguity of PCA box axes but
did not consistently improve the recorded-scene metrics. Box orientation is
separate from the velocity-derived motion heading.

The repository's `tools/benchmark_tracking.py` captures fixed detections and
compares five configurations, including native output equivalence and algorithm
ablations. See `docs/benchmarks/tracking/README.md` for results and commands.

`Frame.images` stores RGB images. LiDAR detections are transformed into world
coordinates using `Frame.ego_pose` before tracking. CARLA and KITTI ground truth
carry stable IDs in `Detection.attributes['gt_id']`; pass these IDs to tracking,
HOTA and identity evaluators when objects enter or leave the scene. Camera and
BEV rendering use `common.ego_frame` helpers to transform copies of output state.

## Tests

```bash
pip install -e .[test]
pytest -q
```

> Inside a sourced ROS 2 environment, disable the incompatible ROS pytest
> plugins first: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`.
