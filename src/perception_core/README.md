# perception_core

Framework-agnostic autonomous-driving perception core: **detection → fusion →
tracking → prediction**. No hard dependency on ROS, CARLA, or a deep-learning
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
| Tracking   | `MultiObjectTracker` (const-velocity KF + Hungarian) | scipy             |
| Prediction | `MotionPredictor` (CV / CTRV, multi-modal)         | –                   |

All detectors implement the same `Detector` interface, so backends are
swappable without touching the pipeline.

## Tests

```bash
pip install -e .[test]
pytest -q
```

> Inside a sourced ROS 2 environment, disable the incompatible ROS pytest
> plugins first: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`.
