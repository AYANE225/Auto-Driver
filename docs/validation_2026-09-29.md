# Integration validation — 2026-09-29

CARLA replay now uses the shared dataset reader, coordinate transforms, camera
overlay, track palette and HOTA/IDF1 evaluators. ROS replay publishes recorded
LiDAR and timestamped TF; the perception node tracks in the requested output
frame. The YOLO adapter converts the core's RGB arrays to Ultralytics BGR input.

## Executed checks

| Check | Result |
|-------|--------|
| Core pytest suite, Python 3.12.13, no torch | 86 passed |
| Ruff over `src`, `carla`, `tools` | Passed |
| ROS 2 Humble colcon build | All three ROS packages built |
| CARLA → PointCloud2 + TF → ROS perception | 8/8 track and prediction messages, confirmed objects, `map` output, preserved 0.1 s timestamp intervals |
| Synthetic LiDAR + IMM, 60 frames | Precision 0.95, recall 0.8233, MOTA 0.7767 |
| Synthetic GT + IMM latency, 120 measured frames | Mean 1.085 ms, p95 1.121 ms; 150 ms budget passed |
| Analytic motion prediction bank | Overall ADE 1.695 m, FDE 3.688 m |
| KITTI drive 0014 GT replay, 314 frames | MOTA 0.6410, HOTA 0.6266, IDF1 0.8107 |
| KITTI drive 0014 YOLO fusion, 314 frames | MOTA −0.1950, HOTA 0.1828, IDF1 0.1979 |
| CARLA urban and highway fusion + IMM | 400 frames each; full metric reports linked below |
| Camera + BEV GIFs | Two files, each 1186×448, 120 frames, 24 s; sampled frames visually inspected |

HOTA's eight reported components and IDF1's six fields were additionally checked
against [TrackEval](https://github.com/JonathonLuiten/TrackEval/tree/12c8791b303e0a0b50f753af204249e622d0281a)
on 40 seeded random sequences of 12 frames (seed 20260929), including missed
detections, exits, ID splits and localization noise. All comparisons passed with
absolute tolerance 0.000051, accounting for this project's four-decimal output.
The temporary reference harness supplied NumPy compatibility aliases for
TrackEval's `np.float` and `np.int`; production code does not use those aliases.

## Reproduce local checks

```bash
pip install -e 'src/perception_core[dev]'
make test
make lint
python tools/run_demo.py --frames 60 --detector lidar --tracker imm --no-video
python tools/benchmark.py --detector gt --tracker imm --frames 120 --budget-ms 150
python tools/eval_prediction.py
```

ROS requires a built and sourced workspace. Run the following separately from
other publishers using the same topics:

```bash
python tools/check_ros_replay.py --dataset carla/data/highway --frames 8
```

See [CARLA commands](../carla/README.md),
[urban metrics](screenshots/metrics_carla_urban.json),
[highway metrics](screenshots/metrics_carla_highway.json), and
[KITTI fusion metrics](screenshots/metrics_kitti_fusion.json). Full CARLA replay
used Python 3.11.15 with YOLOv8n on an RTX 5090. The full detection pipeline is
slower than the 10 Hz sensor interval; the synthetic GT benchmark does not
measure LiDAR clustering or YOLO inference. GIF playback does not measure throughput.
