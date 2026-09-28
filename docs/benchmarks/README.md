# CARLA replay and forecasting measurements

These reports were produced locally on 2026-09-29. They accompany the optional
weighted voxel clustering implementation and the recorded-trajectory evaluation.
The [interactive project page](https://ayane225.github.io/Auto-Driver/) reads the
JSON files directly. Values below are rounded only for presentation.

The subsequent [C++ tracking comparison](tracking/README.md) measures native
geometry and association changes separately. The voxel results below retain
the older Python tracking implementation and are historical measurements.

## Replay: raw points versus weighted voxel clustering

All four runs process the same 400 frames per scenario with YOLOv8n camera
confirmation, IMM tracking and the front-camera evaluation FOV. Only
`LidarClusterConfig.voxel_size` changes: 0 (raw points) versus 0.2 m. The latter
clusters voxel centroids with point-count weights, then fits boxes to original
points. The original point density threshold is retained, but voxelization can
change neighborhood geometry and cluster assignments.

| Scene / setting | Detect mean (ms) | Pipeline mean / p95 (ms) | Precision | Recall | HOTA | IDF1 | MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Urban / raw | 833.122 | 888.519 / 1575.756 | 0.5566 | 0.2248 | 0.1833 | 0.1716 | 0.0449 |
| Urban / 0.2 m | 81.038 | 129.466 / 171.157 | 0.5565 | 0.2231 | 0.1840 | 0.1726 | 0.0445 |
| Highway / raw | 209.587 | 289.472 / 599.637 | 0.2637 | 0.2391 | 0.2010 | 0.0586 | −0.4286 |
| Highway / 0.2 m | 69.823 | 152.343 / 497.444 | 0.2526 | 0.2298 | 0.1987 | 0.0585 | −0.4503 |

Pipeline speedups are **6.86× urban** and **1.90× highway**. Detection alone is
10.28× and 3.00× faster. Urban quality changes little at the reported precision;
highway recall and HOTA decrease. This is a speed/quality tradeoff. There is one
run per setting, no confidence interval or statistical claim. Both optimized
means exceed the 100 ms interval of the 10 Hz sensor stream. Tracking remains a
substantial cost, especially in the highway p95.

- Pipeline latency excludes loading, rendering, evaluation and the first five
  warmup frames. It includes detection, fusion, tracking and prediction.
- Python 3.11.15, NumPy 2.4.6, Linux x86_64, YOLO on NVIDIA RTX 5090.
  `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`; no CPU affinity or repeat-run
  variance control. Each JSON stores the configuration and available environment data.
- CARLA 0.9.16 recordings: urban `Town10HD_Opt`, highway `Town04_Opt`.
  Class-agnostic BEV IoU: CLEAR-MOT at 0.3, identity at 0.5, HOTA averaged over
  thresholds 0.05 to 0.95 in steps of 0.05. Stable recorded actor IDs are used.
- GT and tracks share x ∈ [−50, 50] m and y ∈ [−40, 40] m in the sensor frame,
  plus camera FOV. Occluded GT remains included. These are project measurements,
  not an official CARLA or KITTI benchmark.
- The GIF/video previews show the earlier raw-point run in
  `docs/screenshots/metrics_carla_*.json`. This comparison uses fresh no-video
  runs, so its latency values differ from those earlier reports.

Reports: [urban raw](urban_voxel_0.json), [urban voxel](urban_voxel_0.2.json),
[highway raw](highway_voxel_0.json), [highway voxel](highway_voxel_0.2.json).

From the repository root, with the recordings and YOLO environment available:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for scene in urban highway; do
  for voxel in 0 0.2; do
    python carla/replay_demo.py --dataset "carla/data/$scene" \
      --detector fusion --tracker imm --device cuda:0 --fov-eval \
      --voxel-size "$voxel" --no-video \
      --iou-backend python --matching-policy post_filter --box-yaw-period 2pi \
      --report "outputs/comparison/${scene}_voxel_${voxel}.json"
  done
done
```

## Forecasting: town-disjoint actor histories

This separate experiment uses recorded **ground-truth XY histories**, without
sensor detection or tracking errors. It measures model forecasting error and
does not establish end-to-end forecast accuracy.

| Split | Towns | Forecast windows |
|---|---|---:|
| Train | Town01, Town02, Town03 (`_Opt`) | 131,858 |
| Validation | Town04 (`_Opt`) | 40,415 |
| Test | Town05, Town10HD (`_Opt`) | 92,751 |

Each recording has 3,000 ticks at 10 Hz. A window contains 11 historical
positions over 2 s (0.2 s spacing), and six future positions through 3 s
(0.5 s spacing). Forecast origins are 1 s apart; incomplete, non-finite or
temporally discontinuous windows are rejected. Whole towns are assigned to
splits before fitting. Overlapping windows within each town are correlated;
reported means are weighted by window count, not by town or independent actor.

- **CV** estimates velocity over the last 0.4 s of sampled history.
- **Constant acceleration** fits a quadratic to the last 1 s of history.
- **Ridge residual** learns corrections to CV in the frame aligned with recent
  displacement. Feature scaling and coefficient fitting use training towns
  only. Alpha ∈ {1, 10, 100, 1000} is selected by validation all-window ADE;
  alpha 10 was selected. There is no fitting on the test towns.
- **Moving actors** have at least 0.5 m displacement over the past 1 s. This
  selection uses observed history only. There are 28,438 moving test windows.
- ADE averages Euclidean error across six future positions; FDE is the error at
  3 s; miss rate is the proportion with FDE > 2 m.

| Model | All ADE / FDE (m) | All miss rate | Moving ADE / FDE (m) | Moving miss rate |
|---|---:|---:|---:|---:|
| CV | 0.126 / 0.258 | 3.00% | 0.310 / 0.640 | 6.73% |
| Constant acceleration | 0.227 / 0.505 | 5.83% | 0.560 / 1.254 | 14.33% |
| Ridge residual | 0.167 / 0.350 | 2.82% | 0.369 / 0.782 | 6.30% |

Ridge has a lower test miss rate but higher ADE and FDE than CV. The learned
baseline is retained for comparison; it is not integrated into the runtime
predictor. Validation and per-town / per-class results, selected alpha, sample
counts, and SHA-256 hashes of all six inputs are in
[forecasting/metrics.json](forecasting/metrics.json). The fitted coefficients,
scaler and time grid are in [forecasting/ridge.npz](forecasting/ridge.npz).
The measured environment was Python 3.12.13, NumPy 1.26.4 and scikit-learn 1.5.1.

```bash
# Record all required towns with the CARLA client before running this evaluator.
python carla/record_trajectories.py \
  --towns Town01_Opt Town02_Opt Town03_Opt Town04_Opt Town05_Opt Town10HD_Opt \
  --ticks 3000 --out carla/data/trajectories
python tools/eval_trajectories.py --root carla/data/trajectories \
  --out outputs/forecasting
```

The raw recordings are excluded from Git. New CARLA recordings can differ even
with matching settings; compare the stored hashes to identify exact inputs.

## Presentation assets

```bash
# Requires matplotlib and Pillow; --video additionally requires ffmpeg.
python tools/make_showcase_assets.py --out outputs/showcase-assets --video
python -m http.server 8000 --directory docs
```

PNG and SVG figures are generated from the JSON reports. MP4s are transcodes of
the original GIFs with their frame timing retained. The static site uses local
assets and native video controls; it has no external font, analytics or CDN
dependency. Video decoding and loading begin when the visitor plays a clip.
