# C++ 跟踪加速与算法消融（2026-09-29）

本轮把跟踪器的批量旋转框 BEV IoU 计算迁移到 C++14，Python 保留其余调度。
默认启用匹配前的 IoU 阈值处理；π 周期框朝向平滑保留为可选项。
**速度改善已实测，当前 800 帧没有证明精度提升。**

## 固定检测输入：仅比较跟踪计算

在基线提交 `7142c920962a4a7751239ff22fb0ed22ba37ca50` 上，先保存两个场景
各 400 帧的世界坐标系检测、相机标定、评估真值以及已确认轨迹。检测不读取真值。
随后五种配置使用完全相同的检测输入，每种运行三轮，轮换执行顺序。
计时仅包含 `tracker.update`，排除前 5 帧，不含检测、读盘、坐标转换、FOV 筛选或评估。

| 场景 | Python 原逻辑均值 | C++ 原逻辑均值 | 跟踪提速 | Python / C++ 各轮 p95 的平均值 |
|---|---:|---:|---:|---:|
| 城市 | 44.713 ms | 3.216 ms | 13.90× | 79.756 / 4.602 ms |
| 高速 | 81.125 ms | 3.048 ms | 26.62× | 430.756 / 10.294 ms |

三轮均值在报告 `mean_ms` 中；`mean_p95_ms` 是三轮各自 p95 的平均，
**不是合并全部样本后的 p95**。每轮原始统计保存在 `timing_runs`。
两个原逻辑配置在全部 800 帧输出的已确认轨迹 ID、框和速度均与缓存一致，
`max_state_error=0`；这不是对任意输入的逐位等价保证，也不涵盖未返回的内部滤波状态。

原 Python 跟踪器在帧索引 300–319 的局部 cProfile 中，城市 `update` 累计
2.052 s，其中 IoU 1.898 s（54,252 次）；高速为 15.633 s / 15.127 s
（434,048 次）。这些受 profiler 开销影响的局部诊断时间用于定位瓶颈，不能作为普通耗时基准。

## 算法消融与默认选择

指标使用未修改的评估器：相同 LiDAR XY 范围与前视相机 FOV，包含遮挡真值；
CLEAR-MOT 使用 BEV IoU 0.3，IDF1 使用 0.5，HOTA 平均 0.05–0.95 的 19 个阈值。
使用记录中的稳定 actor ID，按类别无关方式匹配。

| 场景 / 配置 | HOTA | IDF1 | MOTA |
|---|---:|---:|---:|
| 城市 / Python 原逻辑 | 0.1840 | 0.1726 | 0.0445 |
| 城市 / C++ 原逻辑 | 0.1840 | 0.1726 | 0.0445 |
| 城市 / C++ + π 周期 | 0.1839 | 0.1715 | 0.0440 |
| 城市 / C++ + 阈值处理（默认） | 0.1840 | 0.1726 | 0.0445 |
| 城市 / C++ + 两项改动 | 0.1839 | 0.1715 | 0.0440 |
| 高速 / Python 原逻辑 | 0.1987 | 0.0585 | −0.4503 |
| 高速 / C++ 原逻辑 | 0.1987 | 0.0585 | −0.4503 |
| 高速 / C++ + π 周期 | 0.1988 | 0.0585 | −0.4627 |
| 高速 / C++ + 阈值处理（默认） | 0.1987 | 0.0585 | −0.4503 |
| 高速 / C++ + 两项改动 | 0.1988 | 0.0585 | −0.4627 |

- **C++ 原逻辑**只替换 IoU 计算。缓存顶点、轴对齐包围框排除和精确多边形裁剪
  减少重复运算；双精度计算交集面积时平移到局部原点。没有改动评估器。
- **阈值处理**修复“低于阈值的边参与分配，挤掉有效匹配后又被过滤”的边界问题。
  例如 IoU 矩阵 `[[0.60, 0.59], [0.59, 0]]`、阈值 0.60，旧策略输出零匹配，
  新策略保留 `(0, 0)`。目标是最大有效 IoU 总和，允许未匹配，不优先最大匹配数。
  真实记录上的评估指标相同，但全场景轨迹状态存在变化，不能称为输出完全等价。
- **π 周期平滑**按 PCA 无向轴处理 `yaw` 与 `yaw+π`，避免符号翻转引发错误旋转。
  它通过专门的交替翻转测试，但这两段记录没有稳定的质量收益，故默认保留 `2π`。
  包围框轴向不是车辆行进方向，速度估计与运动预测分开处理。

报告：[城市五项消融](urban_ablation.json) / [高速五项消融](highway_ablation.json)。
内含全部配置、Precision/Recall、匹配计数、环境、输入 SHA-256 和三轮耗时。

## 完整传感器流水线

另行顺序执行四次 400 帧回放：城市旧/新、高速旧/新。每配置一次，期间未运行
其他重 CPU/GPU 测试。相同 0.2 m 体素、YOLOv8n、IMM、CUDA、FOV；旧配置
为 Python + `post_filter` + `2pi`，新配置为 C++ + `thresholded` + `2pi`。
仅代码、文档编辑与轻量检查同时进行。这组结果不与旧体素实验拼接计算提速。

| 场景 / 配置 | 检测均值 ms | 跟踪阶段均值 / p95 ms | 流水线均值 / p95 ms |
|---|---:|---:|---:|
| 城市 / Python 原逻辑 | 81.179 | 43.096 / 77.265 | 130.215 / 171.837 |
| 城市 / C++ + 阈值处理 | 80.045 | 3.410 / 4.765 | 89.310 / 103.960 |
| 高速 / Python 原逻辑 | 69.353 | 77.144 / 408.479 | 151.993 / 498.151 |
| 高速 / C++ + 阈值处理 | 69.327 | 3.289 / 10.590 | 78.079 / 101.199 |

流水线均值分别提速 **1.46× / 1.95×**，p95 分别下降 **39.5% / 79.7%**。
新旧 Precision、Recall、HOTA、IDF1、MOTA 及匹配计数均相同。
城市 Recall 0.2231，高速 0.2298；质量瓶颈仍在检测与目标覆盖。

计时包含检测、相机确认、坐标变换、跟踪和预测；不含读盘、渲染、评估、前 5 帧预热。
这里的 `track_ms` 包含检测框转世界坐标系的时间，略宽于固定输入的 `tracker.update`。
新均值低于 10 Hz 的 100 ms 间隔，但两个 p95 仍略高于 100 ms，且未计入完整在线输入输出，
不能据此保证实时截止时间。每配置只有一次完整流水线运行，没有置信区间。

报告：[城市旧](urban_python_original.json) / [城市新](urban_cpp_threshold.json) /
[高速旧](highway_python_original.json) / [高速新](highway_cpp_threshold.json)。

## 环境、安装与复现

实测环境：Intel Core Ultra 9 285K，GCC 13.3.0，Linux x86_64，Python 3.11.15，NumPy 2.4.6，YOLO 使用 NVIDIA RTX 5090；
`OPENBLAS_NUM_THREADS=1`，`OMP_NUM_THREADS=1`。未固定 CPU 亲和性。
原始记录、YOLO 权重与检测缓存不在 Git 中；JSON 保存输入文件哈希以便核对。
两个场景是开发对照，不能证明新道路、新天气或实车场景的泛化。

```bash
# 在仓库根目录；C++14 编译器和 Python 开发头文件需可用。
pip install './src/perception_core[yolo,test]'
python -c 'from perception_core.common.iou import resolve_iou_backend; print(resolve_iou_backend())'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for scene in urban highway; do
  # 先准备 carla/data/$scene 记录。缓存路径须未使用。
  python tools/benchmark_tracking.py --dataset "carla/data/$scene" \
    --cache "outputs/tracking/${scene}_detections.json.gz" --device cuda:0
  python tools/benchmark_tracking.py \
    --cache "outputs/tracking/${scene}_detections.json.gz" \
    --report "outputs/tracking/${scene}_ablation.json" --repeats 3
  python carla/replay_demo.py --dataset "carla/data/$scene" \
    --detector fusion --tracker imm --device cuda:0 --fov-eval \
    --voxel-size 0.2 --no-video --iou-backend python \
    --matching-policy post_filter --box-yaw-period 2pi \
    --report "outputs/tracking/${scene}_python_original.json"
  python carla/replay_demo.py --dataset "carla/data/$scene" \
    --detector fusion --tracker imm --device cuda:0 --fov-eval \
    --voxel-size 0.2 --no-video --iou-backend cpp \
    --matching-policy thresholded --box-yaw-period 2pi \
    --report "outputs/tracking/${scene}_cpp_threshold.json"
done
```

`auto` 默认优先使用原生扩展；`python` 强制参考实现；`cpp` 要求扩展可用。
缺少编译器时允许退回纯 Python 安装，也可在新检出目录使用
`PERCEPTION_CORE_NO_NATIVE=1 pip install './src/perception_core[test]'`。
运行不依赖 pybind11，但隔离构建会下载它。详细接口见
[核心说明](../../../src/perception_core/README.md)。

网站视频、原有体素图表和 42 个逐帧样本保留本轮改动前的配置。
新页面的 C++ 对照读取本目录 JSON；旧展示资源不代表新算法输出。

## 验证记录

- 独立 Python 3.12 虚拟环境通过 PEP 517 安装，断言原生后端存在后，104 项核心测试通过。
- `PERCEPTION_CORE_NO_NATIVE=1` 安装后断言实际回退 Python：101 项通过，3 项原生专用测试跳过。
- 将 C/C++ 编译器路径设为不存在的位置，仍成功安装，实际后端为 Python。
- 源码分发包含 `cpp/geometry.cpp`；从 `.tar.gz` 重新安装后，原生后端可用。
- 数值测试覆盖旋转、相切、包含、细长框、空输入、非法输入、非连续数组与大世界坐标；
  跟踪测试覆盖 π 符号翻转、阈值匹配冲突和 Mahalanobis 门控。
- Ruff 与差异空白检查通过。Playwright 核对新表格与全部六份 JSON，验证场景切换、
  失败回退、无 JavaScript 内容，以及既有 42 帧查看、预测、下载和手机布局功能。
- CI 要求 Python 3.9–3.11 原生扩展实际存在，另设纯 Python 安装与测试任务，避免原生测试
  因缺少编译而全部跳过。CI 最终状态以对应提交的 GitHub Actions 为准。
