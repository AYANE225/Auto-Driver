# Auto-Driver 中文工程说明

[项目首页](../README.md) · [在线展示](https://ayane225.github.io/Auto-Driver/) · [详细评估协议（英文）](benchmarks/README.md)

## 各部分的职责

| 部分 | 实现 | 边界 |
|---|---|---|
| LiDAR 检测 | ROI 裁剪、RANSAC 地面移除、DBSCAN、PCA 定向框 | 几何聚类会漏检或把环境结构误检为目标 |
| 相机确认 | 将 LiDAR 框投影到图像，按 2D IoU 匹配 YOLO 框 | 框级后融合，没有学习式特征融合 |
| 多目标跟踪 | CV Kalman 或 CV/恒转弯 IMM，Hungarian 关联 | 检测质量和关联门限影响身份连续性 |
| 在线预测 | CV/CTRV，默认未来 3 秒、每 0.5 秒一个位置 | 固定模式权重不是校准概率 |
| 规划与控制 | A* 路线、局部轨迹与速度采样、碰撞检查、Pure Pursuit 与速度反馈 | 合成闭环 20/20 与 CARLA 两次限定场景通过，详见[验证说明](planning_control_zh.md) |
| 离线预测比较 | 使用历史 XY 的 Ridge 残差回归 | 与传感器感知流水线分开评估，尚未接入在线预测 |
| ROS 2 | 点云和 TF 输入，跟踪、预测、RViz 输出 | 当前 ROS 路径使用 LiDAR + IMM，相机融合在离线脚本运行 |
| 网页 | 读取运行导出的 JSON、图像和视频 | 浏览器绘制结果，不执行神经网络推理 |

`Frame` 统一存储时间戳、点云、RGB 图像、标定、自车位姿与可选真值。
`PerceptionOutput` 保存检测、轨迹和预测。算法核心依赖 NumPy/SciPy/scikit-learn，
YOLO、VGGT、Pillow、Matplotlib、ROS 与 CARLA 客户端均按功能选用。

## C++ 跟踪加速与关联配置

Python 保留检测、滤波、关联求解与预测调度；C++14 扩展批量计算旋转框的 BEV IoU。
每个框只生成一次顶点，先用轴对齐包围框排除不相交候选，再用
Sutherland–Hodgman 多边形裁剪计算交集。计算采用双精度，并在局部坐标下求面积。
评估器仍使用原实现，避免把评估计算变化混入质量对照。

```bash
# 正常安装会尝试编译，需要 C++14 编译器和 Python 开发头文件。
pip install './src/perception_core[test]'
python -c 'from perception_core.common.iou import resolve_iou_backend; print(resolve_iou_backend())'
# 新检出目录中，可明确关闭编译：
PERCEPTION_CORE_NO_NATIVE=1 pip install './src/perception_core[test]'
```

`TrackerConfig.iou_backend` 可选 `auto`、`python` 或 `cpp`。默认 `auto` 优先使用
C++，不可用时退回 Python；显式指定 `cpp` 时，缺失扩展会报错。pybind11 由隔离构建
环境安装，运行时不需要它。CARLA 回放使用对应的 `--iou-backend` 参数，JSON 保存实际后端。

默认 `matching_policy="thresholded"` 在分配前把低于 IoU 阈值的边置为零权重，
求解后丢弃这些边，最大化有效 IoU 总和，允许目标未匹配。它不保证优先最大匹配数量。
`post_filter` 保留旧的“先分配再按阈值过滤”行为，可复现历史结果。

框朝向默认保留 `box_yaw_period=2*pi`；可用 `--box-yaw-period pi` 处理 PCA 轴向
正负号等价问题。π 周期平滑在交替翻转的单元场景有效，但未稳定改善当前实录评估，
因此作为可选项。框的无向轴不等于车辆行进方向，运动预测仍使用速度估计。

[本轮完整报告](benchmarks/tracking/README.md)包含固定检测消融、完整流水线计时、
状态一致性和复现命令。网站已有视频及逐帧数据保留旧跟踪配置，不是新算法输出。

## 坐标、时间与身份

点云位于右手 LiDAR 坐标系，x 向前、y 向左、z 向上。`ego_pose` 表示
`world <- lidar`，检测框经这个变换进入世界坐标系后再跟踪。绘图及网页导出将
输出副本变回当前 LiDAR 坐标系，原始状态不改变。

网页中的轨迹速度来自世界系估计，再旋转到当前 LiDAR 轴，不减去自车速度。
因此它不是“相对自车的速度”。真值 actor ID 和跟踪器分配的 ID 是不同编号体系，
不能凭数字相同认定匹配。相机图是当前帧的原始图像，不包含网页 Canvas 上的叠加标注。

## 导出逐帧查看数据

先按 [CARLA 说明](../carla/README.md) 准备带前视相机的记录，然后在项目根目录运行：

```bash
pip install -e './src/perception_core[yolo]'
pip install pillow
python carla/replay_demo.py --dataset carla/data/urban --detector fusion \
  --tracker imm --device cuda:0 --fov-eval --voxel-size 0.2 \
  --no-video --report outputs/urban_export.json \
  --export-replay outputs/urban_viewer --export-every 20 --export-points 2500
```

将 `urban` 换为 `highway` 即可处理另一场景。没有 CUDA 时，可设 `--device cpu`。
所有输入帧仍逐帧更新跟踪器；`--export-every` 只控制网页样本的保存间隔，最后一帧始终保存。
`--export-points` 只控制网页显示点数，检测与评估不受影响。点云显示范围为
x ±55 m、y ±45 m、z −3 至 4 m，并以帧 ID 为随机种子进行可重复抽样。

导出目录包含：

```text
index.json       场景、坐标约定、配置、全部样本时间戳和路径
000000.json      点云样本、真值框、跟踪状态、历史与预测
000000.jpg       同一时刻的原始前视相机画面
...
```

输出到新目录；存在 `index.json` 时会拒绝覆盖，避免悄悄替换已有展示。
把导出文件放在 `docs/assets/replay/urban/` 或 `highway/` 后，可由现有页面读取。
自己保存的数据建议先在其他目录生成并核对，再用于网页。

```bash
python -m http.server 8000 --directory docs
```

网页支持选择采样时刻、图层、跟踪 ID、缩放和拖动画布，以及 PNG/JSON 下载。
目标框和轨迹沿用评估区域及可选相机 FOV 筛选，点云显示范围独立于目标筛选范围。
连续查看每个样本之间等待约 1 秒，受网络载入影响；该操作不是实时性能测试。
视频则保留既有 GIF 的帧时序。

## 预测示例如何产生

```bash
python tools/export_forecast_examples.py \
  --root carla/data/trajectories \
  --model docs/benchmarks/forecasting/ridge.npz \
  --out outputs/forecast_examples.json
```

在每个测试城镇中，先用过去 1 秒位移至少 0.5 m 选出移动窗口，再沿窗口索引等间隔抽取
12 个示例，共 24 个。选择过程不使用预测误差。每个样本输出 2 秒历史、3 秒实际未来和
CV/CA/Ridge 预测，以最后观测位置为原点，保留世界 XY 轴方向。浏览器用相同尺度绘制
横纵坐标，并从显示的四位小数坐标计算当前样本 ADE/FDE；微小舍入差异不影响总体报告。
JSON 记录原始轨迹与模型文件 SHA-256。示例只用于理解个体行为，全量统计仍以原始评估报告为准。

## ROS 2 与 Docker

在已加载 ROS 2 Humble 的环境中：

```bash
colcon build --packages-select av_perception_msgs av_perception av_bringup
source install/setup.bash
# 合成点云源。
ros2 launch av_bringup perception.launch.py rviz:=false
# 或回放录制点云与对应时间戳 TF。
ros2 launch av_bringup perception.launch.py source:=carla \
  dataset:=/absolute/path/to/carla/data/highway replay_rate:=2.0 rviz:=false
```

下列消息检查应与以上 launch 分开运行，以免重复发布：

```bash
python tools/check_ros_replay.py --dataset carla/data/highway --frames 8
```

已有验证收到了 8/8 条跟踪与预测消息、确认目标、`map` 输出及 0.1 秒传感器时间间隔。
详见[接入记录](validation_2026-09-29.md)。此项与网页新功能的验证分开记录。

```bash
# CPU 核心镜像。
docker build -t auto-driver-core .
docker run --rm auto-driver-core
# ROS 2 工作区镜像。
docker build -f docker/Dockerfile.ros2 -t auto-driver-ros .
docker run --rm auto-driver-ros
```

## 开发与验证

```bash
pip install -e './src/perception_core[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q src/perception_core/tests
ruff check --config src/perception_core/pyproject.toml src carla tools
```

网页使用原生 HTML/CSS/JavaScript，无运行时第三方包。浏览器检查需要单独安装 Playwright，
覆盖场景切换、时间轴、图层、目标选择、预测对照、文件下载、命令配置、移动端与无 JavaScript 回退。
在一个终端运行 `make site`，另一个终端运行：

```bash
npm install --prefix /tmp/auto-driver-browser playwright@1.63.0
/tmp/auto-driver-browser/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/auto-driver-browser/node_modules make site-test
```

测试还会模拟旧帧相机图像延迟和数据请求失败，验证画面同步与错误恢复。
CI 自动运行相同脚本并保存桌面和手机截图。
GitHub Actions 发布 `docs/index.html`、`assets/` 与 `benchmarks/`，原始传感器记录不上传。

## 三维点云与交互演示

展示页新增独立的三维点云查看器、分阶段耗时图、可调阈值的匹配演示和 KITTI/VGGT 视频展区。
三维显示点来自原始帧的额外抽样，按需加载，默认 12,000 点、最多 20,000 点；旧二维查看器
仍保持原来的 2,500 点样本。框与轨迹沿用旧配置，不把显示点数变化作为检测改进。
数据格式、复现脚本与检查范围见[本轮展示说明](showcase_2026-09-29.md)。
