# 路径规划、车辆控制与闭环验证

Auto-Driver 已增加规划与控制模块，复用既有检测、跟踪和预测输出。
这里的结果来自实际执行的 Python 闭环程序和 CARLA 客户端；网页只回放导出记录。

[交互回放](https://ayane225.github.io/Auto-Driver/#driving) ·
[20 次合成运行指标与哈希](assets/driving/index.json) ·
[CARLA 真值输入](assets/driving/carla_gt.json) · [CARLA LiDAR 输入](assets/driving/carla_lidar.json)

## 已实现的功能

| 功能 | 实现 | 输入与约束 |
|---|---|---|
| 全局路线搜索 | 有向道路图上的 A*；剔除封闭边后重新搜索 | 道路连接与通行方向由地图提供；不是自动识别地图 |
| 参考路线 | SciPy CubicSpline 平滑、按实际弧长采样 | 平滑本身不保证车辆可行性，局部规划另查曲率与道路边界 |
| 局部绕障 | 沿参考路线生成五次多项式横向过渡，采样横向目标和速度 | 默认 3 个横向目标 × 5 个速度目标，滚动规划未来 4 秒 |
| 动态碰撞检查 | 世界系定向矩形的分离轴检查，覆盖所有预测模式 | 固定余量加时间采样平移余量；失败时不把制动回退标为可行解 |
| 跟车 | 前车速度 + 1.5 秒时间间距 + 2 米静止间距的速度反馈 | 对沿路线同向移动的目标限速；仍检查整条候选轨迹碰撞 |
| 速度规划 | 巡航速度、弯道横向加速度、加减速、加加速度和停车位置限制 | 正常制动 3 m/s²，紧急制动 7 m/s²；均为当前测试车辆设定 |
| 让行与停车 | 横穿目标预测、道路阻断、停车线、绿灯解除约束、到达终点 | 停车线与灯色由场景输入，未实现图像交通灯识别 |
| 路径跟踪 | 后轴参考点 Pure Pursuit + 速度反馈和加速度前馈 | 前进模式，转向角和转向变化率受限 |
| 异常处理 | 观测过期、未来时间戳、非法目标几何、无可行候选时制动 | 控制器也拒绝过期计划；不保证制动距离不足时避免碰撞 |
| CARLA 执行 | 实际发送 `VehicleControl`，用车辆反馈更新下一轮规划 | 自车禁用 autopilot，世界设置在退出时恢复，只清理本脚本创建的车辆和传感器 |

算法位于 `perception_core/planning/`、`control/`、`simulation/`，保留现有包名和感知接口。
此实现是采样规划器，不是 MPC 或全局最优运动规划器。增加 C++ 不是这一阶段的必要条件；
矩形检查已使用 NumPy 批量运算，既有 BEV IoU 仍可使用 C++ 后端。

## 接口与坐标

- 所有规划输入使用同一右手世界坐标系，单位为米、秒、弧度。
- `VehicleState.x/y` 表示自车后轴中心；`Box3D.x/y` 表示目标框中心。
  `rear_to_center` 显式连接两个参考点，检查碰撞时使用整车外廓。
- `PerceptionOutput.timestamp` 与自车状态使用同一时钟；默认最大观测年龄为 0.35 秒。
- `MotionPlan.times` 相对计划时间戳，包含位置、朝向、速度、加速度、候选拒绝原因和状态。
- 未形成确认轨迹的检测仍作为静态障碍使用；已匹配的轨迹使用预测。
  预测结束之后按最后一段速度外推；静止目标固定框朝向，矩形朝向按 π 周期插值，
  避免微小速度噪声或等价反向朝向让障碍框虚假旋转。
- `ControlCommand.steering` 为左转正的前轮角，`acceleration` 为有符号加速度。
  CARLA 的 y 和转向符号相反，适配器显式转换。

```python
from perception_core.planning import LocalPlanner, ReferencePath, VehicleState
from perception_core.control import PathController

route = ReferencePath([[0, 0], [30, 0], [60, 8]])
planner = LocalPlanner(route)
controller = PathController()
# output = pipeline.process(frame)，其检测、轨迹与预测必须已经位于世界系。
state = VehicleState(x=0, y=0, yaw=0, speed=0, timestamp=output.timestamp)
plan = planner.plan(state, output, stop_s=None)
command = controller.command(state, plan, dt=0.1)
```

## 合成闭环验证

固定种子 2026，规划与控制周期 0.1 秒；自行车模型按 0.02 秒步长推进并检查实际碰撞。
每轮观测根据控制后的自车位姿重新生成，规划器不读取场景未来真值。
全部运行在本机 CPU 上完成，Python 3.12；公开指标中的规划加控制 p95 为 **6.1–11.0 ms**，
不含感知、场景生成、动力学、评分或渲染，也没有把它作为完整系统实时性的结论。

| 场景 | 验收目标 | 真值检测 | 合成 LiDAR |
|---|---|---|---|
| 巡航 | 到达并停车 | 通过 | 通过 |
| 弯道 | 道路范围内到达 | 通过 | 通过 |
| 静态障碍 | 展示绕行并到达 | 通过 | 通过 |
| 跟车 | 展示时间间距控制并到达 | 通过 | 通过 |
| 横穿行人 | 展示让行并到达 | 通过 | 通过 |
| 红绿灯 | 停车线前约束、绿灯后到达 | 通过 | 通过 |
| 道路阻断 | 低于 0.15 m/s 持续停车超过 2 秒 | 通过 | 通过 |
| 突现障碍 | 展示紧急制动并持续停车 | 通过 | 通过 |
| 感知超时 | 展示超时制动、恢复后到达 | 通过 | 通过 |
| 封路重选路线 | 避开封闭图边并到达 | 通过 | 通过 |

所有场景还要求无实际矩形碰撞采样、无道路边界越界采样、无红灯越线采样。
到达条件为路线剩余距离和二维终点距离均小于 0.7 米、速度小于 0.2 m/s。
十场景两种输入共 **20/20** 通过，不代表对未测场景的成功率估计。
规划候选会主动减速或被拒绝；报告保留紧急制动状态，不能据通过结果宣称所有过程都平顺。

两种输入必须区分：

- `gt`：直接读取当前时刻的真值框，再经过跟踪和预测，主要验证规划控制。
- `lidar`：使用已有表面点采样器生成点云，执行地面移除、0.2 m 体素 DBSCAN、PCA 框、跟踪和预测。
  不向检测器提供真值框。它有采样噪声，但没有光线遮挡、漏回波和完整传感器物理模型。

`min_separating_axis_gap_m` 是分离轴上的间隙，不是欧氏距离或校准安全裕度；
`max_lateral_offset_m` 是相对路线中心的最大偏移，包含有意绕障，不能全当作控制误差。
离散碰撞与恒速/恒转弯预测存在局限；参考道路范围由场景定义，不包含现实道路通行权限推断。

## CARLA 0.9.16 实际车辆控制

在 Town10HD_Opt 第 6 个生成位置选择前方无路口车道，使用 Tesla Model 3。
轮距和车体尺寸来自 CARLA；为使前轮角与控制输入一致，固定自车转向比例曲线。
输出油门/制动使用简单加速度到执行器的映射，尚未经过多车型标定。

| 测试 | 结果 | 最大路线横向偏移 | 碰撞 / 压线事件 |
|---|---|---:|---:|
| 当前目标真值输入，沿车道到达 | 行驶约 61.08 m，30.8 s，到达并停车 | 0.299 m | 0 / 0 |
| 实际 64 线 LiDAR，前方停放车辆 | 行驶约 27.71 m，15.8 s，制动后保持停车 | 0.159 m | 0 / 0 |

第二项执行真实 CARLA LiDAR → 聚类检测 → 跟踪 → 预测 → 规划 → 控制。
评分使用 CARLA 碰撞和压线传感器，视频来自自车前视相机，每 0.2 秒保存一张图。
这两项是限定路线的接入验证；合成场景中的绕障、横穿行人、红绿灯和重选路线
尚未在 CARLA 城市交通中全面验证。现有 ROS 2 节点仍发布感知结果，尚未接入规划控制消息。

## 复现与导出

在仓库根目录，先安装 `pip install -e './src/perception_core[test]'`：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_driving.py \
  --scenario all --detector gt --out outputs/driving-gt --assert-success
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_driving.py \
  --scenario all --detector lidar --out outputs/driving-lidar --assert-success
python -m pytest src/perception_core/tests/test_planning.py
python tools/export_driving.py --gt outputs/driving-gt --lidar outputs/driving-lidar \
  --out outputs/driving-web
```

各输出目录必须是新目录。完整记录保留全精度浮点数，网页回放保留四位小数，
验收指标保持原精度；索引保存原报告、公开报告和源代码文件的 SHA-256。
网页提供数据源和场景切换、时间轴、播放倍率、候选与跟踪图层、速度曲线、JSON 下载及复现命令。

CARLA 测试需匹配的客户端、Pillow、独立服务器：

```bash
/path/to/CARLA/CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2100 -nosound
python carla/run_driving.py --port 2100 --detector gt --images \
  --out outputs/carla-cruise --assert-success
python carla/run_driving.py --port 2100 --scenario obstacle --detector lidar --images \
  --out outputs/carla-obstacle --assert-success
python tools/export_carla_driving.py --dataset outputs/carla-cruise --name gt \
  --out outputs/carla-cruise-web
python tools/export_carla_driving.py --dataset outputs/carla-obstacle --name lidar \
  --out outputs/carla-obstacle-web
```

不启动 CARLA 也能运行全部十个合成场景。发布视频保留采样时序，报告保存每张源相机图像和视频的哈希。
新增单元测试覆盖路线方向与封路、旋转矩形、过期输入、异常目标、框朝向、停车不倒车、终点附近起步，
并执行全部十个闭环验收场景；网页测试另检查20份回放、控制数据显示、播放、下载、异步切换和移动布局。
