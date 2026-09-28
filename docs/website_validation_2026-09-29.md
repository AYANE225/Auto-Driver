# 中文交互页面验证记录（2026-09-29）

本次新增中文页面、逐帧点云与跟踪查看器、预测示例对照、文件下载和运行命令生成器。
页面读取导出数据；浏览器不执行感知推理。原性能对照报告和视频保持各自的实验来源。

## 导出数据

使用既有城市、高速记录，分别连续处理 400 帧；配置为 YOLOv8n 相机确认、IMM、
0.2 m 体素聚类、前视相机 FOV 评估，YOLO 运行在 `cuda:0`。
`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。

```bash
python carla/replay_demo.py --dataset carla/data/urban --detector fusion \
  --tracker imm --device cuda:0 --fov-eval --voxel-size 0.2 --no-video \
  --export-replay docs/assets/replay/urban \
  --report docs/benchmarks/urban_interactive.json
# 高速场景将上述 urban 替换为 highway。
```

每场景导出 0、20、…、380、399，共 21 个时刻；合计 42 个 JSON、42 张同帧相机图和两份索引。
每帧最多抽样 2,500 个显示点。框与轨迹沿用评估区域/FOV 筛选，点云独立裁剪为
x ±55 m、y ±45 m、z −3 至 4 m。导出不会减少跟踪器处理的输入帧数。

| 新导出运行 | 完整处理帧数 | 流水线平均 / p95（ms） | 质量核对 |
|---|---:|---:|---|
| 城市 | 400 | 130.633 / 174.327 | Precision、Recall、MOTA、HOTA、IDF1 与既有 0.2 m 报告一致 |
| 高速 | 400 | 154.034 / 505.465 | 同上 |

新导出运行的时延单独保存在 [城市报告](benchmarks/urban_interactive.json) 和
[高速报告](benchmarks/highway_interactive.json)，不替换原始性能对照值。
计时范围仍为 pipeline stages，不含读盘、网页导出、渲染与评估，并排除前 5 帧预热。

逐项核验全部样本的帧 ID、轨迹数、显示点数和对应 JPEG 文件。
另对两场景第 0、120、399 帧，从原记录重新读取点云与相机图像：

- 显示点云与按相同规则生成的样本一致。
- 使用原始 `ego_pose` 逆变换核对真值框中心，误差不超过三位小数导出的舍入范围。
- 在相同 Pillow 环境中重建相机 JPEG，文件字节一致。
- 单元测试确认世界系跟踪框、速度转入 LiDAR 系，并保持原始输出不变。

预测示例使用 `tools/export_forecast_examples.py` 从每个测试城镇的移动窗口中
等间隔抽取 12 个窗口，共 24 个；选择规则不读取预测误差。
已核对 `forecast_examples.json` 中的两个原数据 SHA-256 和 Ridge 模型 SHA-256，
与磁盘上的数据和模型一致。示例坐标保留四位小数，全量指标仍来自既有评估报告。

## 自动检查与人工查看

核心测试：**94 passed**。Python 检查使用 Python 3.12、NumPy 1.26.4；
传感器导出和同帧 JPEG 复核使用原 YOLO 环境（Python 3.11）。
Ruff、JavaScript 语法检查、`git diff --check` 均通过。
检查了页面的 36 个链接/资源引用，本地目标均存在，页面 ID 无重复。

浏览器使用 Playwright 1.63.0 / Chromium，运行仓库中的
[`tools/test_website.cjs`](../tools/test_website.cjs)：

- 两场景全部 42 个样本与图片；时间轴首尾；前后帧、连续查看及暂停。
- 点云、真值、跟踪框、历史与预测图层；下拉和点击选择目标；缩放、拖动、复位。
- 故意延迟旧帧相机响应和旧场景索引响应，确认最新选择不被旧响应覆盖。
- PNG、当前帧 JSON、性能 CSV 下载；三种测试范围与两类评估对象筛选。
- 24 个预测示例入口、两个城镇的误差数值、曲线切换与图像导出。
- 高速 MP4 实际解码播放，时长为 24 秒；运行配置禁用规则及剪贴板内容。
- 1440、768、390、320 px 宽度无页面横向溢出；手机导航可打开和收起。
- 单个数据集加载失败时其他功能可用；切场景失败清除旧帧标识；单帧失败保留之前画面。
- 无 JavaScript 时仍可查看视频、静态表和报告链接。

正常浏览过程无 JavaScript 异常或 HTTP 错误。
人工查看了桌面首屏、点云查看器、预测对照，以及手机首屏、查看器、命令面板截图。
另用页面生成的 60 帧 LiDAR + IMM 合成演示命令实际运行，确认可在新的嵌套目录写入报告。

CI 新增同一浏览器检查，保存桌面和手机截图；运行方法见
[中文工程说明](engineering_zh.md#开发与验证)。
