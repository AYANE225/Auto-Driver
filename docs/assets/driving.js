"use strict";

// This page replays Python closed-loop results. It does not simulate control.
(() => {
  const el = (id) => document.getElementById(id);
  const canvas = el("driving-canvas"),
    speedCanvas = el("driving-speed-chart");
  const scene = el("driving-scene"),
    input = el("driving-input"),
    slider = el("driving-time");
  const cache = new Map();
  let report = null,
    summary = null,
    index = 0,
    token = 0,
    started = false;
  let playing = false,
    animation = 0,
    lastTick = 0,
    playTime = 0;
  const statuses = {
    cruise: "沿路线巡航",
    avoidance: "绕行障碍",
    following: "保持跟车间距",
    yielding: "减速让行",
    stop_line: "停车线约束",
    goal_approach: "减速接近终点",
    goal_reached: "已到达并停车",
    emergency_stop: "紧急制动",
    stale_input: "感知超时 · 制动",
    invalid_input: "输入异常 · 制动",
  };
  const reasons = {
    collision: "碰撞",
    road_boundary: "道路边界",
    curvature: "曲率",
    lateral_acceleration: "横向加速度",
    stop_position: "停车位置",
    geometry: "路径几何",
  };
  function context(target) {
    const w = target.clientWidth,
      h = target.clientHeight,
      dpr = Math.min(devicePixelRatio || 1, 2);
    if (target.width !== Math.round(w * dpr))
      target.width = Math.round(w * dpr);
    if (target.height !== Math.round(h * dpr))
      target.height = Math.round(h * dpr);
    const ctx = target.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  function draw() {
    const { ctx, w, h } = context(canvas);
    ctx.fillStyle = "#0b1520";
    ctx.fillRect(0, 0, w, h);
    if (!report) {
      drawSpeed();
      return;
    }
    canvas.parentElement.querySelector(".candidate").hidden =
      !el("driving-candidates").checked;
    const frame = report.frames[index],
      display = DrivingReplay.sample(
        report.frames,
        playTime,
        el("driving-smooth").checked,
      ),
      ego = display.ego,
      path = report.reference_path;
    canvas.dataset.renderTime = String(playTime);
    canvas.dataset.egoX = String(ego.x);
    canvas.dataset.egoY = String(ego.y);
    canvas.dataset.egoYaw = String(ego.yaw);
    const xs = path.map((p) => p[0]),
      ys = path.map((p) => p[1]);
    const minX = Math.min(...xs) - 8,
      maxX = Math.max(...xs) + 8;
    const minY = Math.min(...ys) - report.road_half_width_m - 10,
      maxY = Math.max(...ys) + report.road_half_width_m + 10;
    const follow = el("driving-view-mode").value === "follow",
      scale =
        (follow
          ? Math.min((w - 30) / 32, (h - 165) / 18)
          : Math.min((w - 30) / (maxX - minX), (h - 165) / (maxY - minY))) *
        Number(el("driving-zoom").value),
      cx = follow ? ego.x : (minX + maxX) / 2,
      cy = follow ? ego.y : (minY + maxY) / 2,
      viewY = 105 + (h - 170) / 2;
    const project = ([x, y]) => [
      w / 2 + (x - cx) * scale,
      viewY - (y - cy) * scale,
    ];
    ctx.save();
    ctx.beginPath();
    ctx.rect(12, 100, w - 24, h - 165);
    ctx.clip();
    const line = (points, color, width = 1, dash = []) => {
      ctx.beginPath();
      points.forEach((p, i) => {
        const q = project(p);
        if (i) ctx.lineTo(...q);
        else ctx.moveTo(...q);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    const left = cx - w / scale,
      right = cx + w / scale,
      bottom = cy - h / scale,
      top = cy + h / scale;
    for (let x = Math.ceil(left / 10) * 10; x <= right; x += 10)
      line(
        [
          [x, bottom],
          [x, top],
        ],
        "#16232e",
        1,
      );
    for (let y = Math.ceil(bottom / 10) * 10; y <= top; y += 10)
      line(
        [
          [left, y],
          [right, y],
        ],
        "#16232e",
        1,
      );
    line(path, "#24343f", report.road_half_width_m * 2 * scale + 2);
    line(
      path,
      "#1c2a34",
      Math.max(1, report.road_half_width_m * 2 * scale - 2),
    );
    line(path, "#4b616d", 1, [6, 7]);
    if (report.route_search) {
      const graph = report.route_search;
      for (const [a, b] of graph.edges) {
        const blocked = graph.blocked_edges.some(
          ([x, y]) => a === x && b === y,
        );
        line(
          [graph.nodes[a], graph.nodes[b]],
          blocked ? "#df9b7c" : "#688393",
          blocked ? 3 : 1,
          [5, 5],
        );
      }
    }
    if (report.stop_s !== null) {
      let traveled = 0;
      for (let i = 1; i < path.length; i++) {
        const a = path[i - 1],
          b = path[i],
          length = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (traveled + length >= report.stop_s) {
          const u = (report.stop_s - traveled) / length,
            x = a[0] + u * (b[0] - a[0]),
            y = a[1] + u * (b[1] - a[1]),
            nx = -(b[1] - a[1]) / length,
            ny = (b[0] - a[0]) / length;
          line(
            [
              [
                x + nx * report.road_half_width_m,
                y + ny * report.road_half_width_m,
              ],
              [
                x - nx * report.road_half_width_m,
                y - ny * report.road_half_width_m,
              ],
            ],
            frame.signal === "red" ? "#ee987e" : "#90e0b8",
            3,
          );
          break;
        }
        traveled += length;
      }
    }
    if (el("driving-candidates").checked)
      for (const candidate of frame.candidates)
        line(candidate.xy, candidate.feasible ? "#68879860" : "#c8886650", 1);
    line(
      [
        ...report.frames.slice(0, index + 1).map((f) => [f.ego.x, f.ego.y]),
        [ego.x, ego.y],
      ],
      "#90e0b8",
      2.5,
    );
    line(frame.trajectory, "#ead99a", 2, [5, 3]);
    const box = (x, y, yaw, length, width, fill, stroke) => {
      const p = project([x, y]);
      ctx.save();
      ctx.translate(...p);
      ctx.rotate(-yaw);
      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.2;
      ctx.fillRect(
        (-length * scale) / 2,
        (-width * scale) / 2,
        length * scale,
        width * scale,
      );
      ctx.strokeRect(
        (-length * scale) / 2,
        (-width * scale) / 2,
        length * scale,
        width * scale,
      );
      ctx.restore();
    };
    for (const b of display.obstacles)
      box(b[0], b[1], b[6], b[3], b[4], "#df9b7c35", "#df9b7c");
    if (el("driving-tracks").checked)
      for (const b of frame.tracks)
        box(b[0], b[1], b[6], b[3], b[4], "transparent", "#91b8e3");
    const v = report.vehicle;
    const center = [
      ego.x + v.rear_to_center * Math.cos(ego.yaw),
      ego.y + v.rear_to_center * Math.sin(ego.yaw),
    ];
    box(...center, ego.yaw, v.length, v.width, "#90e0b8", "#b8f3d2");
    // Roof, front windscreen, brake lamps and front-wheel steering cue.
    const carPoint = (x, y) => [
      ego.x + x * Math.cos(ego.yaw) - y * Math.sin(ego.yaw),
      ego.y + x * Math.sin(ego.yaw) + y * Math.cos(ego.yaw),
    ];
    box(
      ...carPoint(v.rear_to_center, 0),
      ego.yaw,
      v.length * 0.42,
      v.width * 0.72,
      "#244c47",
      "#447469",
    );
    for (const side of [-1, 1]) {
      for (const axle of [0, v.wheelbase])
        box(
          ...carPoint(axle, side * v.width * 0.48),
          ego.yaw + (axle ? ego.steering : 0),
          0.65,
          0.22,
          "#07121b",
          "#728f95",
        );
      box(
        ...carPoint(v.rear_to_center - v.length * 0.46, side * v.width * 0.32),
        ego.yaw,
        0.12,
        0.35,
        frame.command.acceleration < -0.3 ? "#ff796c" : "#ae625c",
        "transparent",
      );
    }
    line(
      [
        center,
        [
          center[0] + v.length * 0.8 * Math.cos(ego.yaw),
          center[1] + v.length * 0.8 * Math.sin(ego.yaw),
        ],
      ],
      "#d5f8e8",
      2,
    );
    const goal = project(path[path.length - 1]);
    ctx.strokeStyle = "#90e0b8";
    ctx.beginPath();
    ctx.arc(...goal, 5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "#a9bfcb";
    ctx.font = "10px sans-serif";
    ctx.fillText("终点", goal[0] - 12, goal[1] - 13);
    ctx.restore();
    ctx.fillStyle = "#a9bfcb";
    ctx.font = "10px sans-serif";
    const rulerMeters = scale > 16 ? 5 : 10,
      ruler = rulerMeters * scale;
    ctx.strokeStyle = "#688393";
    ctx.beginPath();
    ctx.moveTo(w - 25 - ruler, h - 50);
    ctx.lineTo(w - 25, h - 50);
    ctx.stroke();
    ctx.fillText(`${rulerMeters} m`, w - 25 - ruler, h - 58);
    drawSpeed();
  }
  function drawSpeed() {
    const { ctx, w, h } = context(speedCanvas);
    ctx.clearRect(0, 0, w, h);
    if (!report) return;
    const kind = el("driving-chart").value,
      values = report.frames.map((f) =>
        kind === "speed"
          ? f.ego.speed
          : kind === "steering"
            ? (f.command.steering * 180) / Math.PI
            : f.command.acceleration,
      ),
      last = report.frames[report.frames.length - 1].time_s,
      low = Math.min(0, ...values),
      high = Math.max(kind === "speed" ? 4 : 1, ...values),
      margin = (high - low) * 0.12,
      px = (t) => 36 + ((w - 50) * t) / Math.max(last, 0.1),
      py = (value) =>
        h -
        23 -
        ((h - 37) * (value - low + margin)) / (high - low + 2 * margin);
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#8ba2b2";
    for (let i = 0; i <= 4; i++) {
      const value = low + ((high - low) * i) / 4;
      ctx.strokeStyle = "#293946";
      ctx.beginPath();
      ctx.moveTo(34, py(value));
      ctx.lineTo(w - 12, py(value));
      ctx.stroke();
      ctx.fillText(value.toFixed(1), 2, py(value) + 3);
    }
    ctx.beginPath();
    report.frames.forEach((f, i) => {
      if (!i) ctx.moveTo(px(f.time_s), py(values[i]));
      else {
        if (kind !== "speed") ctx.lineTo(px(f.time_s), py(values[i - 1]));
        ctx.lineTo(px(f.time_s), py(values[i]));
      }
    });
    ctx.strokeStyle = {
      speed: "#90e0b8",
      acceleration: "#e5b493",
      steering: "#91b8e3",
    }[kind];
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.strokeStyle = "#ead99a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px(playTime), 10);
    ctx.lineTo(px(playTime), h - 20);
    ctx.stroke();
    ctx.fillText("0 s", 34, h - 4);
    ctx.fillText(`${last.toFixed(1)} s`, Math.max(40, w - 52), h - 4);
    speedCanvas.dataset.series = kind;
  }
  function showFrame() {
    if (!report) return;
    index = Math.max(0, Math.min(report.frames.length - 1, index));
    const f = report.frames[index];
    slider.value = String(index);
    slider.setAttribute(
      "aria-valuetext",
      `${f.time_s.toFixed(1)} 秒，${statuses[f.status] || f.status}`,
    );
    canvas.dataset.frame = String(index);
    canvas.dataset.scene = report.scenario;
    canvas.dataset.input = report.detector;
    el("driving-speed").replaceChildren(
      document.createTextNode(f.ego.speed.toFixed(1) + " "),
    );
    const units = document.createElement("small");
    units.textContent = "m/s";
    el("driving-speed").append(units);
    el("driving-status").textContent = statuses[f.status] || f.status;
    el("driving-steer").textContent =
      `${((f.command.steering * 180) / Math.PI).toFixed(1)}°`;
    el("driving-accel").textContent =
      `${f.command.acceleration.toFixed(1)} m/s²`;
    el("driving-progress").textContent = `${f.route_s.toFixed(1)} m`;
    el("driving-count").textContent =
      `${f.candidates.filter((c) => c.feasible).length} / ${f.candidates.length} 可行`;
    el("driving-signal").textContent =
      { red: "红灯", green: "绿灯", none: "无信号约束" }[f.signal] || "—";
    el("driving-rejections").textContent =
      Object.entries(f.rejected)
        .map(([key, value]) => `${reasons[key] || key}拒绝 ${value} 条`)
        .join(" · ") || "当前候选均通过检查。";
    if (!f.feasible)
      el("driving-rejections").textContent =
        "当前执行制动回退；不将回退轨迹标为可行解。";
    el("driving-time-label").textContent = `${playTime.toFixed(1)} s`;
    el("driving-previous").disabled = index === 0;
    el("driving-next").disabled = index === report.frames.length - 1;
    const buttons = [...el("driving-events").querySelectorAll("button")];
    const active = buttons
      .filter((b) => Number(b.dataset.index) <= index)
      .at(-1);
    buttons.forEach((b) =>
      b.setAttribute("aria-pressed", String(b === active)),
    );
    draw();
  }
  function pause() {
    playing = false;
    cancelAnimationFrame(animation);
    animation = 0;
    el("driving-play").textContent = "播放驾驶";
    el("driving-play").setAttribute("aria-pressed", "false");
  }
  function tick(time) {
    if (!playing || !report) return;
    if (lastTick)
      playTime +=
        Math.max(0, (time - lastTick) / 1000) *
        Number(el("driving-rate").value);
    lastTick = time;
    playTime = Math.min(playTime, report.frames.at(-1).time_s);
    const before = index;
    index = DrivingReplay.frameIndex(report.frames, playTime);
    if (index !== before) showFrame();
    else {
      el("driving-time-label").textContent = `${playTime.toFixed(1)} s`;
      draw();
    }
    if (index === report.frames.length - 1) {
      pause();
      return;
    }
    animation = requestAnimationFrame(tick);
  }
  el("driving-play").addEventListener("click", () => {
    if (playing) {
      pause();
      return;
    }
    if (!report) return;
    if (index === report.frames.length - 1) {
      index = 0;
      playTime = report.frames[0].time_s;
    }
    lastTick = 0;
    playing = true;
    showFrame();
    el("driving-play").textContent = "暂停驾驶";
    el("driving-play").setAttribute("aria-pressed", "true");
    animation = requestAnimationFrame(tick);
  });
  function seek(value) {
    if (!report) return;
    pause();
    index = Math.max(0, Math.min(report.frames.length - 1, value));
    playTime = report.frames[index].time_s;
    showFrame();
  }
  slider.addEventListener("input", () => seek(Number(slider.value)));
  el("driving-previous").addEventListener("click", () => seek(index - 1));
  el("driving-next").addEventListener("click", () => seek(index + 1));
  for (const id of [
    "driving-view-mode",
    "driving-zoom",
    "driving-smooth",
    "driving-chart",
  ])
    el(id).addEventListener("input", () => {
      speedCanvas.setAttribute(
        "aria-label",
        el("driving-chart").selectedOptions[0].textContent +
          "；点击或使用方向键定位时刻",
      );
      draw();
    });
  speedCanvas.addEventListener("click", (event) => {
    if (!report) return;
    const rect = speedCanvas.getBoundingClientRect();
    const time =
      Math.max(
        0,
        Math.min(1, (event.clientX - rect.left - 36) / (rect.width - 50)),
      ) * report.frames.at(-1).time_s;
    seek(DrivingReplay.frameIndex(report.frames, time));
  });
  el("driving-save").addEventListener("click", () => {
    if (!report) return;
    const a = document.createElement("a");
    a.download = `${report.scenario}-${report.detector}-${playTime.toFixed(2)}s.png`;
    a.href = canvas.toDataURL("image/png");
    a.click();
  });
  for (const id of ["driving-candidates", "driving-tracks"])
    el(id).addEventListener("change", draw);
  function keyboardSeek(event) {
    if (
      !report ||
      !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
    )
      return;
    event.preventDefault();
    seek(
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? report.frames.length - 1
          : index + (event.key === "ArrowRight" ? 1 : -1),
    );
  }
  canvas.addEventListener("keydown", keyboardSeek);
  speedCanvas.addEventListener("keydown", keyboardSeek);
  async function read(url) {
    if (!cache.has(url)) cache.set(url, await readReport(url));
    return cache.get(url);
  }
  async function load() {
    const request = ++token,
      selected = scene.value,
      mode = input.value;
    pause();
    report = null;
    slider.disabled = true;
    el("driving-play").disabled = true;
    delete canvas.dataset.scene;
    delete canvas.dataset.input;
    delete canvas.dataset.frame;
    for (const key of ["renderTime", "egoX", "egoY", "egoYaw"])
      delete canvas.dataset[key];
    for (const id of ["driving-save", "driving-previous", "driving-next"])
      el(id).disabled = true;
    el("driving-events").replaceChildren();
    slider.value = "0";
    slider.removeAttribute("aria-valuetext");
    el("driving-time-label").textContent = "—";
    el("driving-status").textContent = "正在读取闭环运行记录…";
    for (const id of [
      "driving-speed",
      "driving-steer",
      "driving-accel",
      "driving-progress",
      "driving-count",
      "driving-signal",
    ])
      el(id).textContent = "—";
    el("driving-title").textContent = scene.selectedOptions[0].textContent;
    el("driving-metrics").textContent = "正在读取本次运行指标…";
    el("driving-rejections").textContent = "等待运行记录。";
    el("driving-download").href = `assets/driving/${mode}/${selected}.json`;
    el("driving-command").textContent =
      `python tools/run_driving.py --scenario ${selected} --detector ${mode} --out outputs/driving-${selected}-${mode}`;
    draw();
    try {
      const data = await read(`assets/driving/${mode}/${selected}.json`);
      if (request !== token) return;
      report = data;
      index = 0;
      playTime = report.frames[0].time_s;
      el("driving-save").disabled = false;
      el("driving-events").replaceChildren(
        ...DrivingReplay.events(report.frames, statuses).map((event) => {
          const button = document.createElement("button");
          button.type = "button";
          button.dataset.index = String(event.index);
          button.textContent = `${event.time.toFixed(1)} s · ${event.label}`;
          button.addEventListener("click", () => seek(event.index));
          return button;
        }),
      );
      slider.max = String(report.frames.length - 1);
      slider.disabled = false;
      el("driving-play").disabled = false;
      el("driving-title").textContent = report.title;
      el("driving-download").href = `assets/driving/${mode}/${selected}.json`;
      el("driving-command").textContent =
        `python tools/run_driving.py --scenario ${selected} --detector ${mode} --out outputs/driving-${selected}-${mode}`;
      const m = report.metrics;
      el("driving-metrics").replaceChildren(
        ...[
          [m.success ? "通过" : "未通过", "场景验收"],
          [String(m.collision_samples), "碰撞采样数 · 50 Hz"],
          [m.max_lateral_offset_m.toFixed(2) + " m", "最大路线横向偏移"],
          [m.planning_control_ms.p95.toFixed(1) + " ms", "规划 + 控制 p95"],
        ].map(([value, label]) => {
          const div = document.createElement("div"),
            strong = document.createElement("strong"),
            span = document.createElement("span");
          strong.textContent = value;
          span.textContent = label;
          div.append(strong, span);
          return div;
        }),
      );
      showFrame();
    } catch {
      if (request === token) {
        el("driving-status").textContent =
          "运行记录读取失败，请切换场景或输入重试。";
        el("driving-metrics").textContent = "可从下方报告链接查看数据。";
      }
    }
  }
  async function loadSummary() {
    try {
      summary = await read("assets/driving/index.json");
      el("driving-results-body").replaceChildren(
        ...summary.modes.gt.scenarios.map((row) => {
          const tr = document.createElement("tr"),
            name = document.createElement("th");
          name.scope = "row";
          name.textContent = row.title;
          tr.append(name);
          for (const mode of ["gt", "lidar"]) {
            const td = document.createElement("td"),
              entry = summary.modes[mode].scenarios.find(
                (s) => s.name === row.name,
              ),
              link = document.createElement("a");
            link.textContent = entry.metrics.success
              ? "通过 · 查看记录"
              : "未通过 · 查看记录";
            link.className = entry.metrics.success
              ? "driving-pass"
              : "driving-fail";
            link.href = `assets/driving/${entry.file}`;
            td.append(link);
            tr.append(td);
          }
          return tr;
        }),
      );
    } catch {
      el("driving-results-body").textContent =
        "指标汇总读取失败，可打开上方 JSON 报告。";
    }
  }
  for (const select of [scene, input]) select.addEventListener("change", load);
  el("driving-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(el("driving-command").textContent);
      el("driving-copy-status").textContent = "已复制";
    } catch {
      el("driving-copy-status").textContent = "请选中命令手动复制。";
    }
  });
  new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
      if (!started) {
        started = true;
        load();
        loadSummary();
      }
    } else pause();
  }).observe(el("driving"));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pause();
  });
  new ResizeObserver(draw).observe(canvas);
})();

// Match actuator feedback to the camera sample actually presented by the video.
for (const mode of ["gt", "lidar"]) {
  const video = document.getElementById(`driving-carla-${mode}-video`),
    panel = document.getElementById(`driving-carla-${mode}-telemetry`);
  let report,
    pending,
    callback = 0;
  const value = (name) => panel.querySelector(`[data-value="${name}"]`);
  function show(time) {
    if (!report) return;
    // CARLA float32 timestamps differ slightly from encoded video PTS.
    const index = DrivingReplay.frameIndex(
        report.frames,
        time,
        "video_time_s",
        1e-5,
      ),
      frame = report.frames[index];
    panel.dataset.frame = String(index);
    value("speed").textContent = `${frame.ego.speed.toFixed(2)} m/s`;
    value("throttle").textContent =
      `${(frame.command.throttle * 100).toFixed(0)}%`;
    value("brake").textContent = `${(frame.command.brake * 100).toFixed(0)}%`;
    value("status").textContent =
      {
        cruise: "沿车道行驶",
        goal_approach: "减速接近终点",
        goal_reached: "到达并停车",
        yielding: "减速让行",
        emergency_stop: "紧急制动",
        stale_input: "观测超时制动",
      }[frame.status] || frame.status;
    value("time").textContent =
      `记录 ${frame.time_s.toFixed(1)} s · 转向 ${((frame.command.steering * 180) / Math.PI).toFixed(1)}° · 路线进度 ${frame.route_s.toFixed(1)} m`;
  }
  async function load() {
    if (report) return;
    if (!pending)
      pending = readReport(`assets/driving/carla_${mode}.json`)
        .then((data) => {
          report = data;
          show(video.currentTime);
        })
        .catch(() => {
          value("status").textContent =
            "控制记录读取失败；再次播放或拖动视频可重试。";
        })
        .finally(() => {
          pending = null;
        });
    await pending;
  }
  function next() {
    if (callback || video.paused || !video.requestVideoFrameCallback) return;
    callback = video.requestVideoFrameCallback((_, metadata) => {
      callback = 0;
      show(metadata.mediaTime);
      next();
    });
  }
  video.addEventListener("play", () => {
    load();
    next();
  });
  for (const name of ["loadeddata", "seeked"])
    video.addEventListener(name, async () => {
      await load();
      show(video.currentTime);
    });
  video.addEventListener("timeupdate", () => {
    if (!video.requestVideoFrameCallback) show(video.currentTime);
  });
  video.addEventListener("pause", () => {
    if (callback) video.cancelVideoFrameCallback(callback);
    callback = 0;
    show(video.currentTime);
  });
}
