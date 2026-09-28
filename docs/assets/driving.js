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
    target.width = Math.round(w * dpr);
    target.height = Math.round(h * dpr);
    const ctx = target.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }
  function draw() {
    const { ctx, w, h } = context(canvas);
    ctx.fillStyle = "#0b1520";
    ctx.fillRect(0, 0, w, h);
    if (!report) return;
    const frame = report.frames[index],
      path = report.reference_path;
    const xs = path.map((p) => p[0]),
      ys = path.map((p) => p[1]);
    const minX = Math.min(...xs) - 8,
      maxX = Math.max(...xs) + 8;
    const minY = Math.min(...ys) - report.road_half_width_m - 10,
      maxY = Math.max(...ys) + report.road_half_width_m + 10;
    const scale = Math.min((w - 30) / (maxX - minX), (h - 135) / (maxY - minY));
    const project = ([x, y]) => [
      w / 2 + (x - (minX + maxX) / 2) * scale,
      h * 0.59 - (y - (minY + maxY) / 2) * scale,
    ];
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
    for (let x = Math.ceil(minX / 10) * 10; x <= maxX; x += 10)
      line(
        [
          [x, minY],
          [x, maxY],
        ],
        "#16232e",
        1,
      );
    for (let y = Math.ceil(minY / 10) * 10; y <= maxY; y += 10)
      line(
        [
          [minX, y],
          [maxX, y],
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
      report.frames.slice(0, index + 1).map((f) => [f.ego.x, f.ego.y]),
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
    for (const b of frame.obstacles)
      box(b[0], b[1], b[6], b[3], b[4], "#df9b7c35", "#df9b7c");
    if (el("driving-tracks").checked)
      for (const b of frame.tracks)
        box(b[0], b[1], b[6], b[3], b[4], "transparent", "#91b8e3");
    const ego = frame.ego,
      v = report.vehicle;
    const center = [
      ego.x + v.rear_to_center * Math.cos(ego.yaw),
      ego.y + v.rear_to_center * Math.sin(ego.yaw),
    ];
    box(...center, ego.yaw, v.length, v.width, "#90e0b8", "#b8f3d2");
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
    const ruler = 10 * scale;
    ctx.strokeStyle = "#688393";
    ctx.beginPath();
    ctx.moveTo(w - 25 - ruler, h - 50);
    ctx.lineTo(w - 25, h - 50);
    ctx.stroke();
    ctx.fillText("10 m", w - 25 - ruler, h - 58);
    drawSpeed();
  }
  function drawSpeed() {
    const { ctx, w, h } = context(speedCanvas);
    ctx.clearRect(0, 0, w, h);
    if (!report) return;
    const last = report.frames[report.frames.length - 1].time_s,
      top = Math.max(4, ...report.frames.map((f) => f.ego.speed)) * 1.12;
    const px = (t) => 30 + ((w - 45) * t) / Math.max(last, 0.1),
      py = (v) => h - 20 - ((h - 47) * v) / top;
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "#8ba2b2";
    for (let value = 0; value <= top; value += 2) {
      ctx.strokeStyle = "#293946";
      ctx.beginPath();
      ctx.moveTo(28, py(value));
      ctx.lineTo(w - 12, py(value));
      ctx.stroke();
      ctx.fillText(String(value), 8, py(value) + 3);
    }
    ctx.beginPath();
    report.frames.forEach((f, i) => {
      if (i) ctx.lineTo(px(f.time_s), py(f.ego.speed));
      else ctx.moveTo(px(f.time_s), py(f.ego.speed));
    });
    ctx.strokeStyle = "#90e0b8";
    ctx.lineWidth = 2;
    ctx.stroke();
    const f = report.frames[index];
    ctx.strokeStyle = "#ead99a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px(f.time_s), 20);
    ctx.lineTo(px(f.time_s), h - 20);
    ctx.stroke();
    ctx.fillStyle = "#8ba2b2";
    ctx.fillText("0 s", 28, h - 4);
    ctx.fillText(`${last.toFixed(1)} s`, Math.max(40, w - 52), h - 4);
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
    el("driving-time-label").textContent = `${f.time_s.toFixed(1)} s`;
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
        Math.min(0.15, (time - lastTick) / 1000) *
        Number(el("driving-rate").value);
    lastTick = time;
    const before = index;
    while (
      index < report.frames.length - 1 &&
      report.frames[index + 1].time_s <= playTime
    )
      index++;
    if (index !== before) showFrame();
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
    if (index === report.frames.length - 1) index = 0;
    playTime = report.frames[index].time_s;
    lastTick = 0;
    playing = true;
    showFrame();
    el("driving-play").textContent = "暂停驾驶";
    el("driving-play").setAttribute("aria-pressed", "true");
    animation = requestAnimationFrame(tick);
  });
  slider.addEventListener("input", () => {
    pause();
    index = Number(slider.value);
    showFrame();
  });
  for (const id of ["driving-candidates", "driving-tracks"])
    el(id).addEventListener("change", draw);
  canvas.addEventListener("keydown", (event) => {
    if (
      !report ||
      !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
    )
      return;
    event.preventDefault();
    pause();
    index =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? report.frames.length - 1
          : index + (event.key === "ArrowRight" ? 1 : -1);
    showFrame();
  });
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
