"use strict";

// Interactive views of committed recordings and reports. No inference runs here.
(() => {
  const canvas = byId("cloud-canvas");
  const sceneSelect = byId("cloud-scene");
  const frameSlider = byId("cloud-frame");
  const zoomSlider = byId("cloud-zoom");
  const cache = new Map();
  let manifest = null,
    cloudManifest = null,
    current = null,
    displayedIndex = 6,
    requestId = 0;
  let azimuth = 3.65,
    elevation = 0.68,
    drag = null;
  let rotating = false,
    visible = false,
    animation = 0,
    lastTime = 0;
  const read = async (url) => {
    if (!cache.has(url)) cache.set(url, await readReport(url));
    return cache.get(url);
  };
  function draw() {
    const width = canvas.clientWidth,
      height = canvas.clientHeight;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bg = ctx.createRadialGradient(
      width / 2,
      height * 0.7,
      0,
      width / 2,
      height * 0.5,
      width * 0.65,
    );
    bg.addColorStop(0, "#152c2b");
    bg.addColorStop(1, "#09121c");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);
    const ca = Math.cos(azimuth),
      sa = Math.sin(azimuth);
    const ce = Math.cos(elevation),
      se = Math.sin(elevation);
    const distance = 94 / Number(zoomSlider.value),
      focal = Math.min(width * 0.92, height * 1.65);
    const project = ([x, y, z]) => {
      const dx = x - 10,
        dz = z + 1;
      const depth = distance - ce * ca * dx - ce * sa * y - se * dz;
      if (depth < 2) return null;
      return [
        width / 2 + ((-sa * dx + ca * y) * focal) / depth,
        height * 0.53 -
          ((-se * ca * dx - se * sa * y + ce * dz) * focal) / depth,
        depth,
      ];
    };
    const line = (points, color, weight = 1, dash = []) => {
      ctx.beginPath();
      let started = false;
      points.forEach((p) => {
        const q = project(p);
        if (!q) {
          started = false;
          return;
        }
        if (started) ctx.lineTo(q[0], q[1]);
        else ctx.moveTo(q[0], q[1]);
        started = true;
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = weight;
      ctx.setLineDash(dash);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    for (let v = -50; v <= 60; v += 10) {
      line(
        [
          [v, -50, -2],
          [v, 50, -2],
        ],
        "#29433f",
        0.7,
      );
      line(
        [
          [-50, v, -2],
          [60, v, -2],
        ],
        "#29433f",
        0.7,
      );
    }
    line(
      [
        [0, 0, -1.9],
        [12, 0, -1.9],
      ],
      "#dcb48b",
      2,
    );
    line(
      [
        [0, 0, -1.9],
        [0, 12, -1.9],
      ],
      "#8ab8e7",
      2,
    );
    line(
      [
        [0, 0, -1.9],
        [0, 0, 7],
      ],
      "#90e0b8",
      2,
    );
    ctx.font = "11px sans-serif";
    [
      ["x", [13, 0, -1.9]],
      ["y", [0, 13, -1.9]],
      ["z", [0, 0, 8]],
    ].forEach(([label, p]) => {
      const q = project(p);
      if (q) {
        ctx.fillStyle = "#c0d7d6";
        ctx.fillText(label, q[0], q[1]);
      }
    });
    if (!current) return;
    const limit = Number(byId("cloud-density").value);
    const displayed = current.cloudPoints.filter(
      (_, i, a) =>
        Math.floor((i * limit) / a.length) !==
        Math.floor(((i - 1) * limit) / a.length),
    );
    const points = displayed
      .map((p) => ({ p, q: project(p) }))
      .filter(({ q }) => q)
      .sort((a, b) => b.q[2] - a.q[2]);
    for (const { p, q } of points) {
      if (q[0] < 0 || q[0] > width || q[1] < 0 || q[1] > height) continue;
      const h = Math.max(0, Math.min(1, (p[2] + 2.4) / 5.4));
      ctx.fillStyle = `hsl(${196 - 46 * h} 65% ${35 + 40 * h}%)`;
      const size = Math.max(1.1, Math.min(3.8, 160 / q[2]));
      ctx.fillRect(q[0], q[1], size, size);
    }
    const color = (id) => `hsl(${(id * 137.508 + 30) % 360} 75% 70%)`;
    const box = (b, stroke, dashed = false) => {
      const [x, y, z, l, w, h, yaw] = b,
        c = Math.cos(yaw),
        s = Math.sin(yaw);
      const corners = [];
      for (const dz of [-h / 2, h / 2])
        for (const [dx, dy] of [
          [l / 2, w / 2],
          [-l / 2, w / 2],
          [-l / 2, -w / 2],
          [l / 2, -w / 2],
        ])
          corners.push([x + c * dx - s * dy, y + s * dx + c * dy, z + dz]);
      for (const [a, b] of [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
      ])
        line(
          [corners[a], corners[b]],
          stroke,
          dashed ? 1 : 1.8,
          dashed ? [4, 4] : [],
        );
    };
    if (byId("cloud-gt").checked)
      current.ground_truth.forEach((t) => box(t.box, "#dfbd80", true));
    current.tracks.forEach((t) => {
      if (byId("cloud-paths").checked)
        line(
          t.history.map(([x, y]) => [x, y, t.box[2] - t.box[5] / 2]),
          color(t.id),
          1.6,
          [3, 3],
        );
      if (!byId("cloud-tracks").checked) return;
      box(t.box, color(t.id));
      const q = project([t.box[0], t.box[1], t.box[2] + t.box[5] / 2 + 0.5]);
      if (q) {
        ctx.fillStyle = color(t.id);
        ctx.fillText(`#${t.id}`, q[0] + 3, q[1] - 3);
      }
    });
  }
  function describe() {
    if (!current) return;
    const count = Math.min(
      current.cloudPoints.length,
      Number(byId("cloud-density").value),
    );
    canvas.dataset.points = String(count);
    byId("cloud-status").textContent =
      `原始 ${current.point_count.toLocaleString("zh-CN")} 点 → 显示 ${count.toLocaleString("zh-CN")} 点 · ${current.tracks.length} 条确认轨迹 · ${current.ground_truth.length} 个区域内真值目标${current.denseAvailable ? "" : " · 精细点云不可用，使用已有轻量样本"}`;
  }
  byId("cloud-density").addEventListener("change", () => {
    describe();
    draw();
  });
  const setFrameDisabled = (disabled) => {
    frameSlider.disabled = disabled;
    byId("cloud-save").disabled = disabled;
  };
  async function loadFrame(index) {
    if (!manifest) return;
    const token = ++requestId,
      active = manifest,
      dense = cloudManifest,
      scene = sceneSelect.value;
    byId("cloud-status").textContent = "正在读取点云和同帧相机…";
    try {
      const row = active.frames[index],
        base = `assets/replay/${scene}/`;
      const payload = await read(base + row.file);
      const img = new Image();
      img.src = base + payload.image;
      let cloudPoints = payload.points,
        denseAvailable = false;
      const loadDense = async () => {
        const row = dense?.frames.find((f) => f.frame_id === payload.frame_id);
        if (!row) return;
        try {
          const key = `assets/cloud/${scene}/${row.file}`;
          if (!cache.has(key)) {
            const response = await fetch(key);
            if (!response.ok) throw new Error("Point cloud unavailable");
            const bytes = await response.arrayBuffer();
            if (bytes.byteLength !== row.points * 6)
              throw new Error("Point count mismatch");
            const values = new DataView(bytes),
              decoded = [];
            for (let i = 0; i < row.points; i++)
              decoded.push(
                [0, 1, 2].map(
                  (k) => values.getInt16(i * 6 + k * 2, true) * dense.scale_m,
                ),
              );
            cache.set(key, decoded);
          }
          cloudPoints = cache.get(key);
          denseAvailable = true;
        } catch {
          /* The existing 2,500-point replay remains a usable fallback. */
        }
      };
      await Promise.all([img.decode(), loadDense()]);
      if (token !== requestId) return;
      current = { ...payload, cloudPoints, denseAvailable };
      displayedIndex = index;
      frameSlider.value = String(index);
      canvas.dataset.frame = String(payload.frame_id);
      canvas.dataset.scene = scene;
      byId("cloud-camera").src = img.src;
      byId("cloud-camera").hidden = false;
      byId("cloud-frame-label").textContent =
        `${sceneNames[scene]} · 帧 ${payload.frame_id} · ${payload.elapsed_s.toFixed(1)} s`;
      frameSlider.setAttribute(
        "aria-valuetext",
        `第 ${payload.frame_id} 帧，${payload.elapsed_s.toFixed(1)} 秒`,
      );
      describe();
      setFrameDisabled(false);
      draw();
    } catch {
      if (token !== requestId) return;
      frameSlider.value = String(displayedIndex);
      byId("cloud-status").textContent =
        "当前帧读取失败，请切换采样帧或场景重试。已有画面保持不变。";
      frameSlider.disabled = false;
    }
  }
  async function loadScene() {
    const token = ++requestId,
      scene = sceneSelect.value;
    current = null;
    manifest = null;
    cloudManifest = null;
    setFrameDisabled(true);
    draw();
    delete canvas.dataset.frame;
    delete canvas.dataset.scene;
    byId("cloud-camera").hidden = true;
    byId("cloud-frame-label").textContent = "正在读取场景";
    byId("cloud-status").textContent = "正在读取场景数据…";
    try {
      const [data, dense] = await Promise.all([
        read(`assets/replay/${scene}/index.json`),
        read(`assets/cloud/${scene}/index.json`).catch(() => null),
      ]);
      if (token !== requestId) return;
      manifest = data;
      cloudManifest = dense;
      frameSlider.max = String(data.frames.length - 1);
      displayedIndex = Math.min(6, data.frames.length - 1);
      frameSlider.value = String(displayedIndex);
      await loadFrame(displayedIndex);
    } catch {
      if (token === requestId)
        byId("cloud-status").textContent =
          "场景读取失败，请切换场景重试。下方视频与逐帧查看器仍可使用。";
    }
  }
  function tick(time) {
    animation = 0;
    if (!rotating || !visible || document.hidden) return;
    if (lastTime && time - lastTime < 40) {
      animation = requestAnimationFrame(tick);
      return;
    }
    if (lastTime) azimuth += Math.min(0.1, (time - lastTime) / 1000) * 0.18;
    lastTime = time;
    draw();
    animation = requestAnimationFrame(tick);
  }
  function schedule() {
    lastTime = 0;
    if (!animation && rotating && visible && !document.hidden)
      animation = requestAnimationFrame(tick);
  }
  function stopRotation() {
    rotating = false;
    cancelAnimationFrame(animation);
    animation = 0;
    byId("cloud-spin").textContent = "环绕观察";
    byId("cloud-spin").setAttribute("aria-pressed", "false");
  }
  byId("cloud-spin").addEventListener("click", () => {
    if (rotating) stopRotation();
    else {
      rotating = true;
      byId("cloud-spin").textContent = "暂停环绕";
      byId("cloud-spin").setAttribute("aria-pressed", "true");
      schedule();
    }
  });
  const presets = {
    orbit: [3.65, 0.68],
    top: [Math.PI, 1.54],
    front: [Math.PI, 0.12],
  };
  document.querySelectorAll("[data-cloud-view]").forEach((b) =>
    b.addEventListener("click", () => {
      stopRotation();
      [azimuth, elevation] = presets[b.dataset.cloudView];
      zoomSlider.value = "1";
      selectButton("data-cloud-view", b.dataset.cloudView);
      draw();
    }),
  );
  canvas.addEventListener("pointerdown", (event) => {
    stopRotation();
    drag = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    azimuth -= (event.clientX - drag[0]) * 0.008;
    elevation = Math.max(
      0.08,
      Math.min(1.54, elevation + (event.clientY - drag[1]) * 0.006),
    );
    drag = [event.clientX, event.clientY];
    selectButton("data-cloud-view", "");
    draw();
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
    canvas.addEventListener(name, () => {
      drag = null;
    });
  function changeZoom(delta) {
    zoomSlider.value = String(
      Math.max(0.6, Math.min(2.5, Number(zoomSlider.value) + delta)),
    );
    draw();
  }
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      changeZoom(event.deltaY < 0 ? 0.1 : -0.1);
    },
    { passive: false },
  );
  canvas.addEventListener("keydown", (event) => {
    if (
      ![
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "+",
        "=",
        "-",
      ].includes(event.key)
    )
      return;
    event.preventDefault();
    stopRotation();
    if (["+", "="].includes(event.key)) changeZoom(0.1);
    else if (event.key === "-") changeZoom(-0.1);
    else {
      azimuth +=
        event.key === "ArrowLeft"
          ? -0.12
          : event.key === "ArrowRight"
            ? 0.12
            : 0;
      elevation = Math.max(
        0.08,
        Math.min(
          1.54,
          elevation +
            (event.key === "ArrowUp"
              ? 0.08
              : event.key === "ArrowDown"
                ? -0.08
                : 0),
        ),
      );
      selectButton("data-cloud-view", "");
      draw();
    }
  });
  sceneSelect.addEventListener("change", loadScene);
  frameSlider.addEventListener("input", () =>
    loadFrame(Number(frameSlider.value)),
  );
  for (const id of ["cloud-zoom", "cloud-gt", "cloud-tracks", "cloud-paths"])
    byId(id).addEventListener("input", draw);
  byId("cloud-save").addEventListener("click", () => {
    if (!current) return;
    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/png");
    link.download = `${canvas.dataset.scene}_3d_${current.frame_id}.png`;
    link.click();
  });
  let loaded = false;
  new IntersectionObserver((entries) => {
    visible = entries[0].isIntersecting;
    if (visible && !loaded) {
      loaded = true;
      loadScene();
    }
    if (visible) schedule();
    else {
      cancelAnimationFrame(animation);
      animation = 0;
    }
  }).observe(canvas);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cancelAnimationFrame(animation);
      animation = 0;
    } else schedule();
  });
  new ResizeObserver(draw).observe(canvas);
})();

(async () => {
  const stages = [
    [
      "detect_ms",
      "LiDAR 检测",
      "裁剪点云、移除地面，再用加权体素 DBSCAN 聚类与 PCA 拟合框。当前它占新流水线的大部分时间。",
    ],
    [
      "fuse_ms",
      "相机确认",
      "运行 YOLOv8n 并将 LiDAR 框投影到图像，利用二维框确认目标。",
    ],
    [
      "track_ms",
      "坐标转换与跟踪",
      "批量 C++ IoU 减少旋转框交集计算开销；这一阶段也包含将检测框转换到世界坐标系。",
    ],
    [
      "predict_ms",
      "运动预测",
      "根据已确认轨迹，用 CV / CTRV 生成未来位置；它没有使用未来真值。",
    ],
  ];
  let scene = "urban",
    selected = "track_ms";
  const buttons = document.querySelectorAll(
    "[data-latency-scene],.stage-legend button",
  );
  buttons.forEach((b) => {
    b.disabled = true;
  });
  try {
    const reports = Object.fromEntries(
      await Promise.all(
        ["urban", "highway"].map(async (s) => [
          s,
          await Promise.all(
            ["python_original", "cpp_threshold"].map((v) =>
              readReport(`benchmarks/tracking/${s}_${v}.json`),
            ),
          ),
        ]),
      ),
    );
    function show() {
      const [oldReport, newReport] = reports[scene];
      [oldReport, newReport].forEach((report, index) => {
        const name = index ? "new" : "old",
          stack = byId(`stage-${name}`);
        byId(`stage-${name}-total`).textContent =
          `${report.latency_ms.total_ms.mean.toFixed(1)} ms`;
        stack.replaceChildren(
          ...stages.map(([key, label]) => {
            const b = document.createElement("button"),
              ms = report.latency_ms[key].mean;
            b.type = "button";
            b.dataset.stage = key;
            b.style.width = `${ms / 2}%`;
            b.classList.toggle("selected", key === selected);
            b.setAttribute(
              "aria-label",
              `${index ? "C++" : "Python"} ${label}：${ms.toFixed(3)} 毫秒`,
            );
            b.setAttribute("aria-pressed", String(key === selected));
            b.title = `${label} · ${ms.toFixed(3)} ms`;
            b.addEventListener("click", () => {
              selected = key;
              show();
            });
            return b;
          }),
        );
      });
      const [, label, description] = stages.find(([key]) => key === selected);
      byId("stage-name").textContent = label;
      byId("stage-change").textContent =
        `${oldReport.latency_ms[selected].mean.toFixed(1)} → ${newReport.latency_ms[selected].mean.toFixed(1)} ms`;
      byId("stage-explanation").textContent = description;
      byId("stage-p95").textContent =
        `${newReport.latency_ms.total_ms.p95.toFixed(1)} ms`;
      byId("stage-chart").setAttribute(
        "aria-label",
        `${sceneNames[scene]}流水线：Python ${oldReport.latency_ms.total_ms.mean} ms，C++ ${newReport.latency_ms.total_ms.mean} ms。色块依次为检测、相机确认、坐标转换与跟踪、预测。`,
      );
      for (const name of ["old", "new"])
        byId(`stage-${name}-report`).href =
          `benchmarks/tracking/${scene}_${name === "old" ? "python_original" : "cpp_threshold"}.json`;
      selectButton("data-latency-scene", scene);
      document.querySelectorAll(".stage-legend button").forEach((b) => {
        const active = b.dataset.stage === selected;
        b.classList.toggle("selected", active);
        b.setAttribute("aria-pressed", String(active));
      });
    }
    document.querySelectorAll("[data-latency-scene]").forEach((b) =>
      b.addEventListener("click", () => {
        scene = b.dataset.latencyScene;
        show();
      }),
    );
    document.querySelectorAll(".stage-legend button").forEach((b) =>
      b.addEventListener("click", () => {
        selected = b.dataset.stage;
        show();
      }),
    );
    show();
    buttons.forEach((b) => {
      b.disabled = false;
    });
  } catch {
    byId("stage-status").textContent =
      "分阶段报告读取失败，当前保留静态城市数值。请打开原报告查看。";
  }
})();

(() => {
  const examples = {
    conflict: [
      [0.6, 0.59],
      [0.59, 0],
    ],
    weight: [
      [0.99, 0.4],
      [0.4, 0],
    ],
  };
  const threshold = byId("association-threshold"),
    example = byId("association-case");
  const svgNode = (tag, attrs, text) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text) e.textContent = text;
    return e;
  };
  function diagram(id, matrix, assignment, cutoff) {
    const svg = byId(id);
    svg.replaceChildren();
    assignment.forEach(([i, j]) => {
      const valid = matrix[i][j] >= cutoff;
      svg.append(
        svgNode("line", {
          x1: 62,
          y1: 45 + i * 80,
          x2: 218,
          y2: 45 + j * 80,
          stroke: valid ? "#90e0b8" : "#768492",
          "stroke-width": valid ? 3 : 2,
          "stroke-dasharray": valid ? "" : "5 5",
        }),
      );
      svg.append(
        svgNode(
          "text",
          {
            x: 140,
            y: (45 + i * 80 + 45 + j * 80) / 2 + (i ? -9 : 17),
            fill: valid ? "#90e0b8" : "#92a3af",
            "text-anchor": "middle",
            "font-size": 12,
          },
          matrix[i][j].toFixed(2),
        ),
      );
    });
    for (let i = 0; i < 2; i++)
      for (let side = 0; side < 2; side++) {
        const x = side ? 235 : 45,
          y = 45 + i * 80;
        svg.append(
          svgNode("circle", {
            cx: x,
            cy: y,
            r: 17,
            fill: "#223341",
            stroke: "#809baa",
          }),
        );
        svg.append(
          svgNode(
            "text",
            {
              x,
              y: y + 4,
              fill: "#edf4f8",
              "text-anchor": "middle",
              "font-size": 12,
            },
            side ? ["A", "B"][i] : String(i + 1),
          ),
        );
      }
    svg.append(
      svgNode(
        "text",
        {
          x: 45,
          y: 13,
          fill: "#a6b8c6",
          "text-anchor": "middle",
          "font-size": 10,
        },
        "轨迹",
      ),
    );
    svg.append(
      svgNode(
        "text",
        {
          x: 235,
          y: 13,
          fill: "#a6b8c6",
          "text-anchor": "middle",
          "font-size": 10,
        },
        "检测",
      ),
    );
  }
  function update() {
    const matrix = examples[example.value],
      cutoff = Number(threshold.value);
    const permutations = [
      [
        [0, 0],
        [1, 1],
      ],
      [
        [0, 1],
        [1, 0],
      ],
    ];
    const solve = (filter) =>
      permutations.reduce(
        (best, pairs) => {
          const score = pairs.reduce(
            (sum, [i, j]) =>
              sum + (!filter || matrix[i][j] >= cutoff ? matrix[i][j] : 0),
            0,
          );
          return score > best.score ? { score, pairs } : best;
        },
        { score: -1, pairs: [] },
      ).pairs;
    const oldPairs = solve(false),
      newPairs = solve(true).filter(([i, j]) => matrix[i][j] >= cutoff);
    const kept = oldPairs.filter(([i, j]) => matrix[i][j] >= cutoff);
    byId("association-threshold-label").textContent = cutoff.toFixed(2);
    threshold.setAttribute("aria-valuetext", `IoU ${cutoff.toFixed(2)}`);
    byId("association-matrix").replaceChildren(
      ...matrix.map((values, i) => {
        const tr = document.createElement("tr"),
          th = document.createElement("th");
        th.scope = "row";
        th.textContent = String(i + 1);
        tr.append(th);
        values.forEach((value) => {
          const td = document.createElement("td");
          td.textContent = value.toFixed(2);
          td.classList.toggle("ineligible", value < cutoff);
          tr.append(td);
        });
        return tr;
      }),
    );
    diagram("association-old", matrix, oldPairs, cutoff);
    diagram("association-new", matrix, newPairs, cutoff);
    const summary = (pairs) =>
      `${pairs.length} 个有效匹配 · IoU 总和 ${pairs.reduce((sum, [i, j]) => sum + matrix[i][j], 0).toFixed(2)}`;
    byId("association-old-result").textContent = summary(kept);
    byId("association-new-result").textContent = summary(newPairs);
    let detail;
    if (example.value === "weight")
      detail =
        "两条 0.40 的边总和为 0.80，小于单条 0.99。新策略优化有效 IoU 总和，不优先选择更多匹配。";
    else if (kept.length !== newPairs.length)
      detail =
        "旧策略先选总和 1.18 的两条 0.59 边，但它们低于阈值，最后都被过滤。新策略先排除这些边，保留单条 0.60 匹配。";
    else if (newPairs.length === 0)
      detail = "所有边都低于阈值，两种策略都没有有效匹配。";
    else
      detail =
        "两条 0.59 的边均达到阈值，总和 1.18 高于单条 0.60，两种策略在此示例中结果相同。";
    byId("association-explanation").textContent =
      `当前阈值 ${cutoff.toFixed(2)}。${detail}`;
  }
  function reset() {
    threshold.value = example.value === "weight" ? "0.3" : "0.6";
    update();
  }
  threshold.addEventListener("input", update);
  example.addEventListener("change", reset);
  byId("association-reset").addEventListener("click", reset);
  update();
})();

// Keep audio/video controls native, and avoid simultaneous gallery playback.
for (const video of document.querySelectorAll("video"))
  video.addEventListener("play", () => {
    document.querySelectorAll("video").forEach((other) => {
      if (other !== video) other.pause();
    });
  });
