"use strict";
(() => {
  const colors = {
    history: "#91a6b8",
    future: "#edf4f8",
    cv: "#90e0b8",
    ca: "#79b5f2",
    ridge: "#e2bb80",
  };
  const classes = {
    car: "汽车",
    truck: "卡车",
    bus: "公交车",
    van: "厢式车",
    bicycle: "自行车",
    motorcycle: "摩托车",
    pedestrian: "行人",
    unknown: "未分类",
  };
  function canvasContext(canvas) {
    const width = canvas.clientWidth,
      height = width * Number(canvas.dataset.ratio || 0.8),
      dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.style.height = `${height}px`;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0c131b";
    ctx.fillRect(0, 0, width, height);
    ctx.font = "11px sans-serif";
    return { ctx, width, height };
  }
  function path(ctx, points, project, color, width = 1.5, dash = []) {
    if (!points.length) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    points.forEach((p, i) => {
      const [x, y] = project(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  function snapshot(canvas, name) {
    const a = document.createElement("a");
    a.download = name;
    a.href = canvas.toDataURL("image/png");
    a.click();
  }
  function boxCorners(b) {
    const [x, y, , l, w, , angle] = b,
      c = Math.cos(angle),
      s = Math.sin(angle);
    const corners = [
      [l / 2, w / 2],
      [-l / 2, w / 2],
      [-l / 2, -w / 2],
      [l / 2, -w / 2],
    ];
    return corners.map(([dx, dy]) => [
      x + c * dx - s * dy,
      y + s * dx + c * dy,
    ]);
  }
  const trackColor = (id) => `hsl(${(id * 137.508 + 30) % 360} 75% 70%)`;
  const canvas = byId("bev-canvas"),
    slider = byId("frame-slider"),
    scene = byId("viewer-scene");
  let manifest = null,
    current = null,
    displayedIndex = 0,
    selected = "",
    request = 0,
    playing = false,
    playSession = 0,
    timer = null,
    pan = [0, 0],
    scale = 1,
    project = () => [0, 0];
  const cache = new Map();
  const layers = () =>
    Object.fromEntries(
      [...document.querySelectorAll("[data-layer]")].map((el) => [
        el.dataset.layer,
        el.checked,
      ]),
    );
  function drawBEV() {
    const { ctx, width, height } = canvasContext(canvas),
      enabled = layers();
    scale =
      Math.min((width - 55) / 100, (height - 65) / 110) *
      Number(byId("bev-zoom").value);
    project = ([x, y]) => [
      width / 2 - (y - pan[1]) * scale,
      height / 2 - (x - pan[0]) * scale,
    ];
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, width, height);
    ctx.clip();
    for (let v = -150; v <= 150; v += 10) {
      path(
        ctx,
        [
          [v, -150],
          [v, 150],
        ],
        project,
        "#21313e",
        1,
      );
      path(
        ctx,
        [
          [-150, v],
          [150, v],
        ],
        project,
        "#21313e",
        1,
      );
      const [, py] = project([v, 0]);
      if (py > 40 && py < height - 15) {
        ctx.fillStyle = "#718899";
        ctx.fillText(`${v}`, 6, py - 3);
      }
    }
    ctx.fillStyle = "#9cafbc";
    ctx.fillText("前方 +x ↑  /  左侧 +y ←  /  单位 m", 12, height - 12);
    const [ex, ey] = project([0, 0]);
    ctx.fillStyle = "#edf4f8";
    ctx.beginPath();
    ctx.moveTo(ex, ey - 8);
    ctx.lineTo(ex - 5, ey + 5);
    ctx.lineTo(ex + 5, ey + 5);
    ctx.closePath();
    ctx.fill();
    if (!current) {
      ctx.fillStyle = "#a6b8c6";
      ctx.fillText("正在读取实录数据…", 20, 40);
      ctx.restore();
      return;
    }
    if (enabled.points) {
      current.points.forEach((p) => {
        const [x, y] = project(p);
        ctx.fillStyle = p[2] < -1.2 ? "#425567" : "#7894a7";
        ctx.fillRect(x, y, 1.5, 1.5);
      });
    }
    if (enabled.ground_truth)
      current.ground_truth.forEach((t) => {
        const c = boxCorners(t.box);
        path(ctx, [...c, c[0]], project, "#e2bb80", 1, [4, 4]);
      });
    current.tracks.forEach((t) => {
      const color = trackColor(t.id),
        picked = String(t.id) === selected;
      if (enabled.history) path(ctx, t.history, project, color, 1, [2, 3]);
      if (enabled.tracks) {
        const c = boxCorners(t.box);
        path(ctx, [...c, c[0]], project, color, picked ? 3 : 1.5);
        const [x, y] = project(t.box);
        ctx.fillStyle = color;
        ctx.fillText(`#${t.id}`, x + 5, y - 7);
      }
    });
    if (enabled.predictions)
      current.predictions.forEach((p) => {
        const t = current.tracks.find((t) => t.id === p.id);
        if (!t) return;
        p.trajectories.forEach((tr) =>
          path(ctx, [t.box, ...tr.xy], project, trackColor(p.id), 1.5, [5, 4]),
        );
      });
    ctx.restore();
  }
  function describeTrack() {
    const element = byId("track-details");
    element.replaceChildren();
    const t = current?.tracks.find((t) => String(t.id) === selected);
    const rows = t
      ? [
          ["类别", classes[t.label] || t.label],
          ["速度", `${t.speed_mps.toFixed(2)} m/s`],
          ["中心 x / y", `${t.box[0].toFixed(2)} / ${t.box[1].toFixed(2)} m`],
          [
            "长 / 宽 / 高",
            `${t.box
              .slice(3, 6)
              .map((x) => x.toFixed(2))
              .join(" / ")} m`,
          ],
          ["连续未更新", `${t.missed_frames} 帧`],
        ]
      : [
          [
            "操作提示",
            "选择跟踪 ID 查看状态；真值与跟踪 ID 属于不同编号体系。",
          ],
        ];
    rows.forEach(([key, value]) => {
      const dt = document.createElement("dt"),
        dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      element.append(dt, dd);
    });
  }
  function controlsDisabled(disabled) {
    document
      .querySelectorAll(
        "#previous-frame,#next-frame,#frame-slider,#play-frames",
      )
      .forEach((el) => {
        el.disabled = disabled;
      });
  }
  async function loadFrame(index) {
    if (!manifest) return false;
    const token = ++request,
      active = manifest,
      base = `assets/replay/${scene.value}/`,
      row = active.frames[index];
    byId("viewer-summary").textContent = `正在读取第 ${row.frame_id} 帧…`;
    try {
      const url = base + row.file;
      let payload = cache.get(url);
      if (!payload) {
        payload = await readReport(url);
        cache.set(url, payload);
      }
      let image = null;
      if (payload.image) {
        image = new Image();
        image.src = base + payload.image;
        await image.decode();
      }
      if (token !== request) return false;
      current = payload;
      displayedIndex = index;
      slider.value = String(index);
      if (image) {
        byId("frame-camera").src = image.src;
        byId("frame-camera").hidden = false;
      } else byId("frame-camera").hidden = true;
      byId("frame-label").textContent =
        `帧 ${payload.frame_id} · ${payload.elapsed_s.toFixed(1)} s · ${index + 1}/${active.frames.length}`;
      slider.setAttribute(
        "aria-valuetext",
        `第 ${payload.frame_id} 帧，${payload.elapsed_s.toFixed(1)} 秒`,
      );
      byId("viewer-summary").textContent =
        `原始 ${payload.point_count.toLocaleString("zh-CN")} 点，显示 ${payload.points.length.toLocaleString("zh-CN")} 点；${payload.ground_truth.length} 个真值目标，${payload.tracks.length} 条确认轨迹。`;
      const select = byId("track-select");
      select.replaceChildren(
        new Option(
          payload.tracks.length ? "选择跟踪 ID" : "本帧没有确认轨迹",
          "",
        ),
      );
      payload.tracks.forEach((t) =>
        select.add(
          new Option(
            `#${t.id} · ${classes[t.label] || t.label} · ${t.speed_mps.toFixed(1)} m/s`,
            String(t.id),
          ),
        ),
      );
      if (!payload.tracks.some((t) => String(t.id) === selected)) selected = "";
      select.value = selected;
      byId("save-bev").disabled = false;
      byId("frame-json").hidden = false;
      byId("frame-json").href = url;
      byId("frame-json").download = `${scene.value}_${row.frame_id}.json`;
      describeTrack();
      drawBEV();
      return true;
    } catch {
      if (token === request) {
        slider.value = String(displayedIndex);
        byId("viewer-summary").textContent =
          "当前帧读取失败，请重新选择时间或场景。已有画面保持不变。";
        stop();
      }
      return false;
    }
  }
  function stop() {
    playSession++;
    // A paused or superseded playback request must not change the visible frame.
    if (playing) request++;
    playing = false;
    clearTimeout(timer);
    byId("play-frames").textContent = "连续查看";
  }
  function scheduleNext() {
    if (!playing || !manifest) return;
    const current = manifest.frames[displayedIndex],
      next = manifest.frames[displayedIndex + 1];
    if (!next) {
      stop();
      return;
    }
    timer = setTimeout(
      playNext,
      Math.max(0, next.elapsed_s - current.elapsed_s) * 1000,
    );
  }
  async function playNext() {
    if (!playing) return;
    if (displayedIndex >= manifest.frames.length - 1) {
      stop();
      return;
    }
    const session = playSession;
    const ok = await loadFrame(displayedIndex + 1);
    if (session !== playSession) return;
    if (ok && playing) scheduleNext();
    else stop();
  }
  async function loadScene() {
    stop();
    const token = ++request;
    controlsDisabled(true);
    current = null;
    manifest = null;
    selected = "";
    pan = [0, 0];
    byId("bev-zoom").value = "1";
    byId("frame-camera").hidden = true;
    byId("frame-json").hidden = true;
    byId("save-bev").disabled = true;
    byId("frame-label").textContent = "—";
    byId("viewer-summary").textContent = "正在读取场景数据…";
    displayedIndex = 0;
    slider.value = "0";
    slider.removeAttribute("aria-valuetext");
    byId("track-select").replaceChildren(new Option("正在读取…", ""));
    describeTrack();
    drawBEV();
    try {
      const data = await readReport(`assets/replay/${scene.value}/index.json`);
      if (token !== request) return;
      manifest = data;
      slider.max = String(data.frames.length - 1);
      await loadFrame(Math.min(6, data.frames.length - 1));
      if (manifest === data) controlsDisabled(false);
    } catch {
      if (token === request) {
        byId("viewer-summary").textContent =
          "场景数据读取失败，请切换场景重试。";
      }
    }
  }
  scene.addEventListener("change", loadScene);
  slider.addEventListener("input", () => {
    stop();
    loadFrame(Number(slider.value));
  });
  byId("previous-frame").addEventListener("click", () => {
    stop();
    loadFrame(Math.max(0, displayedIndex - 1));
  });
  byId("next-frame").addEventListener("click", () => {
    stop();
    loadFrame(Math.min(manifest.frames.length - 1, displayedIndex + 1));
  });
  byId("play-frames").addEventListener("click", () => {
    if (playing) {
      stop();
      return;
    }
    playing = true;
    const session = ++playSession;
    byId("play-frames").textContent = "暂停查看";
    if (displayedIndex === manifest.frames.length - 1) {
      loadFrame(0).then((ok) => {
        if (session !== playSession) return;
        if (ok && playing) scheduleNext();
      });
    } else scheduleNext();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });
  byId("track-select").addEventListener("change", () => {
    selected = byId("track-select").value;
    describeTrack();
    drawBEV();
  });
  document
    .querySelectorAll("[data-layer]")
    .forEach((el) => el.addEventListener("change", drawBEV));
  byId("bev-zoom").addEventListener("input", drawBEV);
  byId("reset-view").addEventListener("click", () => {
    pan = [0, 0];
    byId("bev-zoom").value = "1";
    drawBEV();
  });
  byId("save-bev").addEventListener("click", () =>
    snapshot(canvas, `${scene.value}_frame_${current?.frame_id ?? 0}.png`),
  );
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, pan: [...pan], moved: false };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x,
      dy = e.clientY - drag.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (drag.moved) {
      pan = [drag.pan[0] + dy / scale, drag.pan[1] + dx / scale];
      drawBEV();
    }
  });
  canvas.addEventListener("pointerup", (e) => {
    if (drag && !drag.moved && current && layers().tracks) {
      const rect = canvas.getBoundingClientRect(),
        point = [e.clientX - rect.left, e.clientY - rect.top];
      let found = null,
        distance = 24;
      current.tracks.forEach((t) => {
        const p = project(t.box),
          d = Math.hypot(p[0] - point[0], p[1] - point[1]);
        if (d < distance) {
          found = t;
          distance = d;
        }
      });
      selected = found ? String(found.id) : "";
      byId("track-select").value = selected;
      describeTrack();
      drawBEV();
    }
    drag = null;
  });
  canvas.addEventListener("pointercancel", () => {
    drag = null;
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });
  new ResizeObserver(drawBEV).observe(canvas.parentElement);
  // Defer dataset requests until this section approaches the viewport.
  const viewerObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        viewerObserver.disconnect();
        loadScene();
      }
    },
    { rootMargin: "300px" },
  );
  controlsDisabled(true);
  viewerObserver.observe(byId("explorer"));

  const forecastCanvas = byId("forecast-canvas");
  forecastCanvas.dataset.ratio = "0.65";
  let examples = [],
    example = null;
  function drawForecast() {
    const { ctx, width, height } = canvasContext(forecastCanvas);
    if (!example) return;
    const lines = {
      history: example.history,
      future: [[0, 0], ...example.future],
      ...Object.fromEntries(
        Object.entries(example.forecasts).map(([n, xy]) => [
          n,
          [[0, 0], ...xy],
        ]),
      ),
    };
    const points = Object.values(lines).flat(),
      xs = points.map((p) => p[0]),
      ys = points.map((p) => p[1]);
    const xmin = Math.min(...xs),
      xmax = Math.max(...xs),
      ymin = Math.min(...ys),
      ymax = Math.max(...ys),
      cx = (xmin + xmax) / 2,
      cy = (ymin + ymax) / 2;
    const sc = Math.min(
      (width - 90) / Math.max(2, (xmax - xmin) * 1.15),
      (height - 80) / Math.max(2, (ymax - ymin) * 1.15),
    );
    const project = ([x, y]) => [
      width / 2 + (x - cx) * sc,
      height / 2 - (y - cy) * sc,
    ];
    const span = Math.max(width, height) / sc,
      power = 10 ** Math.floor(Math.log10(span / 6)),
      step = [1, 2, 5, 10].map((n) => n * power).find((n) => n >= span / 6);
    ctx.fillStyle = "#8198a9";
    for (
      let x = Math.floor((cx - width / 2 / sc) / step) * step;
      x <= cx + width / 2 / sc;
      x += step
    ) {
      path(
        ctx,
        [
          [x, cy - height / sc],
          [x, cy + height / sc],
        ],
        project,
        "#253744",
        1,
      );
      const [px] = project([x, 0]);
      if (px > 20 && px < width - 20)
        ctx.fillText(x.toFixed(step < 1 ? 1 : 0), px + 3, height - 23);
    }
    for (
      let y = Math.floor((cy - height / 2 / sc) / step) * step;
      y <= cy + height / 2 / sc;
      y += step
    ) {
      path(
        ctx,
        [
          [cx - width / sc, y],
          [cx + width / sc, y],
        ],
        project,
        "#253744",
        1,
      );
      const [, py] = project([0, y]);
      if (py > 25 && py < height - 35)
        ctx.fillText(y.toFixed(step < 1 ? 1 : 0), 8, py - 4);
    }
    ctx.fillStyle = "#a6b8c6";
    ctx.fillText("Y (m)", 10, 18);
    ctx.fillText("X (m)", width - 45, height - 8);
    document.querySelectorAll("[data-forecast-line]").forEach((el) => {
      if (!el.checked) return;
      const name = el.dataset.forecastLine,
        pts = lines[name];
      path(
        ctx,
        pts,
        project,
        colors[name],
        name === "future" ? 3 : 2,
        name === "future" || name === "history" ? [] : [6, 4],
      );
      pts.forEach((p) => {
        const [x, y] = project(p);
        ctx.fillStyle = colors[name];
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    });
    const [ox, oy] = project([0, 0]);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.strokeRect(ox - 5, oy - 5, 10, 10);
  }
  function updateExample() {
    example = examples[Number(byId("forecast-example").value)];
    if (!example) return;
    byId("example-description").textContent =
      `${example.town} · ${["车辆", "行人", "两轮车"][example.class_id] || "其他"} · 窗口索引 ${example.window_index}（从 0 计数）`;
    byId("example-errors").replaceChildren(
      ...Object.entries(example.forecasts).map(([name, pts]) => {
        const errors = pts.map((p, i) =>
          Math.hypot(p[0] - example.future[i][0], p[1] - example.future[i][1]),
        );
        const row = document.createElement("tr"),
          th = document.createElement("th");
        th.scope = "row";
        th.textContent = modelNames[name];
        row.append(th);
        [
          errors.reduce((a, b) => a + b, 0) / errors.length,
          errors.at(-1),
        ].forEach((value) => {
          const td = document.createElement("td");
          td.textContent = value.toFixed(3);
          row.append(td);
        });
        return row;
      }),
    );
    drawForecast();
  }
  byId("forecast-example").disabled = true;
  readReport("assets/forecast_examples.json")
    .then((data) => {
      examples = data.examples;
      byId("forecast-example").replaceChildren(
        ...examples.map(
          (e, i) =>
            new Option(
              `${e.town.replace("_Opt", "")} · 示例 ${(i % 12) + 1}`,
              String(i),
            ),
        ),
      );
      byId("forecast-example").disabled = false;
      byId("save-forecast").disabled = false;
      updateExample();
    })
    .catch(() => {
      byId("example-description").textContent =
        "预测示例读取失败，可尝试刷新页面。";
    });
  byId("forecast-example").addEventListener("change", updateExample);
  document
    .querySelectorAll("[data-forecast-line]")
    .forEach((el) => el.addEventListener("change", drawForecast));
  byId("save-forecast").addEventListener("click", () =>
    snapshot(
      forecastCanvas,
      `forecast_${example?.town || "example"}_${example?.window_index || 0}.png`,
    ),
  );
  new ResizeObserver(drawForecast).observe(forecastCanvas.parentElement);
})();
