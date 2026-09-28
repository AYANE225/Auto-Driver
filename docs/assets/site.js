"use strict";
const byId = (id) => document.getElementById(id);
const sceneNames = { urban: "城市", highway: "高速" };
const modelNames = {
  cv: "恒速度 CV",
  ca: "恒加速度 CA",
  ridge: "Ridge 残差回归",
};
const selectButton = (attribute, value) => {
  document.querySelectorAll(`[${attribute}]`).forEach((button) => {
    const selected = button.getAttribute(attribute) === value;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
};
const readReport = async (path) => {
  const response = await fetch(path);
  if (!response.ok)
    throw new Error(`读取 ${path} 失败：HTTP ${response.status}`);
  return response.json();
};
const downloadBlob = (content, filename, type) => {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
byId("nav-toggle").addEventListener("click", () => {
  const open = byId("main-nav").classList.toggle("open");
  byId("nav-toggle").setAttribute("aria-expanded", String(open));
});
document.querySelectorAll("#main-nav a").forEach((a) =>
  a.addEventListener("click", () => {
    byId("main-nav").classList.remove("open");
    byId("nav-toggle").setAttribute("aria-expanded", "false");
  }),
);
document.querySelectorAll("[data-scene]").forEach((button) => {
  button.addEventListener("click", () => {
    const scene = button.dataset.scene;
    const replay = byId("replay");
    replay.pause();
    replay.poster = `assets/${scene}_poster.jpg`;
    replay.querySelector("source").src = `assets/${scene}.mp4`;
    replay.querySelector("a").href = `assets/${scene}.mp4`;
    replay.querySelector("a").textContent = `下载${sceneNames[scene]}回放`;
    replay.setAttribute(
      "aria-label",
      `${sceneNames[scene]} CARLA 相机与鸟瞰跟踪回放`,
    );
    replay.load();
    byId("scene-caption").textContent =
      scene === "urban" ? "城市 / Town10HD_Opt" : "高速 / Town04_Opt";
    byId("gif-link").href =
      `https://github.com/AYANE225/Auto-Driver/blob/main/docs/screenshots/demo_carla_${scene}.gif`;
    selectButton("data-scene", scene);
  });
});
const trackingFields = [
  ["Precision", (r) => r.precision],
  ["Recall", (r) => r.recall],
  ["HOTA", (r) => r.hota.hota],
  ["IDF1", (r) => r.identity.idf1],
  ["MOTA", (r) => r.mota],
];
const populateTracking = (scene, reports) => {
  const [raw, voxel] = reports[scene],
    a = raw.latency_ms.total_ms,
    b = voxel.latency_ms.total_ms;
  byId("raw-latency").textContent = `${a.mean.toFixed(1)} ms`;
  byId("voxel-latency").textContent = `${b.mean.toFixed(1)} ms`;
  byId("raw-bar").style.width = `${Math.min(100, a.mean / 10)}%`;
  byId("voxel-bar").style.width = `${Math.min(100, b.mean / 10)}%`;
  byId("latency-scene").textContent =
    `${sceneNames[scene]} · 每帧平均耗时（ms）· 越低越好`;
  byId("latency-detail").textContent =
    `提速 ${(a.mean / b.mean).toFixed(2)}× · p95：${a.p95.toFixed(1)} → ${b.p95.toFixed(1)} ms`;
  byId("tracking-metrics").replaceChildren(
    ...trackingFields.map(([label, value]) => {
      const row = document.createElement("tr"),
        th = document.createElement("th");
      th.scope = "row";
      th.textContent = label;
      row.append(th);
      [raw, voxel].forEach((report) => {
        const td = document.createElement("td");
        td.textContent = value(report).toFixed(4);
        row.append(td);
      });
      return row;
    }),
  );
  byId("raw-report").href = `benchmarks/${scene}_voxel_0.json`;
  byId("voxel-report").href = `benchmarks/${scene}_voxel_0.2.json`;
  selectButton("data-benchmark", scene);
};
async function loadTracking() {
  const buttons = document.querySelectorAll(
    "[data-benchmark], #download-metrics",
  );
  buttons.forEach((b) => {
    b.disabled = true;
  });
  try {
    const [urbanRaw, urbanVoxel, highwayRaw, highwayVoxel] = await Promise.all([
      readReport("benchmarks/urban_voxel_0.json"),
      readReport("benchmarks/urban_voxel_0.2.json"),
      readReport("benchmarks/highway_voxel_0.json"),
      readReport("benchmarks/highway_voxel_0.2.json"),
    ]);
    const reports = {
      urban: [urbanRaw, urbanVoxel],
      highway: [highwayRaw, highwayVoxel],
    };
    let current = "urban";
    populateTracking(current, reports);
    document.querySelectorAll("[data-benchmark]").forEach((b) =>
      b.addEventListener("click", () => {
        current = b.dataset.benchmark;
        populateTracking(current, reports);
      }),
    );
    byId("download-metrics").addEventListener("click", () => {
      const [a, b] = reports[current];
      const rows = [
        ["scene", "metric", "raw_points", "voxel_0.2m"],
        [
          current,
          "pipeline_mean_ms",
          a.latency_ms.total_ms.mean,
          b.latency_ms.total_ms.mean,
        ],
        [
          current,
          "pipeline_p95_ms",
          a.latency_ms.total_ms.p95,
          b.latency_ms.total_ms.p95,
        ],
        ...trackingFields.map(([name, value]) => [
          current,
          name,
          value(a),
          value(b),
        ]),
      ];
      downloadBlob(
        rows.map((r) => r.join(",")).join("\n") + "\n",
        `${current}_comparison.csv`,
        "text/csv;charset=utf-8",
      );
    });
    buttons.forEach((b) => {
      b.disabled = false;
    });
  } catch {
    byId("data-status").textContent =
      "交互评估数据读取失败，当前显示静态城市结果。可打开报告链接查看原始 JSON。";
  }
}
const populateForecasting = (population, town, report) => {
  const results = report.results[town],
    models = Object.keys(modelNames),
    columns = ["ade_m", "fde_m", "miss_rate_2m"];
  const best = columns.map((k) =>
    Math.min(...models.map((n) => results[n][population][k])),
  );
  byId("forecast-count").textContent =
    `${results.cv[population].windows.toLocaleString("zh-CN")} 个窗口 · 三项指标均越低越好`;
  byId("forecast-metrics").replaceChildren(
    ...models.map((name) => {
      const row = document.createElement("tr"),
        th = document.createElement("th");
      th.scope = "row";
      th.textContent = modelNames[name];
      row.append(th);
      columns.forEach((k, i) => {
        const value = results[name][population][k],
          cell = document.createElement("td");
        cell.textContent =
          k === "miss_rate_2m"
            ? `${(value * 100).toFixed(2)}%`
            : value.toFixed(3);
        if (value === best[i]) {
          cell.className = "best";
          cell.setAttribute(
            "aria-label",
            `${cell.textContent}，所列模型中的最小值`,
          );
        }
        row.append(cell);
      });
      return row;
    }),
  );
  selectButton("data-population", population);
};
async function loadForecasting() {
  const controls = document.querySelectorAll(
    "[data-population],#forecast-town",
  );
  controls.forEach((b) => {
    b.disabled = true;
  });
  try {
    const report = await readReport("benchmarks/forecasting/metrics.json");
    let population = "all";
    const update = () =>
      populateForecasting(population, byId("forecast-town").value, report);
    document.querySelectorAll("[data-population]").forEach((b) =>
      b.addEventListener("click", () => {
        population = b.dataset.population;
        update();
      }),
    );
    byId("forecast-town").addEventListener("change", update);
    update();
    controls.forEach((b) => {
      b.disabled = false;
    });
  } catch {
    byId("forecast-count").textContent =
      "预测报告读取失败，当前显示静态汇总结果。";
  }
}
function updateCommand() {
  const scene = byId("command-scene").value,
    synthetic = scene === "synthetic";
  const detector = byId("command-detector");
  detector.querySelector('[value="fusion"]').disabled = synthetic;
  if (synthetic && detector.value === "fusion") detector.value = "lidar";
  const fusion = detector.value === "fusion",
    voxel = byId("command-voxel"),
    device = byId("command-device"),
    exp = byId("command-export");
  voxel.disabled = synthetic || detector.value === "gt";
  device.disabled = !fusion;
  exp.disabled = synthetic;
  if (synthetic) exp.checked = false;
  const extras = fusion ? "[yolo]" : "";
  let command = `pip install './src/perception_core${extras}'`;
  if (exp.checked) command += "\npip install pillow";
  command += "\n";
  if (synthetic)
    command += `python tools/run_demo.py --frames 60 --detector ${detector.value}`;
  else
    command += `python carla/replay_demo.py --dataset carla/data/${scene} --detector ${detector.value}`;
  command += ` --tracker ${byId("command-tracker").value} \\\n  --no-video --report outputs/${scene}.json`;
  if (!synthetic) {
    command += " --fov-eval";
    if (detector.value !== "gt") command += ` --voxel-size ${voxel.value}`;
  }
  if (fusion) command += ` --device ${device.value}`;
  if (exp.checked)
    command += ` \\\n  --export-replay outputs/${scene}_viewer --export-every 20`;
  byId("quickstart-command").textContent = command;
  byId("copy-status").textContent = "";
  byId("command-note").textContent = synthetic
    ? "先克隆仓库并进入 Auto-Driver 目录。合成场景不需要外部数据、CARLA、ROS 或 GPU；GT 模式用于隔离跟踪误差。"
    : "先克隆仓库并准备所选场景的 CARLA 记录和相机标定。YOLO 模式需要模型权重；CUDA 需要匹配的 PyTorch 环境。导出逐帧数据还需要相机图像，并应选择未使用的输出目录。";
}
byId("command-options").addEventListener("change", updateCommand);
byId("command-options").addEventListener("submit", (e) => e.preventDefault());
byId("copy-command").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(byId("quickstart-command").textContent);
    byId("copy-status").textContent = "命令已复制。";
  } catch {
    byId("copy-status").textContent = "请选中上方命令后手动复制。";
  }
});
updateCommand();
loadTracking();
loadForecasting();
