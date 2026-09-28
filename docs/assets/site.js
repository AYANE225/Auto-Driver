"use strict";

const byId = (id) => document.getElementById(id);
const selectButton = (attribute, value) => {
  document.querySelectorAll(`[${attribute}]`).forEach((button) => {
    const selected = button.getAttribute(attribute) === value;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
};

document.querySelectorAll("[data-scene]").forEach((button) => {
  button.addEventListener("click", () => {
    const scene = button.dataset.scene;
    const replay = byId("replay");
    replay.pause();
    replay.poster = `assets/${scene}_poster.jpg`;
    replay.querySelector("source").src = `assets/${scene}.mp4`;
    replay.querySelector("a").href = `assets/${scene}.mp4`;
    replay.querySelector("a").textContent = `Download the ${scene} replay.`;
    replay.setAttribute(
      "aria-label",
      `${scene} CARLA replay: camera and bird's-eye tracking views`,
    );
    replay.load();
    byId("scene-caption").textContent =
      scene === "urban" ? "Urban / Town10HD_Opt" : "Highway / Town04_Opt";
    byId("gif-link").href =
      `https://github.com/AYANE225/Auto-Driver/blob/main/docs/screenshots/demo_carla_${scene}.gif`;
    selectButton("data-scene", scene);
  });
});

const readReport = async (path) => {
  const response = await fetch(path);
  if (!response.ok)
    throw new Error(`Cannot load ${path}: HTTP ${response.status}`);
  return response.json();
};

const populateTracking = (scene, reports) => {
  const [raw, voxel] = reports[scene];
  const a = raw.latency_ms.total_ms;
  const b = voxel.latency_ms.total_ms;
  byId("raw-latency").textContent = `${a.mean.toFixed(1)} ms`;
  byId("voxel-latency").textContent = `${b.mean.toFixed(1)} ms`;
  // Shared 0–1000 ms scale for both scenes.
  byId("raw-bar").style.width = `${Math.min(100, a.mean / 10)}%`;
  byId("voxel-bar").style.width = `${Math.min(100, b.mean / 10)}%`;
  byId("latency-scene").textContent =
    `${scene === "urban" ? "Urban" : "Highway"} · mean milliseconds per frame · lower is better`;
  byId("latency-detail").textContent =
    `${(a.mean / b.mean).toFixed(2)}× speedup · p95: ${a.p95.toFixed(1)} → ${b.p95.toFixed(1)} ms`;
  const metrics = [
    ["Precision", (r) => r.precision],
    ["Recall", (r) => r.recall],
    ["HOTA", (r) => r.hota.hota],
    ["IDF1", (r) => r.identity.idf1],
    ["MOTA", (r) => r.mota],
  ];
  byId("tracking-metrics").replaceChildren(
    ...metrics.map(([label, value]) => {
      const row = document.createElement("tr");
      const header = document.createElement("th");
      header.scope = "row";
      header.textContent = label;
      row.append(header);
      [raw, voxel].forEach((report) => {
        const cell = document.createElement("td");
        cell.textContent = value(report).toFixed(4);
        row.append(cell);
      });
      return row;
    }),
  );
  byId("raw-report").href = `benchmarks/${scene}_voxel_0.json`;
  byId("voxel-report").href = `benchmarks/${scene}_voxel_0.2.json`;
  selectButton("data-benchmark", scene);
};

const populateForecasting = (population, report) => {
  const results = report.results.test_pooled;
  const models = [
    ["cv", "Constant velocity"],
    ["ca", "Constant acceleration"],
    ["ridge", "Ridge residual"],
  ];
  const columns = ["ade_m", "fde_m", "miss_rate_2m"];
  const best = columns.map((key) =>
    Math.min(...models.map(([name]) => results[name][population][key])),
  );
  byId("forecast-count").textContent =
    `${results.cv[population].windows.toLocaleString("en-US")} windows · lower is better for every metric`;
  byId("forecast-metrics").replaceChildren(
    ...models.map(([name, label]) => {
      const row = document.createElement("tr");
      const header = document.createElement("th");
      header.scope = "row";
      header.textContent = label;
      row.append(header);
      columns.forEach((key, index) => {
        const value = results[name][population][key];
        const cell = document.createElement("td");
        cell.textContent =
          key === "miss_rate_2m"
            ? `${(value * 100).toFixed(2)}%`
            : value.toFixed(3);
        if (value === best[index]) {
          cell.className = "best";
          cell.setAttribute(
            "aria-label",
            `${cell.textContent}, lowest among these models`,
          );
        }
        row.append(cell);
      });
      return row;
    }),
  );
  selectButton("data-population", population);
};

async function loadResults() {
  const buttons = document.querySelectorAll(
    "[data-benchmark], [data-population]",
  );
  buttons.forEach((button) => {
    button.disabled = true;
  });
  try {
    const [urbanRaw, urbanVoxel, highwayRaw, highwayVoxel, forecasting] =
      await Promise.all([
        readReport("benchmarks/urban_voxel_0.json"),
        readReport("benchmarks/urban_voxel_0.2.json"),
        readReport("benchmarks/highway_voxel_0.json"),
        readReport("benchmarks/highway_voxel_0.2.json"),
        readReport("benchmarks/forecasting/metrics.json"),
      ]);
    const reports = {
      urban: [urbanRaw, urbanVoxel],
      highway: [highwayRaw, highwayVoxel],
    };
    populateTracking("urban", reports);
    populateForecasting("all", forecasting);
    document.querySelectorAll("[data-benchmark]").forEach((button) => {
      button.addEventListener("click", () =>
        populateTracking(button.dataset.benchmark, reports),
      );
    });
    document.querySelectorAll("[data-population]").forEach((button) => {
      button.addEventListener("click", () =>
        populateForecasting(button.dataset.population, forecasting),
      );
    });
    buttons.forEach((button) => {
      button.disabled = false;
    });
  } catch (error) {
    byId("data-status").textContent =
      "Interactive reports could not load. The static urban summary is shown; open the linked JSON reports for complete results.";
    console.error(error);
  }
}

byId("copy-command").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(byId("quickstart-command").textContent);
    byId("copy-status").textContent = "Command copied.";
  } catch {
    byId("copy-status").textContent = "Select and copy the command above.";
  }
});
loadResults();
