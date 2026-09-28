"use strict";
(async function loadNativeTracking() {
  const controls = document.querySelectorAll("[data-native-scene]");
  controls.forEach((button) => { button.disabled = true; });
  const variants = [
    ["python_original", "Python 原逻辑"],
    ["cpp_original", "C++ 原逻辑"],
    ["cpp_axis", "C++ + π 周期"],
    ["cpp_threshold", "C++ + 阈值处理（默认）"],
    ["cpp_axis_threshold", "C++ + 两项改动"],
  ];
  const row = (label, values) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = label;
    tr.append(th);
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    });
    return tr;
  };
  try {
    const scenes = ["urban", "highway"];
    const data = Object.fromEntries(await Promise.all(scenes.map(async (scene) => {
      const reports = await Promise.all(["ablation", "python_original", "cpp_threshold"].map(
        (name) => readReport(`benchmarks/tracking/${scene}_${name}.json`),
      ));
      return [scene, reports];
    })));
    const show = (scene) => {
      const [ablation, oldPipeline, newPipeline] = data[scene];
      const results = ablation.results;
      const a = results.python_original.mean_ms;
      const b = results.cpp_original.mean_ms;
      byId("native-python-time").textContent = `${a.toFixed(1)} ms`;
      byId("native-cpp-time").textContent = `${b.toFixed(1)} ms`;
      byId("native-cpp-bar").style.width = `${100 * b / a}%`;
      byId("native-latency-scene").textContent = `${sceneNames[scene]} · 仅 tracker.update 平均耗时 · 越低越好`;
      byId("native-speedup").textContent = `提速 ${(a / b).toFixed(1)}× · 仅跟踪计算，不含检测`;
      byId("native-pipeline").replaceChildren(...[oldPipeline, newPipeline].map((report, i) => {
        const t = report.latency_ms.total_ms;
        return row(i ? "C++ + 阈值处理" : "Python 原逻辑", [`${t.mean.toFixed(1)} / ${t.p95.toFixed(1)}`]);
      }));
      byId("native-ablation").replaceChildren(...variants.map(([key, label]) => {
        const r = results[key];
        return row(label, [r.mean_ms.toFixed(2), r.quality.hota.hota.toFixed(4),
          r.quality.identity.idf1.toFixed(4), r.quality.mota.toFixed(4)]);
      }));
      byId("native-ablation-caption").textContent = `${sceneNames[scene]} · 固定检测输入 · ${ablation.frames} 帧`;
      byId("native-ablation-report").href = `benchmarks/tracking/${scene}_ablation.json`;
      byId("native-pipeline-report").href = `benchmarks/tracking/${scene}_cpp_threshold.json`;
      byId("native-status").textContent = `${ablation.repeats} 轮固定输入对照；完整流水线每配置 1 轮。实际后端：${newPipeline.iou_backend}。`;
      selectButton("data-native-scene", scene);
    };
    controls.forEach((button) => {
      button.addEventListener("click", () => show(button.dataset.nativeScene));
      button.disabled = false;
    });
    show("urban");
  } catch {
    byId("native-status").textContent = "C++ 对照数据读取失败，当前显示静态城市结果。请打开报告链接查看。";
  }
})();
