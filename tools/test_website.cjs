// Run against a static server: NODE_PATH=/path/to/node_modules node tools/test_website.cjs [URL]
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const base = process.argv[2] || "http://127.0.0.1:8765/";

async function main() {
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(String(error)));
    page.on("response", (response) => {
      if (response.status() >= 400)
        errors.push(`${response.status()} ${response.url()}`);
    });
    await page.goto(base, { waitUntil: "networkidle" });
    const read = async (relative) => {
      const response = await context.request.get(new URL(relative, base).href);
      assert.equal(response.status(), 200, relative);
      return response.json();
    };
    const text = (selector) => page.locator(selector).textContent();
    const pixels = (selector) =>
      page.locator(selector).evaluate((canvas) => canvas.toDataURL());
    const range = (selector, value) =>
      page.locator(selector).evaluate((el, value) => {
        el.value = String(value);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }, value);
    const frameReady = (scene, id) =>
      page.waitForFunction(
        ({ scene, id }) => {
          const link = document.getElementById("frame-json");
          return (
            !link.hidden &&
            link.href.endsWith(
              `/replay/${scene}/${String(id).padStart(6, "0")}.json`,
            ) &&
            document
              .getElementById("viewer-summary")
              .textContent.includes("确认轨迹")
          );
        },
        { scene, id },
      );
    const download = async (selector, filename) => {
      const pending = page.waitForEvent("download");
      await page.locator(selector).click();
      const item = await pending;
      assert.equal(item.suggestedFilename(), filename);
      const stream = await item.createReadStream();
      const chunks = [];
      for await (const chunk of stream) chunks.push(chunk);
      return Buffer.concat(chunks);
    };
    const screenshot = async (name, selector) => {
      if (!process.env.WEBSITE_SCREENSHOTS) return;
      await fs.mkdir(process.env.WEBSITE_SCREENSHOTS, { recursive: true });
      const target = selector ? page.locator(selector) : page;
      await target.screenshot({
        path: path.join(process.env.WEBSITE_SCREENSHOTS, `${name}.png`),
      });
    };
    assert.equal(await page.locator("html").getAttribute("lang"), "zh-CN");
    await screenshot("desktop");
    await page.locator("#explorer").scrollIntoViewIfNeeded();
    await frameReady("urban", 120);
    for (const scene of ["urban", "highway"]) {
      const manifest = await read(`assets/replay/${scene}/index.json`);
      assert.equal(manifest.processed_frames, 400);
      assert.deepEqual(
        manifest.frames.map((f) => f.frame_id),
        [...Array.from({ length: 20 }, (_, i) => i * 20), 399],
      );
      for (const row of manifest.frames) {
        const sample = await read(`assets/replay/${scene}/${row.file}`);
        assert.equal(sample.frame_id, row.frame_id);
        assert.equal(sample.tracks.length, row.tracks);
        assert.ok(sample.points.length <= 2500 && sample.points.length > 0);
        assert.ok(sample.point_count >= sample.roi_point_count);
        assert.ok(
          sample.points.every(
            (p) => p.length === 3 && p.every(Number.isFinite),
          ),
        );
        const image = await context.request.get(
          new URL(`assets/replay/${scene}/${sample.image}`, base).href,
        );
        assert.equal(image.status(), 200);
        assert.equal(
          (await image.body()).subarray(0, 2).toString("hex"),
          "ffd8",
        );
      }
    }
    const initial = await read("assets/replay/urban/000120.json");
    assert.equal(
      await page.locator("#track-select option").count(),
      initial.tracks.length + 1,
    );
    assert.match(
      await text("#viewer-summary"),
      new RegExp(`${initial.ground_truth.length} 个真值目标`),
    );
    const track = initial.tracks[0];
    await page.locator("#track-select").selectOption(String(track.id));
    assert.ok(
      (await text("#track-details")).includes(
        `${track.speed_mps.toFixed(2)} m/s`,
      ),
    );
    await screenshot("viewer", "#explorer");
    for (const layer of [
      "points",
      "ground_truth",
      "tracks",
      "history",
      "predictions",
    ]) {
      const before = await pixels("#bev-canvas");
      await page.locator(`[data-layer="${layer}"]`).uncheck();
      assert.notEqual(await pixels("#bev-canvas"), before, layer);
      await page.locator(`[data-layer="${layer}"]`).check();
    }
    const original = await pixels("#bev-canvas");
    await range("#bev-zoom", 2);
    assert.notEqual(await pixels("#bev-canvas"), original);
    await page.locator("#reset-view").click();
    assert.equal(await pixels("#bev-canvas"), original);
    const rect = await page.locator("#bev-canvas").boundingBox();
    await page.mouse.move(rect.x + 100, rect.y + 100);
    await page.mouse.down();
    await page.mouse.move(rect.x + 160, rect.y + 140, { steps: 5 });
    await page.mouse.up();
    assert.notEqual(await pixels("#bev-canvas"), original);
    await page.locator("#reset-view").click();
    await page.locator("#track-select").selectOption("");
    const scale = Math.min((rect.width - 55) / 100, (rect.height - 65) / 110);
    await page.locator("#bev-canvas").click({
      position: {
        x: rect.width / 2 - track.box[1] * scale,
        y: rect.height / 2 - track.box[0] * scale,
      },
    });
    assert.equal(
      await page.locator("#track-select").inputValue(),
      String(track.id),
    );
    const png = await download("#save-bev", "urban_frame_120.png");
    assert.equal(png.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
    assert.equal(
      JSON.parse(await download("#frame-json", "urban_120.json")).frame_id,
      120,
    );
    await range("#frame-slider", 0);
    await frameReady("urban", 0);
    assert.equal(await page.locator("#track-select option").count(), 1);
    await page.locator("#next-frame").click();
    await frameReady("urban", 20);
    await page.locator("#previous-frame").click();
    await frameReady("urban", 0);
    await range("#frame-slider", 20);
    await frameReady("urban", 399);
    await page.locator("#play-frames").click();
    await frameReady("urban", 0);
    await frameReady("urban", 20);
    await page.locator("#play-frames").click();
    assert.equal(await text("#play-frames"), "连续查看");

    // Pause while the next frame's image is delayed: releasing it must not jump.
    const paused = await context.newPage();
    let releasePaused, sawPaused;
    const pausedGate = new Promise((resolve) => {
        releasePaused = resolve;
      }),
      pausedRequest = new Promise((resolve) => {
        sawPaused = resolve;
      });
    await paused.route("**/replay/urban/000020.jpg", async (route) => {
      sawPaused();
      await pausedGate;
      await route.continue();
    });
    await paused.goto(base);
    await paused.locator("#explorer").scrollIntoViewIfNeeded();
    await paused.waitForFunction(() =>
      document.getElementById("frame-label").textContent.includes("帧 120"),
    );
    await paused.locator("#frame-slider").evaluate((el) => {
      el.value = "0";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await paused.waitForFunction(() =>
      document.getElementById("frame-label").textContent.includes("帧 0 ·"),
    );
    const playbackStart = Date.now();
    await paused.locator("#play-frames").click();
    await pausedRequest;
    assert.ok(
      Date.now() - playbackStart >= 1800,
      "respect the 2 s recorded interval",
    );
    await paused.locator("#play-frames").click();
    const responsePaused = paused.waitForResponse("**/replay/urban/000020.jpg");
    releasePaused();
    await responsePaused;
    await paused.waitForLoadState("networkidle");
    assert.equal(await paused.locator("#frame-slider").inputValue(), "0");
    assert.equal(
      await paused.locator("#play-frames").textContent(),
      "连续查看",
    );
    await paused.close();

    // Hold the older frame's camera response while a newer selection completes.
    let releaseImage, imageRequested;
    const imageGate = new Promise((resolve) => {
      releaseImage = resolve;
    });
    const seenImage = new Promise((resolve) => {
      imageRequested = resolve;
    });
    await page.route("**/replay/urban/000200.jpg", async (route) => {
      imageRequested();
      await imageGate;
      await route.continue();
    });
    await range("#frame-slider", 10);
    await seenImage;
    await range("#frame-slider", 11);
    await frameReady("urban", 220);
    const oldImageDone = page.waitForResponse("**/replay/urban/000200.jpg");
    releaseImage();
    await oldImageDone;
    await page.waitForLoadState("networkidle");
    assert.ok(
      (await page.locator("#frame-camera").getAttribute("src")).endsWith(
        "urban/000220.jpg",
      ),
    );
    await frameReady("urban", 220);
    await page.unroute("**/replay/urban/000200.jpg");
    await page.locator("#viewer-scene").selectOption("highway");
    await frameReady("highway", 120);
    await range("#frame-slider", 20);
    await frameReady("highway", 399);
    assert.ok(
      (await page.locator("#frame-camera").getAttribute("src")).endsWith(
        "highway/000399.jpg",
      ),
    );

    let releaseScene, sceneRequested;
    const sceneGate = new Promise((resolve) => {
      releaseScene = resolve;
    });
    const seenScene = new Promise((resolve) => {
      sceneRequested = resolve;
    });
    await page.route("**/replay/urban/index.json", async (route) => {
      sceneRequested();
      await sceneGate;
      await route.continue();
    });
    await page.locator("#viewer-scene").selectOption("urban");
    await seenScene;
    await page.locator("#viewer-scene").selectOption("highway");
    await frameReady("highway", 120);
    const oldSceneDone = page.waitForResponse("**/replay/urban/index.json");
    releaseScene();
    await oldSceneDone;
    await page.waitForLoadState("networkidle");
    await frameReady("highway", 120);
    await page.unroute("**/replay/urban/index.json");

    await page.locator('[data-benchmark="highway"]').click();
    assert.equal(await text("#raw-latency"), "289.5 ms");
    assert.equal(await text("#voxel-latency"), "152.3 ms");
    assert.match(await text("#tracking-metrics"), /-0.4503/);
    const csv = (
      await download("#download-metrics", "highway_comparison.csv")
    ).toString();
    assert.match(csv, /^scene,metric,raw_points,voxel_0.2m\n/);
    const baseline = await read("benchmarks/highway_voxel_0.2.json");
    assert.ok(csv.includes(`,${baseline.latency_ms.total_ms.mean}\n`));
    await page.locator("#native-tracking details").evaluate((el) => {
      el.open = true;
    });
    for (const scene of ["urban", "highway"]) {
      await page.locator(`[data-native-scene="${scene}"]`).click();
      const ablation = await read(`benchmarks/tracking/${scene}_ablation.json`);
      const oldPipeline = await read(
        `benchmarks/tracking/${scene}_python_original.json`,
      );
      const newPipeline = await read(
        `benchmarks/tracking/${scene}_cpp_threshold.json`,
      );
      const a = ablation.results.python_original;
      const b = ablation.results.cpp_original;
      assert.equal(b.matches_captured_states, true);
      assert.equal(b.max_state_error, 0);
      assert.equal(
        await text("#native-python-time"),
        `${a.mean_ms.toFixed(1)} ms`,
      );
      assert.equal(
        await text("#native-cpp-time"),
        `${b.mean_ms.toFixed(1)} ms`,
      );
      assert.ok(
        (await text("#native-speedup")).includes(
          `${(a.mean_ms / b.mean_ms).toFixed(1)}×`,
        ),
      );
      for (const report of [oldPipeline, newPipeline]) {
        const t = report.latency_ms.total_ms;
        assert.ok(
          (await text("#native-pipeline")).includes(
            `${t.mean.toFixed(1)} / ${t.p95.toFixed(1)}`,
          ),
        );
      }
      for (const [index, result] of Object.values(ablation.results).entries()) {
        const cells = await page
          .locator("#native-ablation tr")
          .nth(index)
          .locator("td")
          .allTextContents();
        assert.deepEqual(cells, [
          result.mean_ms.toFixed(2),
          result.quality.hota.hota.toFixed(4),
          result.quality.identity.idf1.toFixed(4),
          result.quality.mota.toFixed(4),
        ]);
      }
      assert.equal(newPipeline.iou_backend, "cpp");
      assert.equal(
        await page.locator("#native-ablation-report").getAttribute("href"),
        `benchmarks/tracking/${scene}_ablation.json`,
      );
    }
    await screenshot("native-tracking", "#native-tracking");
    const report = await read("benchmarks/forecasting/metrics.json");
    assert.equal(await page.locator("#forecast-town option").count(), 3);
    for (const town of ["test_pooled", "Town05_Opt", "Town10HD_Opt"]) {
      await page.locator("#forecast-town").selectOption(town);
      for (const population of ["all", "moving"]) {
        await page.locator(`[data-population="${population}"]`).click();
        const expected = report.results[town].cv[population];
        assert.ok(
          (await text("#forecast-count")).includes(
            expected.windows.toLocaleString("zh-CN"),
          ),
        );
        assert.ok(
          (await text("#forecast-metrics")).includes(expected.fde_m.toFixed(3)),
        );
      }
    }
    const examples = await read("assets/forecast_examples.json");
    assert.equal(await page.locator("#forecast-example option").count(), 24);
    for (const i of [0, 11, 12, 23]) {
      await page.locator("#forecast-example").selectOption(String(i));
      const example = examples.examples[i];
      assert.ok((await text("#example-description")).includes(example.town));
      const error = Math.hypot(
        ...example.forecasts.cv
          .at(-1)
          .map((n, j) => n - example.future.at(-1)[j]),
      );
      assert.ok((await text("#example-errors")).includes(error.toFixed(3)));
    }
    const forecastBefore = await pixels("#forecast-canvas");
    await page.locator('[data-forecast-line="future"]').uncheck();
    assert.notEqual(await pixels("#forecast-canvas"), forecastBefore);
    await page.locator('[data-forecast-line="future"]').check();
    const example = examples.examples[23];
    await download(
      "#save-forecast",
      `forecast_${example.town}_${example.window_index}.png`,
    );
    await screenshot("forecast", "#forecast-examples");
    await page.locator('[data-scene="highway"]').click();
    await page.locator("#replay").evaluate((video) => video.play());
    await page.waitForFunction(
      () => document.getElementById("replay").currentTime > 0.25,
    );
    assert.ok(
      Math.abs(
        (await page.locator("#replay").evaluate((v) => v.duration)) - 24,
      ) < 0.1,
    );
    await page.locator("#replay").evaluate((v) => v.pause());

    assert.ok(await page.locator("#command-export").isDisabled());
    await page.locator("#command-scene").selectOption("urban");
    await page.locator("#command-detector").selectOption("fusion");
    await page.locator("#command-device").selectOption("cuda:0");
    await page.locator("#command-export").check();
    assert.match(await text("#quickstart-command"), /--device cuda:0/);
    assert.match(
      await text("#quickstart-command"),
      /--export-replay outputs\/urban_viewer/,
    );
    await page.locator("#command-detector").selectOption("gt");
    assert.ok(await page.locator("#command-voxel").isDisabled());
    assert.doesNotMatch(
      await text("#quickstart-command"),
      /--voxel-size|--device/,
    );
    await page.locator("#command-scene").selectOption("synthetic");
    assert.doesNotMatch(
      await text("#quickstart-command"),
      /--export-replay|--fov-eval/,
    );
    assert.ok(
      await page
        .locator('#command-detector option[value="fusion"]')
        .evaluate((option) => option.disabled),
    );
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.locator("#copy-command").click();
    await page.waitForFunction(
      () =>
        document.getElementById("copy-status").textContent === "命令已复制。",
    );
    assert.equal(
      await page.evaluate(() => navigator.clipboard.readText()),
      await text("#quickstart-command"),
    );
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `overflow at ${width}`,
      );
    }
    await page.setViewportSize({ width: 390, height: 900 });
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
    await screenshot("mobile");
    await page.locator("#nav-toggle").click();
    assert.equal(
      await page.locator("#nav-toggle").getAttribute("aria-expanded"),
      "true",
    );
    await page.locator('#main-nav a[href="#run"]').click();
    assert.equal(
      await page.locator("#nav-toggle").getAttribute("aria-expanded"),
      "false",
    );
    await screenshot("mobile-command", "#run");
    await screenshot("mobile-viewer", "#explorer");
    assert.deepEqual(errors, []);

    // Verify useful fallback content when individual datasets fail or JS is disabled.
    const failed = await context.newPage();
    const fallbackErrors = [];
    failed.on("pageerror", (e) => fallbackErrors.push(String(e)));
    await failed.route("**/benchmarks/urban_voxel_0.json", (r) => r.abort());
    await failed.route("**/assets/forecast_examples.json", (r) => r.abort());
    await failed.route("**/benchmarks/tracking/urban_ablation.json", (r) =>
      r.abort(),
    );
    await failed.goto(base);
    await failed.waitForFunction(() =>
      document.getElementById("data-status").textContent.includes("读取失败"),
    );
    await failed.waitForFunction(() =>
      document.getElementById("native-status").textContent.includes("读取失败"),
    );
    assert.ok(
      await failed.locator('[data-native-scene="highway"]').isDisabled(),
    );
    assert.ok(
      (await failed.locator("#example-description").textContent()).includes(
        "读取失败",
      ),
    );
    assert.ok(await failed.locator("#save-forecast").isDisabled());
    await failed.locator("#explorer").scrollIntoViewIfNeeded();
    await failed.waitForFunction(() =>
      document.getElementById("frame-label").textContent.includes("帧 120"),
    );
    await failed.route("**/assets/replay/highway/index.json", (r) => r.abort());
    await failed.locator("#viewer-scene").selectOption("highway");
    await failed.waitForFunction(() =>
      document
        .getElementById("viewer-summary")
        .textContent.includes("读取失败"),
    );
    assert.equal(await failed.locator("#frame-label").textContent(), "—");
    assert.ok(await failed.locator("#frame-camera").isHidden());
    assert.ok(await failed.locator("#frame-json").isHidden());
    await failed.locator("#viewer-scene").selectOption("urban");
    await failed.waitForFunction(() =>
      document.getElementById("frame-label").textContent.includes("帧 120"),
    );
    await failed.route("**/assets/replay/urban/000140.json", (r) => r.abort());
    await failed.locator("#next-frame").click();
    await failed.waitForFunction(() =>
      document
        .getElementById("viewer-summary")
        .textContent.includes("读取失败"),
    );
    assert.ok(
      (await failed.locator("#frame-camera").getAttribute("src")).endsWith(
        "urban/000120.jpg",
      ),
    );
    assert.deepEqual(fallbackErrors, []);
    const nojs = await browser.newPage({ javaScriptEnabled: false });
    await nojs.goto(base);
    assert.equal(await nojs.locator("#raw-latency").textContent(), "888.5 ms");
    assert.equal(
      await nojs.locator("#native-cpp-time").textContent(),
      "3.2 ms",
    );
    assert.ok(await nojs.locator("noscript").isVisible());
    console.log(
      "PASS: C++ reports and ablations, 42 frames, image synchronization, layers, selection, zoom/pan, downloads, forecasts, video, commands, mobile and failure fallbacks.",
    );
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
