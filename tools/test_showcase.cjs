// Companion checks for 3D points, stage timings, association examples and videos.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const path = require("node:path");
const base = process.argv[2] || "http://127.0.0.1:8765/";
async function main() {
  const browser = await chromium.launch();
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 },
    });
    const page = await context.newPage(),
      errors = [];
    const gifRequests = [];
    page.on("request", (r) => {
      if (/\/driving\/gifs\/.*\.gif$/.test(r.url())) gifRequests.push(r.url());
    });
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("response", (r) => {
      if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
    });
    const get = async (relative) => {
      const r = await context.request.get(new URL(relative, base).href);
      assert.equal(r.status(), 200, relative);
      return r;
    };
    const json = async (relative) => (await get(relative)).json();
    const text = (selector) => page.locator(selector).textContent();
    const pixels = () =>
      page.locator("#cloud-canvas").evaluate((c) => c.toDataURL());
    const range = (selector, value) =>
      page.locator(selector).evaluate((el, value) => {
        el.value = String(value);
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }, value);
    const ready = (scene, frame) =>
      page.waitForFunction(
        ({ scene, frame }) => {
          const canvas = document.getElementById("cloud-canvas");
          return (
            canvas.dataset.frame === String(frame) &&
            canvas.dataset.scene === scene
          );
        },
        { scene, frame },
      );
    const shot = async (name, selector) => {
      if (!process.env.WEBSITE_SCREENSHOTS) return;
      await fs.mkdir(process.env.WEBSITE_SCREENSHOTS, { recursive: true });
      await page.locator(selector).screenshot({
        path: path.join(process.env.WEBSITE_SCREENSHOTS, `${name}.png`),
      });
    };
    await page.goto(base, { waitUntil: "networkidle" });
    // A gallery should not download every animation on page load. Playback is
    // explicit, stoppable, and only one card can animate at a time.
    assert.equal(gifRequests.length, 0);
    const gifIndex = await json("assets/driving/gifs/index.json");
    assert.equal(gifIndex.clips.length, 5);
    assert.equal(await page.locator("[data-driving-gif]").count(), 5);
    for (const clip of gifIndex.clips) {
      const asset = await (
        await get(`assets/driving/gifs/${clip.file}`)
      ).body();
      assert.equal(asset.length, clip.bytes);
      assert.equal(
        crypto.createHash("sha256").update(asset).digest("hex"),
        clip.sha256,
      );
      assert.equal(asset.subarray(0, 6).toString(), "GIF89a");
      assert.deepEqual(
        [asset.readUInt16LE(6), asset.readUInt16LE(8)],
        clip.size_px,
      );
      const source = await (
        await get(clip.source.replace(/^docs\//, ""))
      ).body();
      assert.equal(
        crypto.createHash("sha256").update(source).digest("hex"),
        clip.source_sha256,
      );
      const report = JSON.parse(source);
      assert.deepEqual(clip.metrics, report.metrics);
      assert.deepEqual(
        clip.source_frame_indices.map((i) => report.frames[i].time_s),
        clip.source_times_s,
      );
      assert.equal(clip.interpolation, "none");
      assert.equal(clip.playback_rate, 1);
      assert.equal(
        clip.frame_durations_ms.length,
        clip.source_frame_indices.length + 1,
      );
      assert.equal(clip.frame_durations_ms.at(-1), 1200);
    }
    const gifCards = page.locator("[data-driving-gif]");
    const firstGif = gifCards.nth(0),
      secondGif = gifCards.nth(1);
    await firstGif.locator("button").click();
    await page.waitForFunction(() => {
      const img = document.querySelector("[data-driving-gif] img");
      return (
        img.src.endsWith(".gif") && img.complete && img.naturalWidth === 1000
      );
    });
    assert.equal(
      await firstGif.locator("button").getAttribute("aria-pressed"),
      "true",
    );
    await secondGif.locator("button").click();
    assert.equal(
      await firstGif.locator("button").getAttribute("aria-pressed"),
      "false",
    );
    assert.equal(
      await page
        .locator('[data-driving-gif] button[aria-pressed="true"]')
        .count(),
      1,
    );
    await secondGif.locator("button").click();
    assert.ok(
      (await secondGif.locator("img").getAttribute("src")).endsWith(".jpg"),
    );
    await shot("driving-gifs", "#driving-gifs");
    await page.locator("#pointcloud").scrollIntoViewIfNeeded();
    await ready("urban", 120);
    for (const scene of ["urban", "highway"]) {
      const manifest = await json(`assets/cloud/${scene}/index.json`);
      const replay = await json(`assets/replay/${scene}/index.json`);
      assert.equal(manifest.dtype, "int16_le_xyz");
      assert.equal(manifest.scale_m, 0.002);
      assert.deepEqual(
        manifest.frames.map((f) => f.frame_id),
        replay.frames.map((f) => f.frame_id),
      );
      for (const row of manifest.frames) {
        const bytes = await (
          await get(`assets/cloud/${scene}/${row.file}`)
        ).body();
        assert.equal(bytes.length, row.points * 6);
        assert.ok(row.points > 0 && row.points <= 20000);
        assert.equal(
          crypto.createHash("sha256").update(bytes).digest("hex"),
          row.sha256,
        );
        for (let i = 0; i < bytes.length; i += 6) {
          for (let k = 0; k < 3; k++) {
            const value = bytes.readInt16LE(i + k * 2) * manifest.scale_m;
            assert.ok(
              value >= manifest.bounds_m[k][0] - 0.001 &&
                value <= manifest.bounds_m[k][1] + 0.001,
            );
          }
        }
      }
    }
    assert.equal(
      await page.locator("#cloud-canvas").getAttribute("data-points"),
      "12000",
    );
    const initial = await pixels();
    await page.locator('[data-cloud-view="top"]').click();
    assert.notEqual(await pixels(), initial);
    await page.locator('[data-cloud-view="front"]').click();
    assert.notEqual(await pixels(), initial);
    await page.locator('[data-cloud-view="orbit"]').click();
    assert.equal(await pixels(), initial);
    await page.locator("#cloud-canvas").focus();
    await page.keyboard.press("ArrowLeft");
    assert.notEqual(await pixels(), initial);
    await page.keyboard.press("+");
    assert.equal(await page.locator("#cloud-zoom").inputValue(), "1.1");
    await page.locator('[data-cloud-view="orbit"]').click();
    const box = await page.locator("#cloud-canvas").boundingBox();
    await page.mouse.move(box.x + 100, box.y + 140);
    await page.mouse.down();
    await page.mouse.move(box.x + 170, box.y + 160);
    await page.mouse.up();
    assert.notEqual(await pixels(), initial);
    await page.locator('[data-cloud-view="orbit"]').click();
    for (const id of ["cloud-gt", "cloud-tracks", "cloud-paths"]) {
      const before = await pixels();
      await page.locator("#" + id).click();
      assert.notEqual(await pixels(), before);
      await page.locator("#" + id).click();
    }
    await page.locator("#cloud-density").selectOption("20000");
    assert.equal(
      await page.locator("#cloud-canvas").getAttribute("data-points"),
      "20000",
    );
    assert.notEqual(await pixels(), initial);
    await shot("pointcloud-desktop", "#pointcloud");
    await page.locator("#cloud-spin").click();
    await page.waitForFunction(
      (start) => document.getElementById("cloud-canvas").toDataURL() !== start,
      await pixels(),
    );
    await page.locator("#cloud-spin").click();
    assert.equal(
      await page.locator("#cloud-spin").getAttribute("aria-pressed"),
      "false",
    );
    const pending = page.waitForEvent("download");
    await page.locator("#cloud-save").click();
    const download = await pending;
    assert.equal(download.suggestedFilename(), "urban_3d_120.png");
    const stream = await download.createReadStream(),
      chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    assert.equal(
      Buffer.concat(chunks).subarray(0, 8).toString("hex"),
      "89504e470d0a1a0a",
    );
    await range("#cloud-frame", 0);
    await ready("urban", 0);
    assert.match(await text("#cloud-status"), /0 条确认轨迹/);
    await range("#cloud-frame", 20);
    await ready("urban", 399);
    await page.locator("#cloud-scene").selectOption("highway");
    await ready("highway", 120);
    assert.ok(
      (await page.locator("#cloud-camera").getAttribute("src")).endsWith(
        "/highway/000120.jpg",
      ),
    );
    // A delayed older camera must never overwrite a newer point cloud/image pair.
    let release, seen;
    const gate = new Promise((r) => {
        release = r;
      }),
      requested = new Promise((r) => {
        seen = r;
      });
    await page.route("**/replay/highway/000200.jpg", async (route) => {
      seen();
      await gate;
      await route.continue();
    });
    await range("#cloud-frame", 10);
    await requested;
    await range("#cloud-frame", 11);
    await ready("highway", 220);
    const done = page.waitForResponse("**/replay/highway/000200.jpg");
    release();
    await done;
    assert.equal(
      await page.locator("#cloud-canvas").getAttribute("data-frame"),
      "220",
    );
    assert.ok(
      (await page.locator("#cloud-camera").getAttribute("src")).endsWith(
        "/highway/000220.jpg",
      ),
    );
    await page.unroute("**/replay/highway/000200.jpg");
    await page.locator("#latency-lab").scrollIntoViewIfNeeded();
    for (const scene of ["urban", "highway"]) {
      await page.locator(`[data-latency-scene="${scene}"]`).click();
      const a = await json(`benchmarks/tracking/${scene}_python_original.json`),
        b = await json(`benchmarks/tracking/${scene}_cpp_threshold.json`);
      assert.equal(
        await text("#stage-old-total"),
        `${a.latency_ms.total_ms.mean.toFixed(1)} ms`,
      );
      assert.equal(
        await text("#stage-new-total"),
        `${b.latency_ms.total_ms.mean.toFixed(1)} ms`,
      );
      assert.equal(
        await text("#stage-p95"),
        `${b.latency_ms.total_ms.p95.toFixed(1)} ms`,
      );
      for (const key of ["detect_ms", "fuse_ms", "track_ms", "predict_ms"]) {
        await page.locator(`.stage-legend [data-stage="${key}"]`).click();
        assert.equal(
          await text("#stage-change"),
          `${a.latency_ms[key].mean.toFixed(1)} → ${b.latency_ms[key].mean.toFixed(1)} ms`,
        );
        const width = await page
          .locator(`#stage-new [data-stage="${key}"]`)
          .evaluate((e) => parseFloat(e.style.width));
        assert.ok(Math.abs(width - b.latency_ms[key].mean / 2) < 1e-6);
      }
    }
    await shot("stage-timings", "#latency-lab");
    assert.match(await text("#association-old-result"), /^0 个/);
    assert.match(await text("#association-new-result"), /^1 个/);
    await range("#association-threshold", 0.59);
    assert.match(await text("#association-new-result"), /^2 个.*1.18/);
    await range("#association-threshold", 0.61);
    assert.match(await text("#association-new-result"), /^0 个/);
    await page.locator("#association-case").selectOption("weight");
    assert.match(await text("#association-new-result"), /^1 个.*0.99/);
    await page.locator("#association-case").selectOption("conflict");
    assert.equal(await text("#association-threshold-label"), "0.60");
    await shot("association", "#association-lab");
    const gallery = await json("assets/gallery.json");
    for (const name of ["kitti", "vggt"]) {
      const video = page.locator(`#${name}-video`);
      await video.scrollIntoViewIfNeeded();
      await video.evaluate((v) => v.play());
      await page.waitForFunction(
        (id) => document.getElementById(id).currentTime > 0.15,
        `${name}-video`,
      );
      assert.ok(
        Math.abs(
          (await video.evaluate((v) => v.duration)) -
            Number(gallery[name].source_metadata.format.duration),
        ) < 0.1,
      );
      await video.evaluate((v) => v.pause());
    }
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `overflow ${width}`,
      );
    }
    await shot("pointcloud-mobile", "#pointcloud");
    await shot("association-mobile", "#association-lab");
    assert.deepEqual(errors, []);
    const failed = await context.newPage();
    const fallbackErrors = [];
    failed.on("pageerror", (e) => fallbackErrors.push(String(e)));
    await failed.route("**/cloud/urban/000120.bin", (r) => r.abort());
    await failed.route("**/tracking/urban_python_original.json", (r) =>
      r.abort(),
    );
    await failed.goto(base);
    await failed.locator("#pointcloud").scrollIntoViewIfNeeded();
    await failed.waitForFunction(() =>
      document
        .getElementById("cloud-status")
        .textContent.includes("精细点云不可用"),
    );
    assert.equal(
      await failed.locator("#cloud-canvas").getAttribute("data-points"),
      "2500",
    );
    assert.ok(
      (await failed.locator("#stage-status").textContent()).includes(
        "读取失败",
      ),
    );
    await failed.route("**/replay/highway/index.json", (r) => r.abort());
    await failed.locator("#cloud-scene").selectOption("highway");
    await failed.waitForFunction(() =>
      document
        .getElementById("cloud-status")
        .textContent.includes("场景读取失败"),
    );
    assert.ok(await failed.locator("#cloud-camera").isHidden());
    assert.ok(await failed.locator("#cloud-save").isDisabled());
    await failed.locator("#cloud-scene").selectOption("urban");
    await failed.waitForFunction(
      () => document.getElementById("cloud-canvas").dataset.frame === "120",
    );
    assert.deepEqual(fallbackErrors, []);
    const racing = await context.newPage();
    let releaseScene, seenScene;
    const sceneGate = new Promise((resolve) => {
      releaseScene = resolve;
    });
    const sceneRequest = new Promise((resolve) => {
      seenScene = resolve;
    });
    await racing.route("**/replay/urban/index.json", async (route) => {
      seenScene();
      await sceneGate;
      await route.continue();
    });
    await racing.goto(base);
    await racing.locator("#pointcloud").scrollIntoViewIfNeeded();
    await sceneRequest;
    await racing.locator("#cloud-scene").selectOption("highway");
    await racing.waitForFunction(
      () => document.getElementById("cloud-canvas").dataset.scene === "highway",
    );
    const oldSceneDone = racing.waitForResponse("**/replay/urban/index.json");
    releaseScene();
    await oldSceneDone;
    await racing.waitForLoadState("networkidle");
    assert.equal(
      await racing.locator("#cloud-canvas").getAttribute("data-scene"),
      "highway",
    );
    assert.ok(
      (await racing.locator("#cloud-camera").getAttribute("src")).endsWith(
        "/highway/000120.jpg",
      ),
    );

    console.log(
      "PASS: 5 GIF excerpts, source hashes, frame provenance and playback controls; 42 dense point files and hashes; 3D views, input controls, camera synchronization, download, density fallback; stage timings, assignment examples, gallery video, mobile layouts.",
    );
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
