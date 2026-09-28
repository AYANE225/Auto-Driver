// Verify exported Python results, live controls, async recovery and CARLA videos.
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
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("response", (r) => {
      if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
    });
    const get = async (relative) => {
      const response = await context.request.get(new URL(relative, base).href);
      assert.equal(response.status(), 200, relative);
      return response;
    };
    const json = async (relative) => (await get(relative)).json();
    const index = await json("assets/driving/index.json");
    const reports = {};
    for (const mode of ["gt", "lidar"]) {
      assert.equal(index.modes[mode].scenarios.length, 10);
      assert.equal(index.modes[mode].passed, 10);
      for (const entry of index.modes[mode].scenarios) {
        const bytes = await (await get(`assets/driving/${entry.file}`)).body();
        assert.equal(
          crypto.createHash("sha256").update(bytes).digest("hex"),
          entry.sha256,
        );
        const data = JSON.parse(bytes);
        reports[`${mode}/${entry.name}`] = data;
        assert.deepEqual(data.metrics, entry.metrics);
        assert.equal(data.source_sha256, entry.source_sha256);
        assert.equal(data.detector, mode);
        assert.ok(data.frames.length > 20);
        for (let i = 0; i < data.frames.length; i++) {
          const frame = data.frames[i];
          assert.ok(
            Number.isFinite(frame.ego.x) && Number.isFinite(frame.ego.y),
          );
          assert.ok(
            frame.ego.speed >= 0 &&
              Math.abs(frame.command.acceleration) <= 7.0001,
          );
          if (i) assert.ok(frame.time_s > data.frames[i - 1].time_s);
        }
      }
    }
    await page.goto(base, { waitUntil: "networkidle" });
    await page.locator("#driving").scrollIntoViewIfNeeded();
    const ready = async (mode, scenario) =>
      page.waitForFunction(
        ({ mode, scenario }) => {
          const d = document.getElementById("driving-canvas").dataset;
          return d.input === mode && d.scene === scenario;
        },
        { mode, scenario },
      );
    const seek = async (value) =>
      page.locator("#driving-time").evaluate((e, value) => {
        e.value = String(value);
        e.dispatchEvent(new Event("input", { bubbles: true }));
      }, value);
    const pixels = () =>
      page.locator("#driving-canvas").evaluate((c) => c.toDataURL());
    await ready("gt", "obstacle");
    for (const mode of ["gt", "lidar"]) {
      await page.locator("#driving-input").selectOption(mode);
      for (const entry of index.modes[mode].scenarios) {
        await page.locator("#driving-scene").selectOption(entry.name);
        await ready(mode, entry.name);
        const report = reports[`${mode}/${entry.name}`],
          sample = Math.min(25, report.frames.length - 1);
        await seek(sample);
        assert.ok(
          (await page.locator("#driving-speed").textContent()).includes(
            report.frames[sample].ego.speed.toFixed(1),
          ),
        );
        assert.equal(
          await page.locator("#driving-accel").textContent(),
          `${report.frames[sample].command.acceleration.toFixed(1)} m/s²`,
        );
        assert.ok(
          (await page.locator("#driving-command").textContent()).includes(
            `--scenario ${entry.name} --detector ${mode}`,
          ),
        );
        assert.equal(await page.locator("#driving-metrics > div").count(), 4);
      }
    }
    await page.locator("#driving-input").selectOption("gt");
    await page.locator("#driving-scene").selectOption("obstacle");
    await ready("gt", "obstacle");
    await seek(25);
    // Interpolation must visibly advance between recorded 0.2 s samples.
    await page.locator("#driving-rate").selectOption("0.5");
    await page.locator("#driving-play").click();
    const motion = await page.evaluate(
      () =>
        new Promise((resolve) => {
          const samples = [],
            start = performance.now();
          function tick(time) {
            const c = document.getElementById("driving-canvas");
            samples.push({
              t: Number(c.dataset.renderTime),
              x: Number(c.dataset.egoX),
              frame: Number(c.dataset.frame),
              image: c.toDataURL(),
            });
            if (time - start < 800) requestAnimationFrame(tick);
            else resolve(samples);
          }
          requestAnimationFrame(tick);
        }),
    );
    await page.locator("#driving-play").click();
    assert.ok(
      new Set(motion.map((s) => s.image)).size > 8,
      "visible motion between source frames",
    );
    assert.ok(
      new Set(motion.map((s) => s.x)).size >
        new Set(motion.map((s) => s.frame)).size * 2,
    );
    const frames = reports["gt/obstacle"].frames;
    for (const s of motion) {
      const a = frames[s.frame],
        b = frames[Math.min(s.frame + 1, frames.length - 1)];
      const u = (s.t - a.time_s) / (b.time_s - a.time_s);
      assert.ok(Math.abs(s.x - (a.ego.x + (b.ego.x - a.ego.x) * u)) < 1e-6);
    }
    const stoppedTime = Number(
      await page.locator("#driving-canvas").getAttribute("data-render-time"),
    );
    await page.locator("#driving-play").click();
    await page.waitForFunction(
      (t) =>
        Number(document.getElementById("driving-canvas").dataset.renderTime) >
        t + 0.05,
      stoppedTime,
    );
    await page.locator("#driving-play").click();
    await seek(25);
    await page.locator("#driving-smooth").uncheck();
    await page.locator("#driving-play").click();
    await page.waitForFunction(
      () =>
        Number(document.getElementById("driving-canvas").dataset.renderTime) >
        5.05,
    );
    await page.locator("#driving-play").click();
    const rawIndex = Number(await page.locator("#driving-time").inputValue());
    assert.equal(
      Number(await page.locator("#driving-canvas").getAttribute("data-ego-x")),
      frames[rawIndex].ego.x,
    );
    await page.locator("#driving-smooth").check();
    await page.locator("#driving-rate").selectOption("1");
    await seek(25);
    const overview = await pixels();
    await page.locator("#driving-view-mode").selectOption("follow");
    assert.notEqual(await pixels(), overview);
    await page.locator("#driving-view-mode").selectOption("overview");
    await page.locator("#driving-next").click();
    assert.equal(await page.locator("#driving-time").inputValue(), "26");
    await page.locator("#driving-previous").click();
    assert.equal(await page.locator("#driving-time").inputValue(), "25");
    for (const kind of ["acceleration", "steering", "speed"]) {
      await page.locator("#driving-chart").selectOption(kind);
      assert.equal(
        await page.locator("#driving-speed-chart").getAttribute("data-series"),
        kind,
      );
    }
    await page
      .locator("#driving-speed-chart")
      .click({ position: { x: 36, y: 60 } });
    assert.equal(await page.locator("#driving-time").inputValue(), "0");
    await seek(25);
    const saved = page.waitForEvent("download");
    await page.locator("#driving-save").click();
    assert.match((await saved).suggestedFilename(), /^obstacle-gt-5.00s\.png$/);
    assert.equal(await page.locator("#driving-candidates").isChecked(), false);
    await page.locator("#driving-candidates").check();
    const before = await pixels();
    await page.locator("#driving-candidates").uncheck();
    assert.notEqual(await pixels(), before);
    await page.locator("#driving-candidates").check();
    await page.locator("#driving-tracks").check();
    assert.notEqual(await pixels(), before);
    await page.locator("#driving-canvas").focus();
    await page.keyboard.press("ArrowRight");
    assert.equal(await page.locator("#driving-time").inputValue(), "26");
    await page.keyboard.press("End");
    assert.match(await page.locator("#driving-status").textContent(), /到达/);
    await page.locator("#driving-play").click();
    await page.waitForFunction(
      () => Number(document.getElementById("driving-time").value) > 1,
    );
    await page.locator("#driving-play").click();
    assert.equal(
      await page.locator("#driving-play").getAttribute("aria-pressed"),
      "false",
    );
    const pending = page.waitForEvent("download");
    await page.locator("#driving-download").click();
    const download = await pending;
    assert.equal(download.suggestedFilename(), "obstacle.json");
    const stream = await download.createReadStream(),
      chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    assert.equal(JSON.parse(Buffer.concat(chunks)).scenario, "obstacle");
    await page.locator("#driving-scene").selectOption("emergency");
    await ready("gt", "emergency");
    const emergency = reports["gt/emergency"].frames.findIndex(
      (f) => f.status === "emergency_stop",
    );
    assert.ok(emergency >= 0);
    await page
      .locator(`#driving-events button[data-index="${emergency}"]`)
      .click();
    assert.equal(
      Number(await page.locator("#driving-time").inputValue()),
      emergency,
    );
    assert.match(
      await page.locator("#driving-status").textContent(),
      /紧急制动/,
    );
    assert.match(
      await page.locator("#driving-rejections").textContent(),
      /不将回退轨迹标为可行/,
    );
    await page.locator("#driving-scene").selectOption("traffic_light");
    await ready("gt", "traffic_light");
    await seek(0);
    assert.equal(await page.locator("#driving-signal").textContent(), "红灯");
    await seek(reports["gt/traffic_light"].frames.length - 1);
    assert.equal(await page.locator("#driving-signal").textContent(), "绿灯");
    for (const mode of ["gt", "lidar"]) {
      const report = await json(`assets/driving/carla_${mode}.json`);
      assert.equal(report.autopilot, false);
      assert.equal(report.metrics.success, true);
      assert.equal(report.metrics.collision_events, 0);
      assert.equal(report.metrics.lane_invasion_events, 0);
      const video = page.locator(`#driving-carla-${mode}-video`);
      await video.scrollIntoViewIfNeeded();
      await video.evaluate((v) => v.play());
      await page.waitForFunction(
        (id) => document.getElementById(id).currentTime > 0.1,
        `driving-carla-${mode}-video`,
      );
      assert.ok(
        Math.abs(
          (await video.evaluate((v) => v.duration)) - report.video.duration_s,
        ) < 0.1,
      );
      await video.evaluate((v) => v.pause());
      for (const time of [report.video.duration_s * 0.7, 1.0, 0.05]) {
        await video.evaluate((v, t) => {
          v.currentTime = t;
        }, time);
        const sample = report.frames.findLastIndex(
          (f) => f.video_time_s <= time + 1e-5,
        );
        const panel = page.locator(`#driving-carla-${mode}-telemetry`);
        await page.waitForFunction(
          ({ mode, sample }) =>
            document.getElementById(`driving-carla-${mode}-telemetry`).dataset
              .frame === String(sample),
          { mode, sample },
        );
        assert.equal(
          await panel.locator('[data-value="speed"]').textContent(),
          `${report.frames[sample].ego.speed.toFixed(2)} m/s`,
        );
        assert.equal(
          await panel.locator('[data-value="brake"]').textContent(),
          `${(report.frames[sample].command.brake * 100).toFixed(0)}%`,
        );
      }
    }
    await page.locator("#driving-scene").selectOption("obstacle");
    await ready("gt", "obstacle");
    await seek(25);
    for (const width of [1440, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `overflow ${width}`,
      );
    }
    if (process.env.WEBSITE_SCREENSHOTS) {
      await fs.mkdir(process.env.WEBSITE_SCREENSHOTS, { recursive: true });
      await page.locator("#driving").screenshot({
        path: path.join(process.env.WEBSITE_SCREENSHOTS, "driving-mobile.png"),
      });
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.locator("#driving").screenshot({
        path: path.join(process.env.WEBSITE_SCREENSHOTS, "driving-desktop.png"),
      });
    }
    assert.deepEqual(errors, []);
    const failed = await context.newPage();
    await failed.route("**/driving/gt/obstacle.json", (r) => r.abort());
    await failed.goto(base);
    await failed.locator("#driving").scrollIntoViewIfNeeded();
    await failed.waitForFunction(() =>
      document
        .getElementById("driving-status")
        .textContent.includes("读取失败"),
    );
    assert.ok(await failed.locator("#driving-play").isDisabled());
    await failed.locator("#driving-scene").selectOption("cruise");
    await failed.waitForFunction(
      () =>
        document.getElementById("driving-canvas").dataset.scene === "cruise",
    );
    const racing = await context.newPage();
    let release, seen;
    const gate = new Promise((r) => (release = r)),
      requested = new Promise((r) => (seen = r));
    await racing.route("**/driving/gt/obstacle.json", async (r) => {
      seen();
      await gate;
      await r.continue();
    });
    await racing.goto(base);
    await racing.locator("#driving").scrollIntoViewIfNeeded();
    await requested;
    await racing.locator("#driving-scene").selectOption("curve");
    await racing.waitForFunction(
      () => document.getElementById("driving-canvas").dataset.scene === "curve",
    );
    const completed = racing.waitForResponse("**/driving/gt/obstacle.json");
    release();
    await completed;
    await racing.waitForLoadState("networkidle");
    assert.equal(
      await racing.locator("#driving-canvas").getAttribute("data-scene"),
      "curve",
    );
    console.log(
      "PASS: 20 driving replays and hashes; feedback, scenarios, braking, signals, playback, download, 2 CARLA videos, mobile layouts, failed and delayed requests.",
    );
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
