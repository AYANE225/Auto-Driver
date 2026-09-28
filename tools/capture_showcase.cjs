// Capture fixed views for README previews: node tools/capture_showcase.cjs [URL] [OUT]
const { chromium } = require("playwright");
const fs = require("node:fs/promises");
const path = require("node:path");
async function main() {
  const base = process.argv[2] || "http://127.0.0.1:8765/";
  const out = process.argv[3] || "outputs/showcase-previews";
  await fs.mkdir(out, { recursive: true });
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
      deviceScaleFactor: 1,
    });
    await page.goto(base, { waitUntil: "networkidle" });
    await page.locator("#pointcloud").scrollIntoViewIfNeeded();
    await page.waitForFunction(
      () => document.getElementById("cloud-canvas").dataset.frame === "120",
    );
    await page.locator("#cloud-density").selectOption("20000");
    for (const [name, selector] of [
      ["pointcloud_preview", "#cloud-stage"],
      ["stage_preview", "#latency-lab .latency-lab-panel"],
      ["association_preview", "#association-lab .association-panel"],
    ])
      await page.locator(selector).screenshot({
        path: path.join(out, `${name}.jpg`),
        type: "jpeg",
        quality: 92,
      });
    await page.locator("#driving").scrollIntoViewIfNeeded();
    await page.waitForFunction(
      () =>
        document.getElementById("driving-canvas").dataset.scene === "obstacle",
    );
    await page.locator("#driving-time").evaluate((el) => {
      el.value = "28";
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page
      .locator("#driving .driving-panel")
      .screenshot({
        path: path.join(out, "driving_preview.jpg"),
        type: "jpeg",
        quality: 92,
      });
  } finally {
    await browser.close();
  }
}
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
