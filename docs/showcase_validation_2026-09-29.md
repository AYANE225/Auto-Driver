# Showcase and benchmark validation — 2026-09-29

This update completes the weighted-voxel replay comparison and town-disjoint
forecasting experiment, reorganizes the README, and adds a static project page.
The four full replay reports and forecasting artifacts were completed in the
preceding work session and checked against the current implementation and local
recordings before publication.

| Check | Result |
|---|---|
| Core tests, Python 3.12.13 / NumPy 1.26.4 | 92 passed |
| Ruff, `src`, `carla`, `tools` | Passed |
| JavaScript syntax and `git diff --check` | Passed |
| Replay comparisons | Both pairs contain 400 frames with matching evaluation settings and pipeline configuration except voxel size |
| Forecasting provenance | All six input SHA-256 hashes match local recordings; saved Ridge alpha and coefficient dimensions match the report |
| Benchmark figures | PNG and SVG generated from report JSON; axes, values and layouts visually inspected |
| Website, Chromium | Urban/highway video and benchmark switching; all/moving forecast switching; report links and copy command checked |
| Video playback | H.264 MP4 decoded in Chromium, duration 24 s; original GIF timing retained |
| Responsive layout | No document overflow at 320, 390, 768 or 1440 px; desktop and mobile screenshots inspected |
| Fallback | Static urban summary and report links remain available without JavaScript |
| Browser errors | No page exceptions or HTTP resource errors during the local check |
| Documentation | Local links resolve in README, CARLA guide, engineering guide, benchmark protocol and website |

The additional regression tests cover weighted voxel density, raw-point box
extents, invalid voxel sizes, constant-velocity and constant-acceleration
forecasting, global-frame translation/rotation, and rejection of history windows
that cross actor boundaries or missing ticks.

Local commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -o addopts='' -q src/perception_core/tests
ruff check --config src/perception_core/pyproject.toml src carla tools
node --check docs/assets/site.js
python tools/make_showcase_assets.py --out outputs/showcase-assets --video
python -m http.server 8000 --directory docs
```

The browser check used Playwright in a temporary tooling directory; the site
has no runtime JavaScript package dependencies. Full CARLA comparisons are
documented in [benchmarks/README.md](benchmarks/README.md). ROS 2 and KITTI checks
belong to the [preceding integration validation](validation_2026-09-29.md) and
were not rerun for this presentation update.
