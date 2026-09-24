# syntax=docker/dockerfile:1
#
# Framework-agnostic perception_core image.
#
# Runs the offline pipeline, the unit tests and the latency/prediction
# benchmarks with no ROS / CUDA / CARLA — mirroring the CI environment so a
# plain `docker run` reproduces the published numbers on any machine.
#
#   docker build -t auto-driver-core .
#   docker run --rm auto-driver-core                                  # offline demo report
#   docker run --rm auto-driver-core pytest -q src/perception_core/tests
#   docker run --rm auto-driver-core python tools/benchmark.py --detector gt --budget-ms 150
FROM python:3.11-slim

# The BEV renderer / doc figures use matplotlib's headless Agg backend, so no
# system GUI libraries are required.
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only what the core needs (see .dockerignore for the exclusions).
COPY src/perception_core ./src/perception_core
COPY tools ./tools

# Install the core with the test extra plus matplotlib for the headless
# renderer. open3d (part of the [viz] extra) is intentionally skipped to keep
# the image slim — the Agg BEV renderer needs only matplotlib.
RUN pip install ./src/perception_core[test] matplotlib

# Default command: run the offline end-to-end pipeline headless and emit metrics.
CMD ["python", "tools/run_demo.py", "--frames", "60", "--detector", "lidar", \
     "--no-video", "--report", "/tmp/metrics.json"]
