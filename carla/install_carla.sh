#!/usr/bin/env bash
# Set up CARLA 0.9.15 (server + Python 3.8 client env) for scenario recording.
#
# CARLA's client library targets Python 3.7/3.8, which is deliberately decoupled
# from the ROS 2 Humble side (Python 3.11): we RECORD scenarios here, then REPLAY
# the resulting dataset through perception_core / ROS 2. The two never need to
# share an interpreter.
#
# The server package is several GB and needs a GPU + display (or -RenderOffScreen).
set -euo pipefail

CARLA_VERSION="${CARLA_VERSION:-0.9.15}"
CARLA_ROOT="${CARLA_ROOT:-$HOME/CARLA_${CARLA_VERSION}}"
VENV="${VENV:-$HOME/.venvs/carla-client}"
URL="https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_${CARLA_VERSION}.tar.gz"

echo ">> CARLA ${CARLA_VERSION}"
echo "   server dir : ${CARLA_ROOT}"
echo "   client venv: ${VENV}"

# --- 1. Server package ------------------------------------------------------
if [ ! -d "${CARLA_ROOT}" ]; then
  echo ">> downloading CARLA server (large; several GB) ..."
  mkdir -p "${CARLA_ROOT}"
  tmp="$(mktemp -d)"
  wget -O "${tmp}/carla.tar.gz" "${URL}"
  tar -xzf "${tmp}/carla.tar.gz" -C "${CARLA_ROOT}"
  rm -rf "${tmp}"
else
  echo ">> server already present, skipping download"
fi

# --- 2. Python 3.8 client env ----------------------------------------------
if ! command -v python3.8 >/dev/null 2>&1; then
  echo "!! python3.8 not found. Install it first (e.g. sudo apt install python3.8 python3.8-venv)"
  exit 1
fi
python3.8 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip
pip install "carla==${CARLA_VERSION}" numpy pyyaml pillow

cat <<EOF

Done.

Run the server (headless):
  ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen -quality-level=Epic

Then, in another shell, record a scenario with the client env:
  source ${VENV}/bin/activate
  python record_scenario.py --scenario config/scenarios/urban.yaml --out data/urban --images

Replay it through the pipeline (any env with perception_core, no CARLA needed):
  python replay_demo.py --dataset data/urban --gif docs/screenshots/demo_carla.gif
EOF
