# Auto-Driver — common developer tasks.  Run `make` (or `make help`) for the list.
#
# PY lets you point at a specific interpreter, e.g.  make test PY=python3.11
PY   ?= python
CORE := src/perception_core

.DEFAULT_GOAL := help
.PHONY: help install install-dev test cov lint typecheck \
        demo demo-imm kitti bench predict figures \
        docker-core docker-ros clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install:  ## Install perception_core (editable) with the viz extra
	$(PY) -m pip install -e "$(CORE)[viz]"

install-dev:  ## Install with test + lint + type-check tooling
	$(PY) -m pip install -e "$(CORE)[dev,viz]"

test:  ## Run the unit tests
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q $(CORE)/tests

cov:  ## Run the unit tests with a coverage report
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest -q $(CORE)/tests \
	  --cov=perception_core --cov-report=term-missing

lint:  ## Lint the core with ruff (pinned rule set)
	ruff check $(CORE)

typecheck:  ## Static type-check the core with mypy
	mypy $(CORE)/perception_core

demo:  ## Offline BEV demo (LiDAR detector) -> GIF + metrics
	$(PY) tools/run_demo.py --frames 60 --detector lidar \
	  --gif docs/screenshots/demo_lidar.gif --report metrics.json

demo-imm:  ## Offline demo with the IMM (CV+CT) tracker bank
	$(PY) tools/run_demo.py --frames 80 --detector lidar --tracker imm \
	  --gif docs/screenshots/demo_imm.gif --report metrics_imm.json

kitti:  ## Real KITTI-raw GT-replay demo (needs data/kitti)
	$(PY) tools/run_kitti_demo.py --root data/kitti --drive 14 --detector gt \
	  --gif docs/screenshots/demo_kitti.gif --report metrics_kitti.json

bench:  ## Latency benchmark with a real-time budget gate
	$(PY) tools/benchmark.py --detector gt --tracker imm --frames 120 --budget-ms 150

predict:  ## ADE / FDE prediction accuracy on the analytic manoeuvre bank
	$(PY) tools/eval_prediction.py --report prediction.json

figures:  ## Regenerate the README architecture + results figures
	$(PY) tools/make_docs_figures.py --out docs/screenshots

docker-core:  ## Build the framework-agnostic core image
	docker build -t auto-driver-core .

docker-ros:  ## Build the ROS 2 Humble workspace image
	docker build -f docker/Dockerfile.ros2 -t auto-driver-ros .

clean:  ## Remove caches, coverage and colcon build artefacts
	rm -rf build install log .pytest_cache .ruff_cache .mypy_cache coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
