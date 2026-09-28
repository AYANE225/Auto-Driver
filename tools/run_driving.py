#!/usr/bin/env python3
"""Run and export deterministic closed-loop planning/control scenarios."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from perception_core.simulation.driving import run_scenario, scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", choices=["all", *scenarios()])
    parser.add_argument("--detector", default="gt", choices=["gt", "lidar"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("outputs/driving"))
    parser.add_argument("--assert-success", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    args.out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src/perception_core/perception_core").rglob("*.py"))
    }
    reports = []
    selected = (
        scenarios()
        if args.scenario == "all"
        else {args.scenario: scenarios()[args.scenario]}
    )
    for name, scenario in selected.items():
        result = run_scenario(scenario, detector=args.detector, seed=args.seed)
        text = (
            json.dumps(
                result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        )
        file = name + ".json"
        (args.out / file).write_text(text)
        reports.append(
            {
                "name": name,
                "title": scenario.title,
                "file": file,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "metrics": result["metrics"],
            }
        )
        print(
            f"{name}: {json.dumps(result['metrics'], ensure_ascii=False)}", flush=True
        )
    index = {
        "source_files_sha256": source_hashes,
        "schema_version": 1,
        "source": "closed_loop_kinematic_bicycle",
        "detector": args.detector,
        "seed": args.seed,
        "python": platform.python_version(),
        "timing_scope": "LocalPlanner.plan + PathController.command; excludes perception, plant, evaluation and rendering",
        "scenarios": reports,
        "passed": sum(r["metrics"]["success"] for r in reports),
        "total": len(reports),
    }
    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if args.assert_success and index["passed"] != index["total"]:
        raise SystemExit(
            "One or more driving scenarios did not meet the acceptance conditions"
        )


if __name__ == "__main__":
    main()
