#!/usr/bin/env python3
"""Vary interaction conditions in three complex closed-loop scenarios."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from perception_core.simulation.driving import run_scenario, scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assert-success", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src/perception_core/perception_core").rglob("*.py"))
    }
    entries = []
    for mode in ("gt", "lidar"):
        for name, parameter, values in [
            ("lead_braking", "lead_initial_x_m", [16, 20, 24]),
            ("cut_in", "cut_in_duration_s", [1.5, 2.5, 3.5]),
            ("signal_crossing", "green_time_s", [8, 10, 12]),
        ]:
            for value in values:
                scenario = scenarios()[name]
                if name == "lead_braking":
                    scenario.actors[0] = replace(scenario.actors[0], x=value)
                elif name == "cut_in":
                    scenario.actors[0] = replace(
                        scenario.actors[0], change_duration=value
                    )
                    scenario.events[1]["time_s"] = (
                        scenario.actors[0].change_time + value
                    )
                else:
                    scenario.green_time = value
                    scenario.events[0]["time_s"] = value
                report = run_scenario(scenario, detector=mode)
                data = (
                    json.dumps(
                        report,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
                file = f"{name}-{mode}-{value}.json"
                (args.out / file).write_bytes(data)
                entry = {
                    "scenario": name,
                    "title": scenario.title,
                    "detector": mode,
                    "parameter": parameter,
                    "value": value,
                    "seed": 2026,
                    "file": file,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "metrics": report["metrics"],
                }
                entries.append(entry)
                print(name, mode, value, report["metrics"]["success"], flush=True)
    summary = {
        "source": "closed_loop_kinematic_bicycle",
        "source_files_sha256": source_hashes,
        "total": len(entries),
        "passed": sum(e["metrics"]["success"] for e in entries),
        "runs": entries,
        "scope": "One-factor parameter sweeps; fixed seed 2026. Synthetic LiDAR surface points have no ray-cast occlusion. These are selected tests, not a success-rate estimate.",
    }
    (args.out / "index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if args.assert_success and summary["passed"] != summary["total"]:
        raise SystemExit(
            "Some interaction cases failed; inspect index.json and raw reports"
        )


if __name__ == "__main__":
    main()
