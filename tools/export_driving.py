#!/usr/bin/env python3
"""Export compact website replays without changing full-precision metrics."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--lidar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interactions", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    args.out.mkdir(parents=True)
    index = {
        "schema_version": 1,
        "source": "closed_loop_kinematic_bicycle",
        "modes": {},
        "note": "Replay values rounded to four decimals for display; acceptance metrics retain original precision. GT and synthetic surface LiDAR are separate inputs.",
    }
    for mode, directory in [("gt", args.gt), ("lidar", args.lidar)]:
        summary = json.loads((directory / "index.json").read_text())
        if (
            "source_files_sha256" in index
            and index["source_files_sha256"] != summary["source_files_sha256"]
        ):
            raise ValueError("the two runs used different source files")
        index["source_files_sha256"] = summary["source_files_sha256"]
        (args.out / mode).mkdir()
        for entry in summary["scenarios"]:
            source = directory / entry["file"]
            contents = source.read_bytes()
            if hashlib.sha256(contents).hexdigest() != entry["sha256"]:
                raise ValueError("source report hash mismatch")
            report = json.loads(contents)
            public = rounded(report)
            public["metrics"] = report["metrics"]
            public["source_sha256"] = entry["sha256"]
            # Recorded samples already contain all candidate paths and controls.
            data = (
                json.dumps(
                    public, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            ).encode()
            (args.out / mode / entry["file"]).write_bytes(data)
            entry["source_sha256"] = entry["sha256"]
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            entry["file"] = mode + "/" + entry["file"]
        index["modes"][mode] = summary
    (args.out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if args.interactions:
        summary = json.loads((args.interactions / "index.json").read_text())
        if summary["source_files_sha256"] != index["source_files_sha256"]:
            raise ValueError("interaction runs used different source files")
        target = args.out / "interactions"
        target.mkdir()
        for entry in summary["runs"]:
            source = (args.interactions / entry["file"]).read_bytes()
            if hashlib.sha256(source).hexdigest() != entry["sha256"]:
                raise ValueError("interaction source hash mismatch")
            compressed = gzip.compress(source, mtime=0)
            entry["source_sha256"] = entry["sha256"]
            entry["sha256"] = hashlib.sha256(compressed).hexdigest()
            entry["file"] += ".gz"
            (target / entry["file"]).write_bytes(compressed)
        (target / "index.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
