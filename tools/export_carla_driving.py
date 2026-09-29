#!/usr/bin/env python3
"""Publish a CARLA driving report and a video from its synchronized images."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", choices=["gt", "lidar", "lead_braking"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    args.out.mkdir(parents=True)
    source = (args.dataset / "report.json").read_bytes()
    report = json.loads(source)
    frames = report["frames"]
    step = frames[1]["time_s"] - frames[0]["time_s"]
    if any(not f["image"] or not (args.dataset / f["image"]).exists() for f in frames):
        raise ValueError("run carla/run_driving.py with --images first")
    if any(
        abs(b["time_s"] - a["time_s"] - step) > 1e-5 for a, b in zip(frames, frames[1:])
    ):
        raise ValueError("this exporter requires uniformly sampled images")
    video = args.out / f"driving_carla_{args.name}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(1 / step),
            "-pattern_type",
            "glob",
            "-i",
            str(args.dataset / "images/*.jpg"),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    poster = args.out / f"driving_carla_{args.name}.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(step * len(frames) * 0.6),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(poster),
        ],
        check=True,
    )
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=nb_frames",
                "-of",
                "json",
                str(video),
            ],
            text=True,
        )
    )
    if int(info["streams"][0]["nb_frames"]) != len(frames):
        raise ValueError("video frame count differs from the recorded camera samples")
    report["source_sha256"] = hashlib.sha256(source).hexdigest()
    report["video"] = {
        "file": video.name,
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "duration_s": float(info["format"]["duration"]),
        "frames": len(frames),
        "first_simulation_time_s": frames[0]["time_s"],
    }
    for frame in frames:
        frame["image_sha256"] = hashlib.sha256(
            (args.dataset / frame.pop("image")).read_bytes()
        ).hexdigest()
        frame["video_time_s"] = frame["time_s"] - frames[0]["time_s"]
    (args.out / f"carla_{args.name}.json").write_text(
        json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
