#!/usr/bin/env python3
"""Smoke-test CARLA -> TF/PointCloud2 -> perception -> ROS messages.

Run after colcon build and sourcing install/setup.bash:
    python tools/check_ros_replay.py --dataset carla/data/highway --frames 8
"""
import argparse
import json
from pathlib import Path
from time import monotonic

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from av_perception.carla_replay_node import CarlaReplayNode
from av_perception.perception_node import PerceptionNode
from av_perception_msgs.msg import PredictedObjectArray, TrackedObjectArray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    rclpy.init(args=['--ros-args', '-p', f'dataset:={args.dataset.resolve()}',
                    '-p', f'max_frames:={args.frames}', '-p', 'rate:=2.0'])
    replay = CarlaReplayNode()
    perception = PerceptionNode()
    observer = Node('replay_observer')
    tracks, predictions = [], []
    observer.create_subscription(TrackedObjectArray, '/perception_node/tracks', tracks.append, 10)
    observer.create_subscription(PredictedObjectArray, '/perception_node/predictions', predictions.append, 10)
    executor = SingleThreadedExecutor()
    for node in (replay, perception, observer):
        executor.add_node(node)
    try:
        deadline = monotonic() + 60
        while monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if len(tracks) >= args.frames and len(predictions) >= args.frames:
                break
        assert len(tracks) == args.frames, f'received {len(tracks)}/{args.frames} track messages'
        assert len(predictions) == args.frames, 'missing prediction messages'
        assert all(m.header.frame_id == 'map' for m in tracks + predictions)
        assert any(m.objects for m in tracks), 'no confirmed tracks'
        assert any(m.objects for m in predictions), 'no forecasts'
        stamps = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for m in tracks]
        assert all(abs((b-a)-replay.dataset.dt) < 1e-6 for a,b in zip(stamps,stamps[1:]))
        assert not perception._pending_clouds, 'unprocessed clouds remain'
        result = {'frames': args.frames, 'track_messages': len(tracks),
                  'prediction_messages': len(predictions), 'output_frame': 'map',
                  'sensor_frame': replay.sensor_frame, 'passed': True}
        print(json.dumps(result, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        executor.shutdown()
        for node in (replay, perception, observer):
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
