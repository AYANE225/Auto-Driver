"""Publish a looping synthetic driving scene as PointCloud2 (no CARLA needed).

Lets the full perception graph run end-to-end on any machine: it reuses the
same synthetic generator perception_core ships, so the point clouds are in the
exact ``Frame`` format the real CARLA bridge will produce.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from av_perception import conversions as cvt
from perception_core.io.synthetic import generate_frames, make_default_scene


class SyntheticPublisher(Node):
    def __init__(self) -> None:
        super().__init__("synthetic_publisher")

        self.frame_id = self.declare_parameter("frame_id", "map").value
        topic = self.declare_parameter("topic", "/lidar/points").value
        rate = float(self.declare_parameter("rate", 10.0).value)
        num_frames = int(self.declare_parameter("num_frames", 60).value)
        dt = float(self.declare_parameter("dt", 0.1).value)
        seed = int(self.declare_parameter("seed", 7).value)

        self.frames = generate_frames(
            make_default_scene(), num_frames=num_frames, dt=dt, seed=seed)
        self._idx = 0

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(PointCloud2, topic, qos)
        self.timer = self.create_timer(1.0 / max(rate, 1e-3), self._tick)
        self.get_logger().info(
            f"synthetic_publisher up: {len(self.frames)} frames looping on "
            f"'{topic}' at {rate:.1f} Hz (frame '{self.frame_id}')")

    def _tick(self) -> None:
        frame = self.frames[self._idx % len(self.frames)]
        self._idx += 1

        header = self._header()
        cloud = cvt.numpy_to_pointcloud2(frame.lidar, header)
        self.pub.publish(cloud)

    def _header(self):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.frame_id
        return h


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyntheticPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
