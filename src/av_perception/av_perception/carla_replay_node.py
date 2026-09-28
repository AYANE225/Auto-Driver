"""Publish a CARLA recording as LiDAR PointCloud2 plus world <- LiDAR TF."""
from __future__ import annotations

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from av_perception import conversions as cvt
from perception_core.io.carla_dataset import CarlaDataset


class CarlaReplayNode(Node):
    def __init__(self):
        super().__init__('carla_replay')
        dataset = self.declare_parameter('dataset', '').value
        if not dataset:
            raise ValueError('set the dataset parameter to a CARLA recording directory')
        self.dataset = CarlaDataset(dataset, count_points=False)
        self.world_frame = self.declare_parameter('world_frame', 'map').value
        self.sensor_frame = self.declare_parameter('sensor_frame', 'lidar').value
        rate = float(self.declare_parameter('rate', 2.0).value)
        if rate <= 0 or self.world_frame == self.sensor_frame:
            raise ValueError('rate must be positive and world/sensor frames must differ')
        self.max_frames = int(self.declare_parameter('max_frames', 0).value)
        topic = self.declare_parameter('topic', '/lidar/points').value
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(PointCloud2, topic, qos)
        self.broadcaster = TransformBroadcaster(self)
        self.index = 0
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f'Replaying {len(self.dataset)} frames from {dataset} at {rate} Hz')

    def _tick(self):
        limit = min(len(self.dataset), self.max_frames) if self.max_frames > 0 else len(self.dataset)
        if self.index >= limit:
            self.timer.cancel()
            self.get_logger().info(f'Replay complete: {self.index} frames')
            return
        frame = self.dataset.read_frame(self.index)
        self.index += 1
        if frame.ego_pose is None:
            raise ValueError('CARLA replay requires a stored world <- LiDAR pose')
        nanoseconds = int(round(frame.timestamp * 1e9))
        stamp = Time(sec=nanoseconds // 10**9, nanosec=nanoseconds % 10**9)
        transform = TransformStamped()
        transform.header = Header(stamp=stamp, frame_id=self.world_frame)
        transform.child_frame_id = self.sensor_frame
        t = transform.transform.translation
        t.x, t.y, t.z = (float(v) for v in frame.ego_pose[:3, 3])
        q = Rotation.from_matrix(frame.ego_pose[:3, :3]).as_quat()
        r = transform.transform.rotation
        r.x, r.y, r.z, r.w = (float(v) for v in q)
        self.broadcaster.sendTransform(transform)
        header = Header(stamp=stamp, frame_id=self.sensor_frame)
        self.publisher.publish(cvt.numpy_to_pointcloud2(frame.lidar, header))


def main(args=None):
    rclpy.init(args=args)
    node = CarlaReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
