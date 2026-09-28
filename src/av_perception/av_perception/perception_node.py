"""ROS 2 node wrapping perception_core: LiDAR -> tracks + motion forecasts.

Subscribes to a ``sensor_msgs/PointCloud2`` LiDAR stream, runs the classical
detect -> track -> predict pipeline from :mod:`perception_core`, and publishes
tracked objects, predicted trajectories and RViz markers.
"""
from __future__ import annotations

from collections import deque
from time import monotonic

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import MarkerArray

from av_perception import conversions as cvt
from av_perception_msgs.msg import PredictedObjectArray, TrackedObjectArray
from perception_core.common.types import Frame
from perception_core.pipeline import PerceptionPipeline, PipelineConfig


def _sensor_qos() -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=5,
    )


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")

        self.frame_id = self.declare_parameter("frame_id", "map").value
        input_topic = self.declare_parameter("input_topic", "/lidar/points").value

        cfg = PipelineConfig()
        cfg.lidar.dbscan_eps = self.declare_parameter("dbscan_eps", cfg.lidar.dbscan_eps).value
        cfg.lidar.ground_dist_thresh = self.declare_parameter(
            "ground_dist_thresh", cfg.lidar.ground_dist_thresh).value
        cfg.tracker.iou_threshold = self.declare_parameter(
            "tracker.iou_threshold", cfg.tracker.iou_threshold).value
        cfg.tracker.min_hits = self.declare_parameter("tracker.min_hits", cfg.tracker.min_hits).value
        cfg.tracker.max_age = self.declare_parameter("tracker.max_age", cfg.tracker.max_age).value
        cfg.tracker.motion_model = self.declare_parameter("tracker.motion_model", "imm").value
        cfg.tracker.gating = self.declare_parameter("tracker.gating", False).value
        cfg.predictor.horizon = self.declare_parameter(
            "prediction.horizon", cfg.predictor.horizon).value
        cfg.predictor.step = self.declare_parameter("prediction.step", cfg.predictor.step).value

        self.pipeline = PerceptionPipeline(config=cfg)
        self._frame_counter = 0
        self._last_timestamp = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._pending_clouds = deque(maxlen=5)
        self._tf_timer = self.create_timer(0.01, self._process_pending)

        self.pub_tracks = self.create_publisher(TrackedObjectArray, "~/tracks", 10)
        self.pub_preds = self.create_publisher(PredictedObjectArray, "~/predictions", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "~/markers", 10)
        self.sub = self.create_subscription(
            PointCloud2, input_topic, self._on_cloud, _sensor_qos())

        self.get_logger().info(
            f"perception_node up: subscribing '{input_topic}', frame '{self.frame_id}', "
            f"dbscan_eps={cfg.lidar.dbscan_eps}, min_hits={cfg.tracker.min_hits}")

    def _on_cloud(self, msg: PointCloud2) -> None:
        self._pending_clouds.append((monotonic(), msg))
        self._process_pending()

    def _process_pending(self) -> None:
        # TF and PointCloud2 may arrive in either order. Preserve timestamp order
        # while waiting briefly for the transform, instead of assuming identity.
        while self._pending_clouds:
            queued_at, msg = self._pending_clouds[0]
            if msg.header.frame_id == self.frame_id:
                ego_pose = np.eye(4)
            else:
                try:
                    tf = self._tf_buffer.lookup_transform(
                        self.frame_id, msg.header.frame_id, Time.from_msg(msg.header.stamp))
                    ego_pose = cvt.transform_to_matrix(tf.transform)
                except TransformException:
                    if monotonic() - queued_at < 1.0:
                        return
                    self._pending_clouds.popleft()
                    self.get_logger().warning(
                        f"Dropped cloud: no {self.frame_id} <- {msg.header.frame_id} transform")
                    continue
            self._pending_clouds.popleft()
            self._process_cloud(msg, ego_pose)

    def _process_cloud(self, msg: PointCloud2, ego_pose: np.ndarray) -> None:
        points = cvt.pointcloud2_to_numpy(msg)
        stamp = msg.header.stamp
        timestamp = stamp.sec + stamp.nanosec * 1e-9
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            self.pipeline.reset()  # a replay restarted or time moved backwards
        self._last_timestamp = timestamp
        frame = Frame(
            timestamp=timestamp,
            frame_id=self._frame_counter,
            lidar=points,
            ego_pose=ego_pose,
        )
        self._frame_counter += 1

        output = self.pipeline.process(frame)

        header = Header(stamp=stamp, frame_id=self.frame_id)
        self.pub_tracks.publish(cvt.tracks_to_msg(output.tracks, header))
        self.pub_preds.publish(cvt.predictions_to_msg(output.predictions, header))
        self.pub_markers.publish(
            cvt.build_marker_array(output, self.frame_id, stamp))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
