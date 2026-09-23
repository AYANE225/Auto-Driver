"""Bring up the synthetic LiDAR publisher, perception node and (optionally) RViz.

    ros2 launch av_bringup perception.launch.py            # with RViz
    ros2 launch av_bringup perception.launch.py rviz:=false # headless
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("av_bringup")
    default_params = os.path.join(share, "config", "perception.yaml")
    default_rviz = os.path.join(share, "rviz", "perception.rviz")

    rviz_arg = DeclareLaunchArgument("rviz", default_value="true",
                                     description="launch RViz2 with the perception layout")
    params_arg = DeclareLaunchArgument("params_file", default_value=default_params,
                                       description="parameter YAML for perception_node")

    params_file = LaunchConfiguration("params_file")

    synthetic = Node(
        package="av_perception", executable="synthetic_publisher",
        name="synthetic_publisher", output="screen", parameters=[params_file],
    )
    perception = Node(
        package="av_perception", executable="perception_node",
        name="perception_node", output="screen", parameters=[params_file],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", default_rviz], output="log",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([rviz_arg, params_arg, synthetic, perception, rviz])
