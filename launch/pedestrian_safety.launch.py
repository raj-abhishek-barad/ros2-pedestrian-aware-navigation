"""Launch file for pedestrian_aware_tb4.

Starts:
  1. pedestrian_pipeline  — IMM tracker + intent estimator
  2. risk_filter          — intent-conditioned velocity gating

Both nodes are configured from config/params.yaml.

Usage
-----
  ros2 launch pedestrian_aware_tb4 pedestrian_safety.launch.py

Override a parameter at launch time:
  ros2 launch pedestrian_aware_tb4 pedestrian_safety.launch.py \\
      imm_sigma_r:=0.10 control_hz:=30.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("pedestrian_aware_tb4")

    # Allow per-launch YAML override
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution([pkg_share, "config", "params.yaml"]),
        description="Full path to the ROS 2 parameters file.",
    )
    params_file = LaunchConfiguration("params_file")

    pipeline_node = Node(
        package="pedestrian_aware_tb4",
        executable="pedestrian_pipeline",
        name="pedestrian_pipeline",
        output="screen",
        parameters=[params_file],
    )

    risk_filter_node = Node(
        package="pedestrian_aware_tb4",
        executable="risk_filter",
        name="risk_filter",
        output="screen",
        parameters=[params_file],
    )

    return LaunchDescription([
        params_file_arg,
        pipeline_node,
        risk_filter_node,
    ])
