"""Full demo launch: fake scenario + full pipeline + RViz2.

Starts:
  1. fake_scenario       — moving pedestrian + odom + cmd_vel_input
  2. pedestrian_pipeline — IMM tracker + intent estimator
  3. risk_filter         — intent-conditioned velocity gating
  4. rviz_overlay        — text labels above pedestrians
  5. rviz2               — pre-configured view

Usage
-----
  ros2 launch pedestrian_aware_tb4 demo.launch.py

Watch in RViz2:
  - Cylinder colour:  green = CV (straight walk)
                      yellow = CTRV (turning)
                      red    = Stop (stationary)
  - Floating label:   intent | risk score | IMM weights
  - Label colour:     cyan=crossing  red=approaching
                      yellow=stopping  green=receding
  - When risk ≥ 3:    /cmd_vel linear.x starts dropping
  - When risk ≥ 8 or hard_stop: /cmd_vel = 0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("pedestrian_aware_tb4")
    params_yaml = os.path.join(pkg, "config", "params.yaml")
    rviz_cfg    = os.path.join(pkg, "config", "demo.rviz")

    fake_scenario = Node(
        package="pedestrian_aware_tb4",
        executable="fake_scenario",
        name="fake_scenario",
        output="screen",
    )

    pipeline = Node(
        package="pedestrian_aware_tb4",
        executable="pedestrian_pipeline",
        name="pedestrian_pipeline",
        output="screen",
        parameters=[params_yaml],
    )

    risk_filter = Node(
        package="pedestrian_aware_tb4",
        executable="risk_filter",
        name="risk_filter",
        output="screen",
        parameters=[params_yaml],
    )

    overlay = Node(
        package="pedestrian_aware_tb4",
        executable="rviz_overlay",
        name="rviz_overlay",
        output="screen",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        output="screen",
    )

    return LaunchDescription([
        fake_scenario,
        pipeline,
        risk_filter,
        overlay,
        rviz2,
    ])
