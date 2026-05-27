import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory("pedestrian_aware_tb4")
    params = os.path.join(pkg, "config", "params.yaml")

    return LaunchDescription([
        Node(package="pedestrian_aware_tb4", executable="fake_scenario", output="screen"),
        Node(package="pedestrian_aware_tb4", executable="pedestrian_pipeline", output="screen", parameters=[params]),
        Node(package="pedestrian_aware_tb4", executable="risk_filter", output="screen", parameters=[params]),
        Node(package="pedestrian_aware_tb4", executable="rviz_overlay", output="screen"),
        Node(package="pedestrian_aware_tb4", executable="robot_simulator", output="screen"),
        Node(package="rviz2", executable="rviz2", output="screen"),
    ])
