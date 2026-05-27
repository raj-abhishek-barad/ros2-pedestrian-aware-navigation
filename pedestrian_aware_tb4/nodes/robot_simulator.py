#!/usr/bin/env python3
"""Simulated robot node for the pedestrian_aware_tb4 demo.

Integrates the filtered velocity command (``/cmd_vel``) to move a robot
marker back and forth in the odom frame, and broadcasts the
``odom → base_link`` TF so RViz2 can display the robot pose correctly.

Topics
------
Subscribed:
  ``/cmd_vel``      geometry_msgs/Twist — safety-filtered velocity from risk_filter

Published:
  ``/odom``         nav_msgs/Odometry   — integrated robot pose
  ``/robot_marker`` visualization_msgs/Marker — body cylinder
  ``/robot_heading`` visualization_msgs/Marker — heading arrow

TF:
  ``odom → base_link``

Notes
-----
* The robot bounces between x_min and x_max at y = 2.2 (fixed).
* ``v`` is taken as signed: positive = forward along current heading.
  The risk_filter only attenuates to 0 and never reverses, so v ≥ 0 in
  practice — but the sign is preserved so reversing behaviour is possible
  without further changes.
* ``self.direction`` has been removed; heading is encoded entirely in
  ``self.yaw``.
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

try:
    import tf2_ros
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class RobotSimulator(Node):
    def __init__(self) -> None:
        super().__init__("robot_simulator")

        # Initial pose — starts at left edge, heading right (+x)
        self.x: float = -4.0
        self.y: float = 2.2
        self.yaw: float = 0.0        # 0 = facing +x, π = facing -x

        # Bounce limits
        self._x_min: float = -4.0
        self._x_max: float =  4.0

        # Velocity from /cmd_vel (signed)
        self._v: float = 0.0
        self._w: float = 0.0
        self._last_time = self.get_clock().now()

        # TF broadcaster (optional — skipped if tf2_ros unavailable)
        self._tf_broadcaster = None
        if _TF_AVAILABLE:
            self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        else:
            self.get_logger().warn(
                "tf2_ros not found — odom→base_link TF will NOT be broadcast."
            )

        # Subscribers / publishers
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self._odom_pub    = self.create_publisher(Odometry, "/odom", 10)
        self._body_pub    = self.create_publisher(Marker, "/robot_marker", 10)
        self._heading_pub = self.create_publisher(Marker, "/robot_heading", 10)

        self.create_timer(0.05, self._step)   # 20 Hz
        self.get_logger().info("RobotSimulator started (TF: %s).", str(_TF_AVAILABLE))

    # ------------------------------------------------------------------

    def _cmd_cb(self, msg: Twist) -> None:
        self._v = msg.linear.x    # signed: risk_filter attenuates toward 0
        self._w = msg.angular.z

    def _step(self) -> None:
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now

        # --- Integrate pose --------------------------------------------
        self.x   += self._v * math.cos(self.yaw) * dt
        self.y   += self._v * math.sin(self.yaw) * dt
        self.yaw += self._w * dt

        # --- Bounce logic: flip heading at boundaries ------------------
        if self.x >= self._x_max:
            self.x   = self._x_max
            self.yaw = math.pi    # face -x
        elif self.x <= self._x_min:
            self.x   = self._x_min
            self.yaw = 0.0        # face +x

        # --- Publish ---------------------------------------------------
        self._publish_odom(now)
        self._publish_markers(now)
        if self._tf_broadcaster is not None:
            self._publish_tf(now)

    # ------------------------------------------------------------------

    def _publish_odom(self, now) -> None:
        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id  = "base_link"
        odom.pose.pose.position.x    = self.x
        odom.pose.pose.position.y    = self.y
        odom.pose.pose.orientation.z = math.sin(self.yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.yaw / 2.0)
        odom.twist.twist.linear.x    = self._v
        odom.twist.twist.angular.z   = self._w
        self._odom_pub.publish(odom)

    def _publish_markers(self, now) -> None:
        stamp = now.to_msg()

        # Body — blue cylinder
        body = Marker()
        body.header.stamp    = stamp
        body.header.frame_id = "odom"
        body.ns   = "robot"
        body.id   = 0
        body.type = Marker.CYLINDER
        body.action = Marker.ADD
        body.pose.position.x    = self.x
        body.pose.position.y    = self.y
        body.pose.position.z    = 0.25
        body.pose.orientation.z = math.sin(self.yaw / 2.0)
        body.pose.orientation.w = math.cos(self.yaw / 2.0)
        body.scale.x = 0.6
        body.scale.y = 0.6
        body.scale.z = 0.5
        body.color   = ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.9)
        self._body_pub.publish(body)

        # Heading arrow — white, points in yaw direction
        arrow = Marker()
        arrow.header.stamp    = stamp
        arrow.header.frame_id = "odom"
        arrow.ns   = "robot_heading"
        arrow.id   = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x    = self.x
        arrow.pose.position.y    = self.y
        arrow.pose.position.z    = 0.5
        arrow.pose.orientation.z = math.sin(self.yaw / 2.0)
        arrow.pose.orientation.w = math.cos(self.yaw / 2.0)
        arrow.scale.x = 0.8    # arrow length
        arrow.scale.y = 0.12   # shaft diameter
        arrow.scale.z = 0.12
        arrow.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
        self._heading_pub.publish(arrow)

    def _publish_tf(self, now) -> None:
        t = TransformStamped()
        t.header.stamp    = now.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id  = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.z    = math.sin(self.yaw / 2.0)
        t.transform.rotation.w    = math.cos(self.yaw / 2.0)
        self._tf_broadcaster.sendTransform(t)


# ---------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
