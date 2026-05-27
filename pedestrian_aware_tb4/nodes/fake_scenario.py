"""Fake scenario node for pedestrian_aware_tb4 demo.

Publishes everything the pipeline and risk_filter need so the full
system can be visualised in RViz2 without any real sensor hardware:

Publications
------------
* ``/pedestrian_detections``  std_msgs/String  — JSON detection stream
* ``/odom``                   nav_msgs/Odometry — robot at origin, stationary
* ``/cmd_vel_input``          geometry_msgs/Twist — constant forward drive

Scenario (loops every ~30 s)
-----------------------------
Phase 1  — Pedestrian walks straight across robot's path (crossing intent).
Phase 2  — Pedestrian stops directly ahead (stopping / hard-stop zone).
Phase 3  — Pedestrian approaches robot head-on (approaching intent).
Phase 4  — Pedestrian recedes away at an angle (receding intent).

The scenario is designed so the risk_filter clearly attenuates / stops
``/cmd_vel`` during phases 2 and 3, making the effect visible in RViz2
via the ``/pedestrian_markers`` colour change (green→yellow→red) and
the risk_state_json overlay text.
"""

from __future__ import annotations

import json
import math
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------
# Each waypoint: (x, y, duration_s)  — linear interpolation between them.
# Robot is at origin facing +x.

_WAYPOINTS: List[Tuple[float, float, float]] = [
    # Phase 1: crossing left-to-right across robot's path ~3 m ahead
    (-2.0,  2.5,  0.0),   # start
    ( 2.0,  3.0,  8.0),   # cross at 0.5 m/s

    # Phase 2: pedestrian stops 1.2 m ahead (hard-stop zone)
    ( 2.5,  1.2,  4.0),   # move in front
    ( 1.5,  1.2,  5.0),   # stand still — should trigger hard-stop

    # Phase 3: pedestrian approaches robot head-on
    ( 0.0,  4.0,  0.5),   # step back
    ( 0.0,  0.8,  6.0),   # walk toward robot — approaching

    # Phase 4: pedestrian recedes away diagonally
    ( 0.0,  0.8,  0.5),
    ( 3.0,  5.0,  8.0),   # walk away — receding
]


def _interpolate_waypoints(
    waypoints: List[Tuple[float, float, float]], t: float
) -> Tuple[float, float]:
    """Return (x, y) at time t by linearly interpolating the waypoint list."""
    total = 0.0
    for i in range(len(waypoints) - 1):
        x0, y0, _ = waypoints[i]
        x1, y1, dt = waypoints[i + 1]
        if t <= total + dt:
            if dt < 1e-6:
                return x1, y1
            alpha = (t - total) / dt
            return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)
        total += dt
    return waypoints[-1][0], waypoints[-1][1]


_SCENARIO_DURATION: float = sum(w[2] for w in _WAYPOINTS)


class FakeScenarioNode(Node):
    """Publishes synthetic detections, odometry, and drive command."""

    def __init__(self) -> None:
        super().__init__("fake_scenario")

        # Publishers
        self._det_pub = self.create_publisher(
            String, "/pedestrian_detections", _RELIABLE_QOS
        )
        self._odom_pub = self.create_publisher(
            Odometry, "/odom", _RELIABLE_QOS
        )
        self._cmd_pub = self.create_publisher(
            Twist, "/cmd_vel_input", _RELIABLE_QOS
        )

        # 10 Hz tick — matches pedestrian_pipeline subscription rate
        self._timer = self.create_timer(0.1, self._tick)
        self._t0: float = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info(
            f"FakeScenario started — scenario loops every {_SCENARIO_DURATION:.0f} s"
        )

    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        start_delay = 5.0
        elapsed_raw = now_sec - self._t0

        if elapsed_raw < start_delay:
           elapsed = 0.0
        else:
           elapsed = (elapsed_raw - start_delay) % _SCENARIO_DURATION

        px, py = _interpolate_waypoints(_WAYPOINTS, elapsed)

        # ---- Detections ------------------------------------------------
        det_msg = String()
        det_msg.data = json.dumps({
            "detections": [{"x": round(px, 4), "y": round(py, 4), "confidence": 0.95}]
        })
        self._det_pub.publish(det_msg)

        # ---- Odometry (robot stationary at origin) ---------------------
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation.w = 1.0   # facing +x
        # self._odom_pub.publish(odom)

        # ---- Constant forward drive command ----------------------------
        cmd = Twist()
        cmd.linear.x = 0.4   # 0.4 m/s forward — safety filter will gate this
        self._cmd_pub.publish(cmd)


# ---------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeScenarioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
