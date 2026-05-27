"""RViz2 overlay node.

Subscribes to ``/risk_state_json`` and ``/tracked_pedestrians_json`` and
publishes ``visualization_msgs/MarkerArray`` text labels so that intent,
risk score, and hard-stop status float above each pedestrian cylinder in
RViz2.

Topic published
---------------
``/ped_labels``  — MarkerArray with TEXT_VIEW_FACING markers.

Label format (one line per field)::

    ID:1  crossing
    risk: 4.73
    cv: 0.71

No external dependencies beyond rclpy and visualization_msgs.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

# Colour per intent
_INTENT_COLOUR = {
    "crossing":    (0.2, 0.8, 1.0),   # cyan
    "approaching": (1.0, 0.3, 0.1),   # red-orange
    "stopping":    (1.0, 0.8, 0.0),   # yellow
    "receding":    (0.3, 1.0, 0.3),   # green
}
_DEFAULT_COLOUR = (0.9, 0.9, 0.9)


class RVizOverlayNode(Node):
    """Publishes text labels above pedestrian markers."""

    def __init__(self) -> None:
        super().__init__("rviz_overlay")

        # Cache latest messages from both topics
        self._risk_data: Dict[str, Any] = {}
        self._track_data: Dict[int, Any] = {}

        self._risk_sub = self.create_subscription(
            String, "/risk_state_json", self._risk_cb, _RELIABLE_QOS
        )
        self._track_sub = self.create_subscription(
            String, "/tracked_pedestrians_json", self._track_cb, _RELIABLE_QOS
        )
        self._label_pub = self.create_publisher(
            MarkerArray, "/ped_labels", _RELIABLE_QOS
        )

        # Publish labels at 10 Hz
        self.create_timer(0.1, self._publish_labels)
        self.get_logger().info("RVizOverlay ready — publishing /ped_labels")

    # ------------------------------------------------------------------

    def _risk_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._risk_data = {p["id"]: p for p in data.get("per_pedestrian", [])}
        except Exception:
            pass

    def _track_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._track_data = {t["id"]: t for t in data.get("tracks", [])}
        except Exception:
            pass

    def _publish_labels(self) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        seen_ids = set()

        for tid, track in self._track_data.items():
            seen_ids.add(tid)
            pos = track.get("position", {})
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))

            # Build label text
            lines = [f"ID:{tid}"]

            intent_data = track.get("intent")
            dominant_intent = None
            if intent_data:
                dominant_intent = intent_data.get("dominant", "?")
                entropy = intent_data.get("entropy", 0.0)
                lines.append(f"{dominant_intent}  H={entropy:.2f}")

            imm = track.get("dominant_model", "?")
            w = track.get("imm_weights", {})
            if w:
                wstr = " ".join(
                    f"{k[0]}:{v:.2f}" for k, v in sorted(w.items(), key=lambda kv: -kv[1])
                )
                lines.append(f"[{wstr}]")

            risk_entry = self._risk_data.get(tid, {})
            if risk_entry:
                rs = risk_entry.get("risk_score", 0.0)
                hs = risk_entry.get("hard_stop", False)
                lines.append(f"risk:{rs:.1f}{'  STOP' if hs else ''}")

            label_text = "\n".join(lines)

            # Colour by intent
            r, g, b = _INTENT_COLOUR.get(dominant_intent or "", _DEFAULT_COLOUR)

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "odom"
            m.ns = "ped_labels"
            m.id = tid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 2.3         # float above the cylinder
            m.pose.orientation.w = 1.0
            m.scale.z = 0.22                # text height in metres
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 1.0
            m.text = label_text
            m.lifetime.sec = 1
            markers.markers.append(m)

        # Delete stale markers for tracks that have disappeared
        for tid in list(self._risk_data.keys()):
            if tid not in seen_ids:
                d = Marker()
                d.header.stamp = stamp
                d.header.frame_id = "odom"
                d.ns = "ped_labels"
                d.id = tid
                d.action = Marker.DELETE
                markers.markers.append(d)

        self._label_pub.publish(markers)


# ---------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RVizOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
