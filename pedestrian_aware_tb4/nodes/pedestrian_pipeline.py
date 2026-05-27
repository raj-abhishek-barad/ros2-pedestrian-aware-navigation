"""Pedestrian perception and tracking pipeline node.

Subscribes to incoming pedestrian detections (position + confidence in the
odom frame), runs the IMM tracker and intent estimator, then publishes:

* ``/tracked_pedestrians_json`` — JSON string with full track + IMM + intent data.
* ``/pedestrian_markers``       — RViz2 ``MarkerArray`` (unchanged format).

Detection input
---------------
The node subscribes to ``/pedestrian_detections`` which carries a
``std_msgs/String`` with a JSON payload::

    {"detections": [{"x": 1.2, "y": 0.5, "confidence": 0.9}, ...]}

In test mode the ``fake_pedestrian`` node publishes on this topic.  In
hardware mode an upstream YOLO/OAK-D perception node publishes here.  The
pipeline is agnostic to the source.

Robot position is obtained from ``/odom`` (``nav_msgs/Odometry``).

Notes
-----
* This file does **not** import any TurtleBot4-specific packages so it runs on
  any ROS 2 Jazzy system.
* ``fake_pedestrian.py`` continues to use ``KalmanTracker`` and is unaffected
  by these changes.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import ColorRGBA, Header, String
from visualization_msgs.msg import Marker, MarkerArray

from pedestrian_aware_tb4.utils.imm_tracker import IMMTrack, IMMTracker
from pedestrian_aware_tb4.utils.intent_estimator import IntentBelief, IntentEstimator

# QoS profile suitable for sensor data
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class PedestrianPipelineNode(Node):
    """ROS 2 node that tracks pedestrians and estimates their intent.

    Parameters (ROS 2 parameters, settable via YAML config):
        imm_max_distance    — gate distance for track assignment (m).
        imm_max_misses      — misses before a track is deleted.
        imm_min_hits        — hits before a track is published.
        imm_sigma_a         — acceleration noise std (m/s²).
        imm_sigma_r         — measurement noise std (m).
        imm_sigma_yaw_dd    — CTRV yaw-rate noise std (rad/s²).
        intent_history      — frame history for intent HMM.
        odom_frame          — frame_id written into track JSON.
    """

    def __init__(self) -> None:
        super().__init__("pedestrian_pipeline")

        # ----------------------------------------------------------------
        # Declare ROS 2 parameters
        # ----------------------------------------------------------------
        self.declare_parameter("imm_max_distance", 1.5)
        self.declare_parameter("imm_max_misses",   5)
        self.declare_parameter("imm_min_hits",     2)
        self.declare_parameter("imm_sigma_a",      0.5)
        self.declare_parameter("imm_sigma_r",      0.15)
        self.declare_parameter("imm_sigma_yaw_dd", 0.3)
        self.declare_parameter("intent_history",   15)
        self.declare_parameter("odom_frame",       "odom")

        p = self.get_parameters([
            "imm_max_distance", "imm_max_misses", "imm_min_hits",
            "imm_sigma_a", "imm_sigma_r", "imm_sigma_yaw_dd",
            "intent_history", "odom_frame",
        ])
        params = {param.name: param.value for param in p}

        # ----------------------------------------------------------------
        # Trackers
        # ----------------------------------------------------------------
        self._tracker = IMMTracker(
            max_distance=float(params["imm_max_distance"]),
            max_misses=int(params["imm_max_misses"]),
            min_hits=int(params["imm_min_hits"]),
            sigma_a=float(params["imm_sigma_a"]),
            sigma_r=float(params["imm_sigma_r"]),
            sigma_yaw_dd=float(params["imm_sigma_yaw_dd"]),
        )
        self._intent_estimator = IntentEstimator(
            history_length=int(params["intent_history"])
        )
        self._frame_id: str = str(params["odom_frame"])

        # ----------------------------------------------------------------
        # Robot state (updated from /odom)
        # ----------------------------------------------------------------
        self._robot_pos: np.ndarray = np.zeros(2)
        self._robot_vel: np.ndarray = np.zeros(2)

        # ----------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------
        self._det_sub = self.create_subscription(
            String,
            "/pedestrian_detections",
            self._detection_callback,
            _SENSOR_QOS,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            _SENSOR_QOS,
        )

        # ----------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------
        self._track_pub = self.create_publisher(
            String, "/tracked_pedestrians_json", _RELIABLE_QOS
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/pedestrian_markers", _RELIABLE_QOS
        )

        self.get_logger().info("PedestrianPipelineNode initialised (IMMTracker + IntentEstimator).")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _odom_callback(self, msg: Odometry) -> None:
        """Cache robot pose and velocity from odometry.

        Args:
            msg: ``nav_msgs/Odometry`` message.
        """
        self._robot_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        self._robot_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
        ])

    def _detection_callback(self, msg: String) -> None:
        """Process incoming detections, update tracker, publish results.

        Args:
            msg: ``std_msgs/String`` containing JSON::

                {"detections": [{"x": …, "y": …, "confidence": …}, …]}
        """
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Detection JSON decode error: {exc}")
            return

        raw_dets: List[Dict[str, Any]] = payload.get("detections", [])
        detections = [
            (float(d["x"]), float(d["y"]), float(d.get("confidence", 1.0)))
            for d in raw_dets
        ]

        now_stamp = self.get_clock().now()
        now_sec = now_stamp.nanoseconds * 1e-9

        tracks = self._tracker.update(detections, now_sec)
        self._publish_tracks(tracks, now_sec)

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------

    def _publish_tracks(self, tracks: List[IMMTrack], stamp_sec: float) -> None:
        """Build and publish JSON and RViz2 markers for all confirmed tracks.

        Also calls :meth:`IntentEstimator.remove_stale` to prune dead
        track histories.

        Args:
            tracks: Confirmed :class:`IMMTrack` list from the IMM tracker.
            stamp_sec: Current time in seconds.
        """
        track_jsons = []
        markers = MarkerArray()

        for tr in tracks:
            # -- Intent estimation -------------------------------------------
            try:
                belief: Optional[IntentBelief] = self._intent_estimator.update(
                    track_id=tr.track_id,
                    position=tr.x[:2],
                    velocity=tr.x[2:4],
                    model_weights=tr.model_weights,
                    robot_position=self._robot_pos,
                    robot_velocity=self._robot_vel,
                    timestamp=stamp_sec,
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Intent estimator error on track {tr.track_id}: {exc}")
                belief = None

            # -- Covariance (xy 2×2 block of 4×4 P) -------------------------
            P_xy = tr.P[:2, :2]
            sigma_pos = float(math.sqrt(max(0.0, (P_xy[0, 0] + P_xy[1, 1]) / 2.0)))

            # -- Build track JSON entry -------------------------------------
            entry: Dict[str, Any] = {
                "id":             tr.track_id,
                "frame_id":       self._frame_id,
                "position":       {"x": float(tr.x[0]), "y": float(tr.x[1]), "z": 0.0},
                "velocity":       {"x": float(tr.x[2]), "y": float(tr.x[3]), "z": 0.0},
                "confidence":     float(tr.confidence),
                "sigma_position": sigma_pos,
                "covariance_xy":  [float(P_xy[0, 0]), float(P_xy[0, 1]),
                                   float(P_xy[1, 0]), float(P_xy[1, 1])],
                "hits":           tr.hits,
                "misses":         tr.misses,
                # IMM-specific fields
                "imm_weights":    tr.model_weights,
                "dominant_model": tr.dominant_model,
            }

            if belief is not None:
                entry["intent"] = {
                    "probs":    belief.probs,
                    "dominant": belief.dominant,
                    "entropy":  float(belief.entropy),
                }

            track_jsons.append(entry)

            # -- RViz2 marker (sphere at pedestrian position) ---------------
            m = Marker()
            m.header.frame_id = self._frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "pedestrians"
            m.id = tr.track_id
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(tr.x[0])
            m.pose.position.y = float(tr.x[1])
            m.pose.position.z = 0.9          # visual centre height
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 1.8
            # Colour encodes dominant model: cv=green, ctrv=yellow, stop=red
            m.color = _model_colour(tr.dominant_model)
            m.lifetime.sec = 1
            markers.markers.append(m)

            # -- Velocity arrow marker --------------------------------------
            va = Marker()
            va.header = m.header
            va.ns = "ped_vel"
            va.id = tr.track_id + 10000
            va.type = Marker.ARROW
            va.action = Marker.ADD
            va.points.append(Point(x=float(tr.x[0]), y=float(tr.x[1]), z=0.9))
            va.points.append(
                Point(x=float(tr.x[0] + tr.x[2]), y=float(tr.x[1] + tr.x[3]), z=0.9)
            )
            va.scale.x = 0.05; va.scale.y = 0.10; va.scale.z = 0.10
            va.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)
            va.lifetime.sec = 1
            markers.markers.append(va)

        # -- Publish JSON --------------------------------------------------
        json_msg = String()
        json_msg.data = json.dumps({
            "stamp_sec": stamp_sec,
            "frame_id":  self._frame_id,
            "tracks":    track_jsons,
        })
        self._track_pub.publish(json_msg)

        # -- Publish markers ----------------------------------------------
        self._marker_pub.publish(markers)

        # -- Prune stale intent histories ---------------------------------
        self._intent_estimator.remove_stale({tr.track_id for tr in tracks})


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------


def _model_colour(dominant_model: str) -> ColorRGBA:
    """Return a RViz2 colour for the dominant IMM model.

    Args:
        dominant_model: One of ``"cv"``, ``"ctrv"``, ``"stop"``.

    Returns:
        ``ColorRGBA`` message.
    """
    palette = {
        "cv":   ColorRGBA(r=0.2, g=0.9, b=0.2, a=0.8),  # green
        "ctrv": ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.8),  # yellow
        "stop": ColorRGBA(r=0.9, g=0.2, b=0.2, a=0.8),  # red
    }
    return palette.get(dominant_model, ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Optional[List[str]] = None) -> None:
    """ROS 2 node entry point."""
    rclpy.init(args=args)
    node = PedestrianPipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
