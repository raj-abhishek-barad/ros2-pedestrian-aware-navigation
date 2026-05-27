"""Velocity safety filter node — 20 Hz.

Subscribes to the tracked pedestrian JSON and robot odometry, evaluates
collision risk for every active pedestrian using :func:`compute_risk`, then
attenuates or stops the incoming velocity command.

Subscriptions
-------------
* ``/tracked_pedestrians_json`` — ``std_msgs/String`` JSON.
* ``/odom``                     — ``nav_msgs/Odometry``.
* ``/cmd_vel_input``            — ``geometry_msgs/Twist`` (upstream command).

Publications
------------
* ``/cmd_vel``          — ``geometry_msgs/Twist`` (safe, filtered command).
* ``/risk_state_json``  — ``std_msgs/String`` JSON with per-pedestrian risk.

Intent integration
------------------
When a track entry in the JSON contains an ``"intent"`` field, an
:class:`IntentBelief` is reconstructed and passed to :func:`compute_risk`
so that the risk score is intent-conditioned.  If the field is absent the
node degrades gracefully to the standard risk model.

This file does **not** use any TurtleBot4-specific packages.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

from pedestrian_aware_tb4.utils.intent_estimator import IntentBelief
from pedestrian_aware_tb4.utils.risk_model import RiskOutput, RiskParams, compute_risk

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

# Maximum elapsed time (s) before a track JSON is considered stale.
_STALE_TRACK_AGE: float = 0.5
# Maximum elapsed time (s) before an odometry reading is considered stale.
_STALE_ODOM_AGE: float = 0.5


class RiskFilterNode(Node):
    """20 Hz velocity safety filter with intent-conditioned risk assessment.

    Parameters (ROS 2 parameters, settable via YAML):
        control_hz              — timer frequency (Hz).
        min_clearance           — hard-stop clearance threshold (m).
        max_ttc                 — hard-stop TTC threshold (s).
        risk_weight_clearance   — coefficient on 1/clearance term.
        risk_weight_ttc         — coefficient on 1/ttc term.
        robot_radius            — effective robot radius (m).
        ped_radius              — effective pedestrian radius (m).
        prediction_horizon      — look-ahead for intent prediction (s).
        risk_attenuation_onset  — risk score at which attenuation begins.
        risk_attenuation_full   — risk score at which speed → 0.
    """

    def __init__(self) -> None:
        super().__init__("risk_filter")

        # ----------------------------------------------------------------
        # Declare parameters
        # ----------------------------------------------------------------
        self.declare_parameter("control_hz",             20.0)
        self.declare_parameter("min_clearance",          0.50)
        self.declare_parameter("max_ttc",                2.0)
        self.declare_parameter("risk_weight_clearance",  2.0)
        self.declare_parameter("risk_weight_ttc",        1.0)
        self.declare_parameter("robot_radius",           0.25)
        self.declare_parameter("ped_radius",             0.35)
        self.declare_parameter("prediction_horizon",     1.0)
        self.declare_parameter("risk_attenuation_onset", 3.0)
        self.declare_parameter("risk_attenuation_full",  8.0)

        self._risk_params = RiskParams(
            min_clearance=self.get_parameter("min_clearance").value,
            max_ttc=self.get_parameter("max_ttc").value,
            risk_weight_clearance=self.get_parameter("risk_weight_clearance").value,
            risk_weight_ttc=self.get_parameter("risk_weight_ttc").value,
            robot_radius=self.get_parameter("robot_radius").value,
            ped_radius=self.get_parameter("ped_radius").value,
            prediction_horizon=self.get_parameter("prediction_horizon").value,
        )
        self._onset: float  = float(self.get_parameter("risk_attenuation_onset").value)
        self._full:  float  = float(self.get_parameter("risk_attenuation_full").value)

        # ----------------------------------------------------------------
        # Mutable state
        # ----------------------------------------------------------------
        self._last_tracks: Optional[Dict[str, Any]] = None
        self._last_tracks_stamp: float = 0.0
        self._last_cmd: Twist = Twist()
        self._robot_pos: np.ndarray = np.zeros(2)
        self._robot_vel: np.ndarray = np.zeros(2)
        self._odom_stamp: float = 0.0

        # ----------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------
        self._track_sub = self.create_subscription(
            String,
            "/tracked_pedestrians_json",
            self._tracks_callback,
            _RELIABLE_QOS,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            _SENSOR_QOS,
        )
        self._cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel_input",
            self._cmd_callback,
            _SENSOR_QOS,
        )

        # ----------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", _RELIABLE_QOS)
        self._risk_pub = self.create_publisher(
            String, "/risk_state_json", _RELIABLE_QOS
        )

        # ----------------------------------------------------------------
        # Control timer
        # ----------------------------------------------------------------
        hz: float = float(self.get_parameter("control_hz").value)
        self._timer = self.create_timer(1.0 / hz, self._control_loop)

        self.get_logger().info(f"RiskFilterNode initialised at {hz} Hz.")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _tracks_callback(self, msg: String) -> None:
        """Cache the latest pedestrian track JSON.

        Args:
            msg: ``std_msgs/String`` JSON payload.
        """
        try:
            self._last_tracks = json.loads(msg.data)
            self._last_tracks_stamp = self._now()
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Track JSON decode error: {exc}")

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
        self._odom_stamp = self._now()

    def _cmd_callback(self, msg: Twist) -> None:
        """Cache the latest requested velocity command.

        Args:
            msg: Incoming ``geometry_msgs/Twist``.
        """
        self._last_cmd = msg

    # ------------------------------------------------------------------
    # Control loop (20 Hz)
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """Evaluate risk and publish a safe velocity command."""
        now = self._now()
        stale_reasons: List[str] = []

        # ---- Staleness checks -----------------------------------------
        if self._last_tracks is None:
            stale_reasons.append("no_track_data")
        elif now - self._last_tracks_stamp > _STALE_TRACK_AGE:
            stale_reasons.append("stale_tracks")

        if now - self._odom_stamp > _STALE_ODOM_AGE and self._odom_stamp > 0.0:
            stale_reasons.append("stale_odom")

        if stale_reasons:
            # No data — pass through or hold
            self._cmd_pub.publish(self._last_cmd)
            self._publish_risk_state(
                stamp_sec=now,
                hard_stop=False,
                stale_reasons=stale_reasons,
                max_risk=0.0,
                min_clearance=float("inf"),
                min_ttc=float("inf"),
                cmd=self._last_cmd,
                per_ped=[],
            )
            return

        # ---- Evaluate risk per pedestrian -----------------------------
        tracks: List[Dict[str, Any]] = self._last_tracks.get("tracks", [])
        per_ped_results: List[Dict[str, Any]] = []
        global_hard_stop = False
        max_risk: float = 0.0
        min_clearance: float = float("inf")
        min_ttc: float = float("inf")

        for tr in tracks:
            ped_pos = np.array([
                tr["position"]["x"],
                tr["position"]["y"],
            ])
            ped_vel = np.array([
                tr["velocity"]["x"],
                tr["velocity"]["y"],
            ])
            ped_sigma = float(tr.get("sigma_position", 0.15))

            # Optionally reconstruct IntentBelief
            intent_belief: Optional[IntentBelief] = None
            intent_data = tr.get("intent")
            if intent_data:
                try:
                    intent_belief = IntentBelief(
                        track_id=int(tr.get("id", -1)),
                        probs=dict(intent_data["probs"]),
                        dominant=str(intent_data["dominant"]),
                        entropy=float(intent_data["entropy"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self.get_logger().debug(
                        f"Could not reconstruct IntentBelief for track "
                        f"{tr.get('id', '?')}: {exc}"
                    )

            result: RiskOutput = compute_risk(
                ped_pos=ped_pos,
                ped_vel=ped_vel,
                ped_sigma=ped_sigma,
                robot_pos=self._robot_pos,
                robot_vel=self._robot_vel,
                params=self._risk_params,
                intent_belief=intent_belief,
            )
            result.pedestrian_id = int(tr.get("id", -1))

            if result.hard_stop:
                global_hard_stop = True
            max_risk = max(max_risk, result.risk_score)
            min_clearance = min(min_clearance, result.clearance)
            if math.isfinite(result.ttc):
                min_ttc = min(min_ttc, result.ttc)

            ped_entry: Dict[str, Any] = {
                "id":            result.pedestrian_id,
                "clearance":     result.clearance,
                "closing_speed": result.closing_speed,
                "ttc":           result.ttc if math.isfinite(result.ttc) else 9999.0,
                "risk_score":    result.risk_score,
                "hard_stop":     result.hard_stop,
            }
            if result.dominant_intent is not None:
                ped_entry["dominant_intent"] = result.dominant_intent
            if result.intent_probs is not None:
                ped_entry["intent_probs"] = result.intent_probs
            per_ped_results.append(ped_entry)

        # ---- Compute attenuation factor and safe command --------------
        safe_cmd = self._attenuate_command(
            self._last_cmd, max_risk, global_hard_stop
        )

        # ---- Publish ---------------------------------------------------
        self._cmd_pub.publish(safe_cmd)
        self._publish_risk_state(
            stamp_sec=now,
            hard_stop=global_hard_stop,
            stale_reasons=stale_reasons,
            max_risk=max_risk,
            min_clearance=min_clearance if math.isfinite(min_clearance) else 0.0,
            min_ttc=min_ttc if math.isfinite(min_ttc) else 9999.0,
            cmd=safe_cmd,
            per_ped=per_ped_results,
        )

    # ------------------------------------------------------------------
    # Command attenuation
    # ------------------------------------------------------------------

    def _attenuate_command(
        self,
        cmd: Twist,
        risk: float,
        hard_stop: bool,
    ) -> Twist:
        """Scale the velocity command based on the global risk score.

        Args:
            cmd: Requested velocity command.
            risk: Current maximum risk score across all pedestrians.
            hard_stop: If ``True``, output zero velocity.

        Returns:
            Safe ``geometry_msgs/Twist``.
        """
        out = Twist()
        if hard_stop:
            return out  # zero

        if risk <= self._onset:
            scale = 1.0
        elif risk >= self._full:
            scale = 0.0
        else:
            scale = 1.0 - (risk - self._onset) / (self._full - self._onset)
            scale = max(0.0, min(1.0, scale))

        out.linear.x  = cmd.linear.x  * scale
        out.linear.y  = cmd.linear.y  * scale
        out.angular.z = cmd.angular.z * scale
        return out

    # ------------------------------------------------------------------
    # Risk state publication
    # ------------------------------------------------------------------

    def _publish_risk_state(
        self,
        stamp_sec: float,
        hard_stop: bool,
        stale_reasons: List[str],
        max_risk: float,
        min_clearance: float,
        min_ttc: float,
        cmd: Twist,
        per_ped: List[Dict[str, Any]],
    ) -> None:
        """Serialise and publish the risk state JSON.

        Args:
            stamp_sec: Current timestamp in seconds.
            hard_stop: Global hard-stop flag.
            stale_reasons: List of reason strings for stale data.
            max_risk: Maximum risk score across all pedestrians.
            min_clearance: Minimum clearance across all pedestrians (m).
            min_ttc: Minimum TTC across all pedestrians (s).
            cmd: The command that was published to ``/cmd_vel``.
            per_ped: Per-pedestrian risk entries.
        """
        payload: Dict[str, Any] = {
            "stamp_sec":     stamp_sec,
            "hard_stop":     hard_stop,
            "stale_reasons": stale_reasons,
            "max_risk":      round(max_risk, 4),
            "min_clearance": round(min_clearance, 4),
            "min_ttc":       round(min_ttc, 4),
            "command": {
                "linear_x":  round(cmd.linear.x,  4),
                "angular_z": round(cmd.angular.z, 4),
            },
            "per_pedestrian": per_ped,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._risk_pub.publish(msg)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _now(self) -> float:
        """Current ROS clock time as a float (seconds).

        Returns:
            Seconds since epoch as float.
        """
        return self.get_clock().now().nanoseconds * 1e-9


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Optional[List[str]] = None) -> None:
    """ROS 2 node entry point."""
    rclpy.init(args=args)
    node = RiskFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
