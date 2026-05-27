"""Pedestrian risk assessment model.

Provides :func:`compute_risk`, which evaluates the collision risk between one
pedestrian track and the robot.  When an :class:`IntentBelief` is supplied the
function computes *intent-conditioned* predicted positions and returns a
probability-weighted risk score.

The module is ROS-free and can be unit-tested with plain pytest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    # Lazy import to avoid circular dependency; only used for type hints.
    from pedestrian_aware_tb4.utils.intent_estimator import IntentBelief

_EPS: float = 1e-9


# ---------------------------------------------------------------------------
# Parameter and output data classes
# ---------------------------------------------------------------------------


@dataclass
class RiskParams:
    """Tunable parameters for :func:`compute_risk`.

    Attributes:
        robot_radius: Effective robot radius for clearance computation (m).
        ped_radius: Effective pedestrian body radius (m).
        min_clearance: Hard-stop clearance threshold (m).
        max_ttc: Hard-stop TTC threshold (s).  TTC < this triggers a stop.
        risk_weight_clearance: Coefficient on the 1/clearance risk term.
        risk_weight_ttc: Coefficient on the 1/ttc risk term.
        prediction_horizon: Look-ahead time for intent-conditioned prediction (s).
        intent_hard_stop_threshold: Probability threshold above which a per-intent
            hard-stop decision is propagated to the fused output.
    """

    robot_radius: float = 0.25
    ped_radius: float = 0.35
    min_clearance: float = 0.50
    max_ttc: float = 2.0
    risk_weight_clearance: float = 2.0
    risk_weight_ttc: float = 1.0
    prediction_horizon: float = 1.0
    intent_hard_stop_threshold: float = 0.15


@dataclass
class RiskOutput:
    """Result of a single :func:`compute_risk` evaluation.

    Attributes:
        clearance: Distance between ped and robot surfaces (m), clipped ≥ 0.
        closing_speed: Rate at which the gap is narrowing (m/s, positive = closing).
        ttc: Time-to-collision in seconds (inf when not closing).
        risk_score: Scalar risk value (higher = more dangerous).
        hard_stop: Whether an emergency stop is warranted.
        pedestrian_id: Originating track ID (–1 if not set by caller).
        intent_probs: Per-intent probabilities if intent belief was provided.
        dominant_intent: Intent state with highest probability (if available).
    """

    clearance: float
    closing_speed: float
    ttc: float
    risk_score: float
    hard_stop: bool
    pedestrian_id: int = -1
    intent_probs: Optional[Dict[str, float]] = None
    dominant_intent: Optional[str] = None


# ---------------------------------------------------------------------------
# Core risk function
# ---------------------------------------------------------------------------


def compute_risk(
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    ped_sigma: float,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    params: RiskParams,
    intent_belief: Optional["IntentBelief"] = None,
) -> RiskOutput:
    """Compute the collision risk between one pedestrian and the robot.

    When *intent_belief* is ``None`` the function computes a standard
    geometry-based risk score.  When provided, it generates intent-conditioned
    predicted positions for each intent state, computes per-intent risk scores,
    and returns a probability-weighted fusion.

    Args:
        ped_pos: Pedestrian position ``[px, py]`` in odom frame.
        ped_vel: Pedestrian velocity ``[vx, vy]`` in odom frame.
        ped_sigma: Positional uncertainty (1-σ, metres) of the pedestrian.
        robot_pos: Robot position ``[px, py]`` in odom frame.
        robot_vel: Robot velocity ``[vx, vy]`` in odom frame.
        params: :class:`RiskParams` instance.
        intent_belief: Optional :class:`IntentBelief` from the intent
            estimator.  Defaults to ``None`` (backward-compatible path).

    Returns:
        A :class:`RiskOutput` with the computed risk metrics.
    """
    # ------------------------------------------------------------------
    # If intent belief is available, compute intent-conditioned risk.
    # ------------------------------------------------------------------
    if intent_belief is not None:
        return _intent_conditioned_risk(
            ped_pos, ped_vel, ped_sigma, robot_pos, robot_vel, params, intent_belief
        )

    # ------------------------------------------------------------------
    # Standard (non-intent) risk computation
    # ------------------------------------------------------------------
    return _base_risk(ped_pos, ped_vel, ped_sigma, robot_pos, robot_vel, params)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _base_risk(
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    ped_sigma: float,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    params: RiskParams,
) -> RiskOutput:
    """Compute geometry-based risk without intent information.

    Args:
        ped_pos: Pedestrian position ``[px, py]``.
        ped_vel: Pedestrian velocity ``[vx, vy]``.
        ped_sigma: Positional 1-σ uncertainty (metres).
        robot_pos: Robot position ``[px, py]``.
        robot_vel: Robot velocity ``[vx, vy]``.
        params: Risk parameters.

    Returns:
        :class:`RiskOutput` without intent fields.
    """
    rel_pos = ped_pos - robot_pos
    centre_dist = float(np.linalg.norm(rel_pos))
    body_margin = params.robot_radius + params.ped_radius + ped_sigma
    clearance = max(centre_dist - body_margin, 0.0)

    # Closing speed: component of relative velocity pointing from robot to ped
    if centre_dist < _EPS:
        closing_speed = 0.0
    else:
        rel_dir = rel_pos / centre_dist          # unit vector robot → ped
        rel_vel = ped_vel - robot_vel
        # Negative projection means approaching
        closing_speed = -float(rel_vel @ rel_dir)

    # Time to collision
    if closing_speed > _EPS:
        ttc = max(clearance / closing_speed, 0.0)
    else:
        ttc = float("inf")

    # Risk score: weighted inverse of clearance and TTC
    risk_clearance = params.risk_weight_clearance / max(clearance, 0.05)
    risk_ttc = (
        params.risk_weight_ttc / max(ttc, 0.1) if math.isfinite(ttc) else 0.0
    )
    risk_score = risk_clearance + risk_ttc

    hard_stop = clearance < params.min_clearance or (
        math.isfinite(ttc) and ttc < params.max_ttc and closing_speed > 0.0
    )

    return RiskOutput(
        clearance=clearance,
        closing_speed=closing_speed,
        ttc=ttc,
        risk_score=risk_score,
        hard_stop=hard_stop,
    )


def _intent_conditioned_risk(
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    ped_sigma: float,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    params: RiskParams,
    intent_belief: "IntentBelief",
) -> RiskOutput:
    """Compute probability-weighted risk conditioned on intent beliefs.

    For each intent state a predicted pedestrian position is computed at
    ``t + params.prediction_horizon`` seconds, then :func:`_base_risk` is
    evaluated independently (no recursion via ``intent_belief``).  The final
    risk score is the probability-weighted sum of per-intent risk scores.

    Args:
        ped_pos: Current pedestrian position ``[px, py]``.
        ped_vel: Current pedestrian velocity ``[vx, vy]``.
        ped_sigma: Positional uncertainty (metres).
        robot_pos: Robot position ``[px, py]``.
        robot_vel: Robot velocity ``[vx, vy]``.
        params: Risk parameters including ``prediction_horizon``.
        intent_belief: Intent posterior from :class:`IntentEstimator`.

    Returns:
        :class:`RiskOutput` with ``intent_probs`` and ``dominant_intent`` set.
    """
    t = params.prediction_horizon
    probs = intent_belief.probs

    # Intent-conditioned position predictions
    _pred_map: Dict[str, np.ndarray] = {
        "crossing":    ped_pos + ped_vel * t,
        "approaching": ped_pos + ped_vel * t,
        "stopping":    ped_pos + ped_vel * (t * 0.3),  # 70% deceleration
        "receding":    ped_pos + ped_vel * t,
    }

    # Per-intent base risk (no intent_belief arg → avoids infinite recursion)
    per_intent: Dict[str, RiskOutput] = {}
    for intent, pred_pos in _pred_map.items():
        per_intent[intent] = _base_risk(
            pred_pos, ped_vel, ped_sigma, robot_pos, robot_vel, params
        )

    # Probability-weighted risk score
    intent_names = ["crossing", "approaching", "stopping", "receding"]
    intent_risk = float(
        sum(probs[i] * per_intent[i].risk_score for i in intent_names)
    )

    # Weighted clearance and closing speed (from base calculation at current pos)
    base = _base_risk(ped_pos, ped_vel, ped_sigma, robot_pos, robot_vel, params)

    # Hard stop: fire if any intent with P > threshold produces a hard stop
    hard_stop = base.hard_stop or any(
        probs[i] > params.intent_hard_stop_threshold and per_intent[i].hard_stop
        for i in intent_names
    )

    dominant_intent = max(probs, key=lambda k: probs[k])

    return RiskOutput(
        clearance=base.clearance,
        closing_speed=base.closing_speed,
        ttc=base.ttc,
        risk_score=intent_risk,
        hard_stop=hard_stop,
        intent_probs=dict(probs),
        dominant_intent=dominant_intent,
    )
