"""Unit tests for pedestrian_aware_tb4.utils.intent_estimator.

Run with:
    PYTHONPATH=. pytest test/test_intent_estimator.py -v
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest

from pedestrian_aware_tb4.utils.intent_estimator import (
    INTENT_NAMES,
    IntentBelief,
    IntentEstimator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNIFORM_MODEL_WEIGHTS: Dict[str, float] = {
    "cv": 1.0 / 3.0,
    "ctrv": 1.0 / 3.0,
    "stop": 1.0 / 3.0,
}
_ROBOT_POS = np.array([0.0, 0.0])
_ROBOT_VEL = np.array([0.2, 0.0])   # robot moving in +x at 0.2 m/s


def _feed_estimator(
    estimator: IntentEstimator,
    track_id: int,
    positions: list,
    velocities: list,
    model_weights_seq: list | None = None,
    robot_positions: list | None = None,
    robot_velocities: list | None = None,
) -> IntentBelief:
    """Push a sequence of observations into the estimator and return the last belief.

    Args:
        estimator: :class:`IntentEstimator` instance.
        track_id: Track identifier.
        positions: List of ``[px, py]`` arrays.
        velocities: List of ``[vx, vy]`` arrays.
        model_weights_seq: Optional per-frame model weights dicts.
        robot_positions: Optional per-frame robot positions.
        robot_velocities: Optional per-frame robot velocities.

    Returns:
        Final :class:`IntentBelief`.
    """
    n = len(positions)
    belief = None
    for i in range(n):
        w = (model_weights_seq[i] if model_weights_seq else _UNIFORM_MODEL_WEIGHTS)
        rp = robot_positions[i] if robot_positions else _ROBOT_POS
        rv = robot_velocities[i] if robot_velocities else _ROBOT_VEL
        belief = estimator.update(
            track_id=track_id,
            position=np.asarray(positions[i]),
            velocity=np.asarray(velocities[i]),
            model_weights=w,
            robot_position=np.asarray(rp),
            robot_velocity=np.asarray(rv),
            timestamp=float(i) * 0.1,
        )
    assert belief is not None
    return belief


# ---------------------------------------------------------------------------
# Test 1 — IntentBelief.probs sums to 1.0
# ---------------------------------------------------------------------------


def test_intent_probs_sum_to_one() -> None:
    """Probability distribution over intent states must always sum to 1."""
    est = IntentEstimator(history_length=15)
    for frame in range(10):
        belief = est.update(
            track_id=0,
            position=np.array([3.0 + frame * 0.1, 0.0]),
            velocity=np.array([-0.5, 0.0]),
            model_weights=_UNIFORM_MODEL_WEIGHTS,
            robot_position=_ROBOT_POS,
            robot_velocity=_ROBOT_VEL,
            timestamp=frame * 0.1,
        )
        total = sum(belief.probs.values())
        assert abs(total - 1.0) < 1e-5, (
            f"Frame {frame}: probs sum = {total:.8f}"
        )


# ---------------------------------------------------------------------------
# Test 2 — consistent approach yields dominant == "approaching"
# ---------------------------------------------------------------------------


def test_approaching_dominant_after_8_frames() -> None:
    """Consistent approach toward the robot should yield dominant='approaching'."""
    est = IntentEstimator(history_length=15)
    n = 8
    # Pedestrian starts at (4, 0) moving toward robot at (0, 0) with -0.5 m/s
    positions  = [[4.0 - i * 0.05, 0.0]  for i in range(n)]
    velocities = [[-0.5, 0.0]] * n

    belief = _feed_estimator(est, 1, positions, velocities)
    assert belief.dominant == "approaching", (
        f"Expected 'approaching', got '{belief.dominant}'. probs={belief.probs}"
    )


# ---------------------------------------------------------------------------
# Test 3 — perpendicular crossing velocity yields dominant == "crossing"
# ---------------------------------------------------------------------------


def test_crossing_dominant_after_8_frames() -> None:
    """Velocity perpendicular to the robot→ped axis should yield dominant='crossing'."""
    est = IntentEstimator(history_length=15)
    n = 8
    # Pedestrian moves in +y direction while robot is at origin on +x axis
    positions  = [[3.0, i * 0.05]  for i in range(n)]
    velocities = [[0.0, 0.6]] * n  # purely lateral → crossing

    belief = _feed_estimator(est, 2, positions, velocities)
    assert belief.dominant == "crossing", (
        f"Expected 'crossing', got '{belief.dominant}'. probs={belief.probs}"
    )


# ---------------------------------------------------------------------------
# Test 4 — decelerating pedestrian with high stop-model weight → "stopping"
# ---------------------------------------------------------------------------


def test_stopping_dominant_after_8_frames() -> None:
    """Decelerating speed + high stop-model weight should yield dominant='stopping'."""
    est = IntentEstimator(history_length=15)
    n = 8
    # Velocity decreases from 0.8 to ~0 over 8 frames
    speeds = np.linspace(0.8, 0.05, n)
    positions  = [[2.0, i * 0.04]  for i in range(n)]
    velocities = [[float(s), 0.0]   for s in speeds]

    # Increasing stop-model weight as pedestrian slows
    stop_weights = np.linspace(0.15, 0.80, n)
    model_weights_seq = [
        {"cv": (1.0 - w) * 0.6, "ctrv": (1.0 - w) * 0.4, "stop": float(w)}
        for w in stop_weights
    ]

    belief = _feed_estimator(
        est, 3, positions, velocities, model_weights_seq=model_weights_seq
    )
    assert belief.dominant == "stopping", (
        f"Expected 'stopping', got '{belief.dominant}'. probs={belief.probs}"
    )


# ---------------------------------------------------------------------------
# Test 5 — remove_stale prunes dead-track histories
# ---------------------------------------------------------------------------


def test_remove_stale_prunes_histories() -> None:
    """remove_stale() must delete history for track IDs not in active_ids."""
    est = IntentEstimator(history_length=10)

    # Feed two tracks
    for tid in (10, 20):
        for i in range(5):
            est.update(
                track_id=tid,
                position=np.array([float(i) * 0.1, 0.0]),
                velocity=np.array([0.2, 0.0]),
                model_weights=_UNIFORM_MODEL_WEIGHTS,
                robot_position=_ROBOT_POS,
                robot_velocity=_ROBOT_VEL,
                timestamp=float(i) * 0.1,
            )

    assert 10 in est._histories, "Track 10 should have history before pruning."
    assert 20 in est._histories, "Track 20 should have history before pruning."

    # Declare only track 20 as active
    est.remove_stale({20})

    assert 10 not in est._histories, "Track 10 history should be pruned."
    assert 20 in est._histories, "Track 20 history should survive."


# ---------------------------------------------------------------------------
# Test 6 — reset() clears one track without affecting others
# ---------------------------------------------------------------------------


def test_reset_clears_single_track_only() -> None:
    """reset() must clear history for the specified track and leave others intact."""
    est = IntentEstimator(history_length=10)

    for tid in (100, 200):
        for i in range(4):
            est.update(
                track_id=tid,
                position=np.array([float(i) * 0.1, 0.0]),
                velocity=np.array([0.2, 0.0]),
                model_weights=_UNIFORM_MODEL_WEIGHTS,
                robot_position=_ROBOT_POS,
                robot_velocity=_ROBOT_VEL,
                timestamp=float(i) * 0.1,
            )

    assert len(est._histories[100]) > 0
    assert len(est._histories[200]) > 0

    est.reset(100)

    assert len(est._histories[100]) == 0, "History for track 100 should be empty after reset."
    assert len(est._histories[200]) > 0, "History for track 200 must be unaffected."
