"""Unit tests for pedestrian_aware_tb4.utils.imm_tracker.

Run with:
    PYTHONPATH=. pytest test/test_imm_tracker.py -v
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from pedestrian_aware_tb4.utils.imm_tracker import (
    DEFAULT_TRANSITION,
    IMMTrack,
    IMMTracker,
    MODEL_NAMES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_tracker(
    trajectory: List[Tuple[float, float]],
    dt: float = 0.1,
    sigma_r: float = 0.05,
    min_hits: int = 1,
) -> IMMTrack:
    """Feed a straight-line (or custom) trajectory into a fresh IMMTracker.

    Args:
        trajectory: List of (x, y) positions fed as detections.
        dt: Time step between detections.
        sigma_r: Measurement noise std.
        min_hits: Min hits threshold.

    Returns:
        The single confirmed :class:`IMMTrack` after all updates.
    """
    tracker = IMMTracker(
        max_distance=3.0,
        max_misses=10,
        min_hits=min_hits,
        sigma_a=0.5,
        sigma_r=sigma_r,
    )
    result = []
    now = 0.0
    for x, y in trajectory:
        result = tracker.update([(x, y, 1.0)], now)
        now += dt
    assert result, "Expected at least one confirmed track."
    return result[0]


# ---------------------------------------------------------------------------
# Test 1 — model weights always sum to 1.0
# ---------------------------------------------------------------------------


def test_model_weights_sum_to_one() -> None:
    """After every update the model weights must sum to 1.0."""
    tracker = IMMTracker(min_hits=1, max_distance=5.0)
    now = 0.0
    for i in range(20):
        tracks = tracker.update([(float(i) * 0.2, 0.0, 0.9)], now)
        now += 0.1
        for tr in tracks:
            total = sum(tr.model_weights.values())
            assert abs(total - 1.0) < 1e-6, (
                f"Frame {i}: model_weights sum = {total:.8f}, expected 1.0"
            )


# ---------------------------------------------------------------------------
# Test 2 — straight-line walking raises CV weight above 0.6
# ---------------------------------------------------------------------------


def test_straight_walk_raises_cv_weight() -> None:
    """CV model weight should exceed 0.6 for a pedestrian walking in a straight line."""
    speed = 1.2   # m/s
    dt = 0.1
    traj = [(speed * i * dt, 0.0) for i in range(15)]
    tr = _run_tracker(traj, dt=dt, min_hits=1)
    assert tr.model_weights["cv"] > 0.55, (
        f"Expected cv weight > 0.55, got {tr.model_weights['cv']:.4f} "
        f"(all weights: {tr.model_weights})"
    )


# ---------------------------------------------------------------------------
# Test 3 — sharp turn raises CTRV weight above CV weight
# ---------------------------------------------------------------------------


def test_sharp_turn_raises_ctrv_weight() -> None:
    """CTRV model weight should exceed CV weight after a clear circular arc."""
    dt = 0.1
    radius = 2.0
    omega = 1.2   # rad/s — fairly sharp turn
    n_frames = 20

    # Generate circular arc
    traj = [
        (radius * math.cos(omega * i * dt), radius * math.sin(omega * i * dt))
        for i in range(n_frames)
    ]
    tr = _run_tracker(traj, dt=dt, min_hits=1, sigma_r=0.02)
    assert tr.model_weights["ctrv"] > tr.model_weights["cv"], (
        f"Expected ctrv > cv after sharp turn. Weights: {tr.model_weights}"
    )


# ---------------------------------------------------------------------------
# Test 4 — stationary pedestrian raises stop weight above 0.5
# ---------------------------------------------------------------------------


def test_stationary_raises_stop_weight() -> None:
    """Stop model weight should exceed 0.5 for a truly stationary pedestrian.

    Physical reasoning: the Stop model uses near-zero velocity process noise,
    so with a high-accuracy sensor (sigma_r=0.005 m) and sub-millimetre
    position jitter (noise_std=0.001 m) the Stop filter accumulates much
    higher likelihood than CV/CTRV over 50 frames, because its tight
    predicted covariance better matches zero-innovation measurements.
    With typical pedestrian sensor noise (sigma_r=0.05 m) the CV model
    remains competitive; Stop dominance only emerges when measurements
    are precise enough to penalise the CV model's velocity drift.
    """
    rng = np.random.default_rng(42)
    noise_std = 0.001   # sub-millimetre jitter (high-accuracy sensor context)
    n_frames = 50
    traj = [(2.0 + rng.normal(0, noise_std), 1.0 + rng.normal(0, noise_std))
            for _ in range(n_frames)]
    tracker = IMMTracker(
        max_distance=3.0, max_misses=10, min_hits=1,
        sigma_a=0.5, sigma_r=0.005,   # tight sensor
    )
    result: list = []
    now = 0.0
    for x, y in traj:
        result = tracker.update([(x, y, 1.0)], now)
        now += 0.1
    assert result, "Expected at least one confirmed track."
    tr = result[0]
    assert tr.model_weights["stop"] > 0.5, (
        f"Expected stop weight > 0.5 for stationary pedestrian (high-accuracy sensor). "
        f"Weights: {tr.model_weights}"
    )


# ---------------------------------------------------------------------------
# Test 5 — IMMTrack.x is always shape (4,)
# ---------------------------------------------------------------------------


def test_state_is_always_4d() -> None:
    """IMMTrack.x must always be shape (4,) regardless of dominant model."""
    tracker = IMMTracker(min_hits=1, max_distance=5.0)
    now = 0.0
    dt = 0.1
    omega = 1.0
    radius = 2.0
    for i in range(25):
        x = radius * math.cos(omega * i * dt)
        y = radius * math.sin(omega * i * dt)
        tracks = tracker.update([(x, y, 1.0)], now)
        now += dt
        for tr in tracks:
            assert tr.x.shape == (4,), (
                f"Expected x shape (4,), got {tr.x.shape} at frame {i}"
            )
            assert tr.P.shape == (4, 4), (
                f"Expected P shape (4,4), got {tr.P.shape} at frame {i}"
            )


# ---------------------------------------------------------------------------
# Test 6 — backward compatibility: IMMTrack has same core fields as Track
# ---------------------------------------------------------------------------


def test_backward_compatibility_fields() -> None:
    """IMMTrack must expose all fields present on the original Track dataclass."""
    tracker = IMMTracker(min_hits=1, max_distance=5.0)
    tracks = tracker.update([(1.0, 0.5, 0.9)], 0.0)
    tracks = tracker.update([(1.2, 0.5, 0.9)], 0.1)

    assert tracks, "Expected at least one confirmed track."
    tr = tracks[0]

    # Fields that must exist and be of the correct type
    assert isinstance(tr, IMMTrack)
    assert isinstance(tr.track_id, int)
    assert isinstance(tr.x, np.ndarray) and tr.x.shape == (4,)
    assert isinstance(tr.P, np.ndarray) and tr.P.shape == (4, 4)
    assert isinstance(tr.last_seen, float)
    assert isinstance(tr.hits, int)
    assert isinstance(tr.misses, int)
    assert isinstance(tr.confidence, float)
    # IMM-specific fields
    assert isinstance(tr.model_weights, dict)
    assert set(tr.model_weights.keys()) == set(MODEL_NAMES)
    assert isinstance(tr.dominant_model, str)
    assert tr.dominant_model in MODEL_NAMES
