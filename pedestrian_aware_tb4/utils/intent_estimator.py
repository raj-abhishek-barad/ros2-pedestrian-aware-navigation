"""Pedestrian intent estimation via a 4-state Hidden Markov Model (HMM).

Classifies the behavioural intent of each tracked pedestrian into one of four
states: *crossing*, *approaching*, *stopping*, *receding*.

The HMM forward algorithm is executed in log-probability space over a sliding
history buffer of observation feature vectors.  All inner operations use numpy
for efficient batch computation.

Performance target: ``IntentEstimator.update()`` < 1 ms per track on a
Raspberry Pi 4 (history of 15 frames × 4 states ≈ 240 scalar multiplications).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENT_NAMES: Tuple[str, ...] = ("crossing", "approaching", "stopping", "receding")
FEATURE_NAMES: Tuple[str, ...] = (
    "heading_to_robot",
    "lateral_speed",
    "speed_trend",
    "stop_model_weight",
    "closing_rate",
)

#: HMM state transition matrix (rows = from-state, cols = to-state).
DEFAULT_TRANSITION: np.ndarray = np.array(
    [
        [0.85, 0.05, 0.05, 0.05],  # from crossing
        [0.05, 0.80, 0.10, 0.05],  # from approaching
        [0.05, 0.05, 0.85, 0.05],  # from stopping
        [0.05, 0.05, 0.05, 0.85],  # from receding
    ],
    dtype=np.float64,
)

# Emission parameters: {state: {feature: (mean, std)}}
# Gaussian p(feature | state).  Hand-tuned for pedestrian kinematics.
EMISSION_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "crossing": {
        "heading_to_robot":  (-0.1, 0.5),
        "lateral_speed":     ( 0.8, 0.4),
        "speed_trend":       ( 0.0, 0.5),
        "stop_model_weight": ( 0.1, 0.1),
        "closing_rate":      (-0.2, 0.4),
    },
    "approaching": {
        "heading_to_robot":  ( 0.85, 0.2),
        "lateral_speed":     ( 0.1,  0.3),
        "speed_trend":       ( 0.0,  0.4),
        "stop_model_weight": ( 0.15, 0.1),
        "closing_rate":      (-0.6,  0.3),
    },
    "stopping": {
        "heading_to_robot":  ( 0.0,  0.5),
        "lateral_speed":     ( 0.1,  0.3),
        "speed_trend":       (-0.8,  0.3),
        "stop_model_weight": ( 0.65, 0.2),
        "closing_rate":      ( 0.1,  0.3),
    },
    "receding": {
        "heading_to_robot":  (-0.8, 0.3),
        "lateral_speed":     ( 0.1, 0.3),
        "speed_trend":       ( 0.0, 0.4),
        "stop_model_weight": ( 0.1, 0.1),
        "closing_rate":      ( 0.5, 0.3),
    },
}

_EPS: float = 1e-9
_N_STATES: int = len(INTENT_NAMES)
_N_FEATURES: int = len(FEATURE_NAMES)

# ---------------------------------------------------------------------------
# Pre-computed emission parameter arrays for vectorised likelihood
# ---------------------------------------------------------------------------
# Shape (n_states, n_features)
_EMIT_MEAN: np.ndarray = np.array(
    [
        [EMISSION_PARAMS[s][f][0] for f in FEATURE_NAMES]
        for s in INTENT_NAMES
    ]
)
_EMIT_STD: np.ndarray = np.array(
    [
        [EMISSION_PARAMS[s][f][1] for f in FEATURE_NAMES]
        for s in INTENT_NAMES
    ]
)

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class IntentBelief:
    """Posterior belief over pedestrian intent states.

    Attributes:
        track_id: Identifier matching the source :class:`~imm_tracker.IMMTrack`.
        probs: Normalised probability per intent state, e.g.
            ``{"crossing": 0.7, "approaching": 0.1, ...}``.
        dominant: Name of the highest-probability intent state.
        entropy: Shannon entropy of *probs* in nats (high ≈ uncertain).
    """

    track_id: int
    probs: Dict[str, float]
    dominant: str
    entropy: float


# ---------------------------------------------------------------------------
# Internal per-track observation record
# ---------------------------------------------------------------------------


@dataclass
class _Obs:
    """One frame's raw observation data, stored in the history buffer."""

    position: np.ndarray        # [px, py] in odom frame
    velocity: np.ndarray        # [vx, vy] in odom frame
    stop_weight: float          # IMMTrack.model_weights["stop"]
    robot_position: np.ndarray  # [px, py]
    robot_velocity: np.ndarray  # [vx, vy]
    timestamp: float            # seconds


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------


def _compute_log_emission(obs_feature: np.ndarray) -> np.ndarray:
    """Compute log p(obs | state) for all states simultaneously.

    Args:
        obs_feature: Feature vector of length :data:`_N_FEATURES`.

    Returns:
        Log-likelihood array of shape (4,), one value per intent state.
    """
    # Vectorised: diff shape (n_states, n_features)
    diff = (obs_feature[np.newaxis, :] - _EMIT_MEAN) / (_EMIT_STD + _EPS)
    # Sum log-Gaussian over features (independence assumption)
    log_lh = -0.5 * (diff ** 2).sum(axis=1) - np.log(_EMIT_STD + _EPS).sum(axis=1)
    return log_lh


def _features_from_obs(
    obs: _Obs, history: Deque["_Obs"]
) -> np.ndarray:
    """Compute the 5-dimensional feature vector from one observation.

    Features:
        1. ``heading_to_robot``: cosine similarity between ped velocity and
           vector pointing from ped to robot.
        2. ``lateral_speed``: speed component perpendicular to robot's forward
           direction (or to robot→ped direction when robot is static).
        3. ``speed_trend``: linear-regression slope of speed over last ≤5
           frames (positive = accelerating).
        4. ``stop_model_weight``: IMM Stop-model weight from the track.
        5. ``closing_rate``: signed rate of distance change w.r.t. robot
           (negative = approaching, positive = receding), in m/s.

    Args:
        obs: Current observation.
        history: Deque of previous observations (oldest first).

    Returns:
        Feature vector of shape (5,).
    """
    to_robot = obs.robot_position - obs.position
    to_robot_norm = float(np.linalg.norm(to_robot))
    vel_norm = float(np.linalg.norm(obs.velocity))

    # 1. Heading to robot
    if to_robot_norm < _EPS or vel_norm < _EPS:
        heading_to_robot = 0.0
    else:
        heading_to_robot = float(obs.velocity @ to_robot) / (vel_norm * to_robot_norm)

    # 2. Lateral speed relative to robot's forward direction
    rob_vel_norm = float(np.linalg.norm(obs.robot_velocity))
    if rob_vel_norm < _EPS:
        ref = to_robot / (to_robot_norm + _EPS)
    else:
        ref = obs.robot_velocity / rob_vel_norm
    lateral_component = obs.velocity - (float(obs.velocity @ ref)) * ref
    lateral_speed = float(np.linalg.norm(lateral_component))

    # 3. Speed trend over last ≤5 frames (slope in m/s per frame)
    recent = list(history)[-4:] + [obs]          # up to 5 entries
    speeds = np.array([float(np.linalg.norm(h.velocity)) for h in recent])
    if len(speeds) >= 2:
        t = np.arange(len(speeds), dtype=float)
        t -= t.mean()
        denom = float(t @ t) + _EPS
        speed_trend = float((t * (speeds - speeds.mean())).sum() / denom)
    else:
        speed_trend = 0.0
    speed_trend = float(np.clip(speed_trend, -3.0, 3.0))

    # 4. Stop model weight (directly from IMMTrack)
    stop_weight = float(np.clip(obs.stop_weight, 0.0, 1.0))

    # 5. Closing rate in m/s (negative = approaching)
    dist = to_robot_norm
    if history:
        prev = history[-1]
        prev_dist = float(np.linalg.norm(prev.robot_position - prev.position))
        dt = obs.timestamp - prev.timestamp
        if dt > _EPS:
            closing_rate = (dist - prev_dist) / dt
        else:
            closing_rate = 0.0
    else:
        closing_rate = 0.0
    closing_rate = float(np.clip(closing_rate, -5.0, 5.0))

    return np.array(
        [heading_to_robot, lateral_speed, speed_trend, stop_weight, closing_rate]
    )


# ---------------------------------------------------------------------------
# HMM forward pass (log-space, vectorised over states)
# ---------------------------------------------------------------------------


def _hmm_forward(
    features_seq: np.ndarray,   # shape (T, n_features)
    log_A: np.ndarray,          # shape (n_states, n_states), log transition
) -> np.ndarray:
    """Run the HMM forward algorithm in log-space.

    Uses a uniform prior over intent states on the first observation.

    Args:
        features_seq: Sequence of feature vectors, shape (T, :data:`_N_FEATURES`).
        log_A: Log of the HMM transition matrix, shape (n_states, n_states).

    Returns:
        Normalised belief vector of shape (n_states,).
    """
    T = len(features_seq)
    n = _N_STATES

    # Uniform prior
    log_alpha = np.full(n, -math.log(n))

    for t in range(T):
        log_emit = _compute_log_emission(features_seq[t])

        # log_alpha_new[j] = log_emit[j] + logsumexp_i(log_alpha[i] + log_A[i, j])
        # Broadcast: (n,) + (n, n) along axis-0 rows
        scores = log_alpha[:, np.newaxis] + log_A           # (n, n)
        # logsumexp over from-states (axis=0)
        max_scores = scores.max(axis=0)                     # (n,)
        log_sum = np.log(np.exp(scores - max_scores).sum(axis=0)) + max_scores  # (n,)
        log_alpha_new = log_emit + log_sum

        # Normalise to avoid underflow accumulation
        log_alpha_new -= np.log(np.exp(log_alpha_new - log_alpha_new.max()).sum()) + log_alpha_new.max()
        log_alpha = log_alpha_new

    probs = np.exp(log_alpha)
    probs /= probs.sum() + _EPS
    return probs


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class IntentEstimator:
    """HMM-based pedestrian intent classifier.

    Maintains a sliding history buffer per track and runs the HMM forward
    algorithm on each call to :meth:`update`.

    Args:
        transition_matrix: Optional 4×4 HMM transition matrix (rows = from,
            cols = to).  Defaults to :data:`DEFAULT_TRANSITION`.
        history_length: Maximum number of frames to retain per track.
        emission_params: Optional override for :data:`EMISSION_PARAMS`.
    """

    def __init__(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        history_length: int = 15,
        emission_params: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    ) -> None:
        self._history_length = history_length

        A = transition_matrix if transition_matrix is not None else DEFAULT_TRANSITION.copy()
        # Clamp to avoid log(0)
        A = np.clip(A, _EPS, 1.0)
        A /= A.sum(axis=1, keepdims=True)
        self._log_A: np.ndarray = np.log(A)

        # Override emission parameters if requested
        if emission_params is not None:
            global _EMIT_MEAN, _EMIT_STD
            _EMIT_MEAN = np.array(
                [[emission_params[s][f][0] for f in FEATURE_NAMES] for s in INTENT_NAMES]
            )
            _EMIT_STD = np.array(
                [[emission_params[s][f][1] for f in FEATURE_NAMES] for s in INTENT_NAMES]
            )

        self._histories: Dict[int, Deque[_Obs]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        track_id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        model_weights: Dict[str, float],
        robot_position: np.ndarray,
        robot_velocity: np.ndarray,
        timestamp: float = 0.0,
    ) -> IntentBelief:
        """Update the intent estimate for one track.

        Args:
            track_id: Identifier matching the source ``IMMTrack``.
            position: Pedestrian position ``[px, py]`` in odom frame.
            velocity: Pedestrian velocity ``[vx, vy]`` in odom frame.
            model_weights: IMM model weights dict, must contain ``"stop"`` key.
            robot_position: Robot position ``[px, py]`` in odom frame.
            robot_velocity: Robot velocity ``[vx, vy]`` in odom frame.
            timestamp: Current time in seconds (used for closing-rate
                computation).

        Returns:
            :class:`IntentBelief` for this track.
        """
        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self._history_length)

        history = self._histories[track_id]

        obs = _Obs(
            position=np.asarray(position, dtype=float),
            velocity=np.asarray(velocity, dtype=float),
            stop_weight=float(model_weights.get("stop", 0.0)),
            robot_position=np.asarray(robot_position, dtype=float),
            robot_velocity=np.asarray(robot_velocity, dtype=float),
            timestamp=float(timestamp),
        )

        # Build feature vector for current observation
        feat = _features_from_obs(obs, history)

        # Append AFTER computing features so history refers to past frames
        history.append(obs)

        # Build feature sequence from stored history (including current)
        hist_list = list(history)
        features_seq = np.array([
            _features_from_obs(h, deque(list(history)[: max(0, i)])  )
            for i, h in enumerate(hist_list)
        ])  # (T, n_features)

        # HMM forward pass
        probs = _hmm_forward(features_seq, self._log_A)

        probs_dict = {name: float(probs[i]) for i, name in enumerate(INTENT_NAMES)}
        dominant = max(probs_dict, key=lambda k: probs_dict[k])
        entropy = float(-np.sum(probs * np.log(probs + _EPS)))

        return IntentBelief(
            track_id=track_id,
            probs=probs_dict,
            dominant=dominant,
            entropy=entropy,
        )

    def reset(self, track_id: int) -> None:
        """Clear the history for a specific track.

        Args:
            track_id: Track whose history should be erased.
        """
        if track_id in self._histories:
            self._histories[track_id].clear()

    def remove_stale(self, active_ids: Set[int]) -> None:
        """Delete history buffers for tracks no longer in the active set.

        Should be called at the end of each pipeline cycle with the set of
        currently tracked IDs.

        Args:
            active_ids: Set of track IDs that are still alive.
        """
        stale = [tid for tid in self._histories if tid not in active_ids]
        for tid in stale:
            del self._histories[tid]
