"""Interacting Multiple Model (IMM) tracker for pedestrian state estimation.

Implements three motion models fused via the IMM algorithm:

  * ``cv``   – Constant Velocity (4-D state ``[px, py, vx, vy]``)
  * ``ctrv`` – Constant Turn Rate & Velocity (5-D state ``[px, py, v, yaw, ω]``)
  * ``stop`` – Near-static (same 4-D state as CV, 10× lower process noise)

The fused output ``IMMTrack.x`` is always ``[px, py, vx, vy]`` shape (4,) and
``IMMTrack.P`` is always 4×4, preserving backward compatibility with the
existing ``KalmanTracker`` / ``Track`` interface used by downstream nodes.

Performance target: ``IMMTracker.update()`` < 5 ms for 8 pedestrians at 10 Hz
on a Raspberry Pi 4 (uses numpy matrix operations throughout; no Python loops
over state dimensions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MODEL_NAMES: Tuple[str, ...] = ("cv", "ctrv", "stop")
_IDX: Dict[str, int] = {n: i for i, n in enumerate(MODEL_NAMES)}
_EPS: float = 1e-9

#: Default Markov transition matrix — rows = from-model, cols = to-model.
DEFAULT_TRANSITION: np.ndarray = np.array(
    [
        [0.90, 0.05, 0.05],  # cv   → cv, ctrv, stop
        [0.10, 0.85, 0.05],  # ctrv → cv, ctrv, stop
        [0.20, 0.05, 0.75],  # stop → cv, ctrv, stop
    ],
    dtype=np.float64,
)

# Measurement matrices H that project state → [px, py]
_H_CV: np.ndarray = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])    # 2×4
_H_CTRV: np.ndarray = np.array(
    [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
)  # 2×5
_H_STOP: np.ndarray = _H_CV.copy()

_H: Tuple[np.ndarray, ...] = (_H_CV, _H_CTRV, _H_STOP)

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class IMMTrack:
    """Fused track produced by :class:`IMMTracker`.

    Attributes:
        track_id: Unique integer identifier for this track.
        x: Fused state vector ``[px, py, vx, vy]``, shape (4,).
        P: Fused 4×4 covariance matrix.
        last_seen: Timestamp (seconds) of the most recent matched detection.
        hits: Number of frames this track has been matched to a detection.
        misses: Consecutive frames with no matching detection.
        confidence: Exponentially smoothed detection confidence in [0, 1].
        model_weights: Per-model probability, e.g. ``{"cv": 0.7, "ctrv": 0.2, "stop": 0.1}``.
        dominant_model: Name of the highest-weight model.
    """

    track_id: int
    x: np.ndarray
    P: np.ndarray
    last_seen: float
    hits: int
    misses: int
    confidence: float
    model_weights: Dict[str, float]
    dominant_model: str


# ---------------------------------------------------------------------------
# Internal per-filter state (not part of the public API)
# ---------------------------------------------------------------------------


@dataclass
class _FilterState:
    x: np.ndarray  # native state: 4D for CV/Stop, 5D for CTRV
    P: np.ndarray  # covariance: 4×4 or 5×5


@dataclass
class _TrackInternal:
    track_id: int
    filters: List[_FilterState]   # length 3, indexed by _IDX
    mu: np.ndarray                 # model weights, shape (3,)
    last_seen: float
    hits: int
    misses: int
    confidence: float


# ---------------------------------------------------------------------------
# Motion model helpers — state transition matrices and process noise
# ---------------------------------------------------------------------------


def _cv_F(dt: float) -> np.ndarray:
    """4×4 constant-velocity state transition matrix.

    Args:
        dt: Time step in seconds.

    Returns:
        4×4 numpy array.
    """
    return np.array(
        [
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _cv_Q(dt: float, sigma_a: float) -> np.ndarray:
    """4×4 discrete white-noise process noise for the CV model.

    Args:
        dt: Time step in seconds.
        sigma_a: Acceleration standard deviation in m/s².

    Returns:
        4×4 numpy array.
    """
    q = sigma_a ** 2
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    return q * np.array(
        [
            [dt4 / 4.0, 0.0,       dt3 / 2.0, 0.0      ],
            [0.0,       dt4 / 4.0, 0.0,       dt3 / 2.0],
            [dt3 / 2.0, 0.0,       dt2,       0.0      ],
            [0.0,       dt3 / 2.0, 0.0,       dt2      ],
        ]
    )


def _stop_Q(dt: float, sigma_a: float) -> np.ndarray:
    """Process noise for the Stop model.

    Position noise is a tiny constant (pedestrian may drift slightly);
    velocity noise is essentially zero — the model assumes the target
    is standing still.  This is deliberately far tighter than CV so
    the IMM likelihood correctly rewards the Stop filter when a
    pedestrian is truly stationary.

    Args:
        dt: Time step in seconds.
        sigma_a: Base acceleration std (m/s^2) — kept for API compat.

    Returns:
        4x4 numpy array.
    """
    sigma_pos = 1e-3   # ~1 mm position drift per step
    sigma_vel = 1e-4   # essentially zero velocity change
    return np.diag([sigma_pos**2, sigma_pos**2,
                    sigma_vel**2, sigma_vel**2])


def _ctrv_predict(x: np.ndarray, dt: float) -> np.ndarray:
    """Nonlinear CTRV state prediction (handles near-zero yaw rate).

    Args:
        x: State vector ``[px, py, v, yaw, ω]``, shape (5,).
        dt: Time step in seconds.

    Returns:
        Predicted state, shape (5,).
    """
    px, py, v, yaw, omega = x
    xn = x.copy()
    if abs(omega) < _EPS:
        xn[0] = px + v * math.cos(yaw) * dt
        xn[1] = py + v * math.sin(yaw) * dt
    else:
        xn[0] = px + (v / omega) * (math.sin(yaw + omega * dt) - math.sin(yaw))
        xn[1] = py + (v / omega) * (-math.cos(yaw + omega * dt) + math.cos(yaw))
    xn[3] = yaw + omega * dt
    # Normalise yaw to [-π, π]
    xn[3] = (xn[3] + math.pi) % (2.0 * math.pi) - math.pi
    return xn


def _ctrv_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """Jacobian of the CTRV prediction function w.r.t. state.

    Args:
        x: State vector ``[px, py, v, yaw, ω]``, shape (5,).
        dt: Time step in seconds.

    Returns:
        5×5 Jacobian matrix.
    """
    _, _, v, yaw, omega = x
    F = np.eye(5)
    if abs(omega) < _EPS:
        F[0, 2] =  math.cos(yaw) * dt
        F[0, 3] = -v * math.sin(yaw) * dt
        F[1, 2] =  math.sin(yaw) * dt
        F[1, 3] =  v * math.cos(yaw) * dt
    else:
        sy  = math.sin(yaw)
        cy  = math.cos(yaw)
        syd = math.sin(yaw + omega * dt)
        cyd = math.cos(yaw + omega * dt)
        F[0, 2] = (syd - sy) / omega
        F[0, 3] = (v / omega) * (cyd - cy)
        F[0, 4] = (v / omega ** 2) * (-syd + sy + omega * dt * cyd)
        F[1, 2] = (-cyd + cy) / omega
        F[1, 3] = (v / omega) * (syd - sy)
        F[1, 4] = (v / omega ** 2) * (cyd - cy + omega * dt * syd)
    F[3, 4] = dt
    return F


def _ctrv_Q(dt: float, sigma_a: float, sigma_yaw_dd: float) -> np.ndarray:
    """5×5 process noise for the CTRV model.

    Args:
        dt: Time step in seconds.
        sigma_a: Speed (longitudinal) noise std in m/s².
        sigma_yaw_dd: Yaw-rate acceleration noise std in rad/s².

    Returns:
        5×5 numpy array.
    """
    Q = np.zeros((5, 5))
    Q[2, 2] = (sigma_a * dt) ** 2
    Q[3, 3] = (sigma_yaw_dd * dt ** 2 / 2.0) ** 2
    Q[4, 4] = (sigma_yaw_dd * dt) ** 2
    return Q


# ---------------------------------------------------------------------------
# State-space conversion helpers
# ---------------------------------------------------------------------------


def _cv_to_ctrv(x4: np.ndarray) -> np.ndarray:
    """Convert ``[px, py, vx, vy]`` → ``[px, py, v, yaw, ω=0]``."""
    px, py, vx, vy = x4
    v   = math.sqrt(vx ** 2 + vy ** 2)
    yaw = math.atan2(vy, vx)
    return np.array([px, py, v, yaw, 0.0])


def _ctrv_to_cv(x5: np.ndarray) -> np.ndarray:
    """Convert ``[px, py, v, yaw, ω]`` → ``[px, py, vx, vy]``."""
    px, py, v, yaw, _ = x5
    return np.array([px, py, v * math.cos(yaw), v * math.sin(yaw)])


def _P_cv_to_ctrv(
    P4: np.ndarray, x4: np.ndarray, var_omega: float = 0.25
) -> np.ndarray:
    """Propagate 4×4 CV covariance → 5×5 CTRV via Jacobian.

    Args:
        P4: 4×4 CV covariance.
        x4: Current CV state ``[px, py, vx, vy]``.
        var_omega: Prior variance added to the yaw-rate diagonal element.

    Returns:
        5×5 CTRV covariance.
    """
    vx, vy = x4[2], x4[3]
    v = max(math.sqrt(vx ** 2 + vy ** 2), _EPS)
    J = np.zeros((5, 4))
    J[0, 0] = 1.0;  J[1, 1] = 1.0
    J[2, 2] =  vx / v;       J[2, 3] = vy / v
    J[3, 2] = -vy / (v ** 2); J[3, 3] = vx / (v ** 2)
    P5 = J @ P4 @ J.T
    P5[4, 4] += var_omega
    return P5


def _P_ctrv_to_cv(P5: np.ndarray, x5: np.ndarray) -> np.ndarray:
    """Propagate 5×5 CTRV covariance → 4×4 CV via Jacobian.

    Args:
        P5: 5×5 CTRV covariance.
        x5: Current CTRV state ``[px, py, v, yaw, ω]``.

    Returns:
        4×4 CV covariance.
    """
    v, yaw = x5[2], x5[3]
    J = np.zeros((4, 5))
    J[0, 0] = 1.0;  J[1, 1] = 1.0
    J[2, 2] =  math.cos(yaw); J[2, 3] = -v * math.sin(yaw)
    J[3, 2] =  math.sin(yaw); J[3, 3] =  v * math.cos(yaw)
    return J @ P5 @ J.T


def _to_4d(
    x: np.ndarray, P: np.ndarray, model_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert any model's native state to common 4D ``[px, py, vx, vy]`` form."""
    if model_idx == _IDX["ctrv"]:
        return _ctrv_to_cv(x), _P_ctrv_to_cv(P, x)
    return x.copy(), P.copy()


def _from_4d(
    x4: np.ndarray, P4: np.ndarray, model_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert common 4D form to a model's native state."""
    if model_idx == _IDX["ctrv"]:
        return _cv_to_ctrv(x4), _P_cv_to_ctrv(P4, x4)
    return x4.copy(), P4.copy()


# ---------------------------------------------------------------------------
# Gaussian likelihood (inline — no scipy)
# ---------------------------------------------------------------------------


def _gaussian_likelihood(y: np.ndarray, S: np.ndarray) -> float:
    """Compute the multivariate Gaussian likelihood N(y; 0, S).

    Args:
        y: Innovation vector, shape (d,).
        S: Innovation covariance, shape (d, d).

    Returns:
        Scalar likelihood value (clamped above ``_EPS``).
    """
    d = len(y)
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return _EPS
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return _EPS
    exponent = -0.5 * float(y @ S_inv @ y)
    log_lh = exponent - 0.5 * (d * math.log(2.0 * math.pi) + logdet)
    return max(math.exp(log_lh), _EPS)


# ---------------------------------------------------------------------------
# IMM sub-routines
# ---------------------------------------------------------------------------


def _mix(
    filters: List[_FilterState],
    mu: np.ndarray,
    P_trans: np.ndarray,
) -> Tuple[List[_FilterState], np.ndarray]:
    """Compute mixed initial conditions for each model.

    All arithmetic is done in the common 4D space; results are projected back
    to each model's native space before being returned.

    Args:
        filters: Current posterior filter states (length 3).
        mu: Current model weights, shape (3,).
        P_trans: Markov transition matrix (3×3), rows=from, cols=to.

    Returns:
        Tuple of (mixed_filters, c) where c[j] is the normalisation constant
        Σ_i P_trans[i,j] · μ_i.
    """
    n = len(MODEL_NAMES)

    # c[j] = Σ_i P_trans[i,j] · μ_i
    c = P_trans.T @ mu                                        # shape (3,)

    # μ_ij[i, j] = P_trans[i,j] · μ_i / c[j]
    mu_ij = (P_trans * mu[:, np.newaxis]) / (c[np.newaxis, :] + _EPS)  # (3,3)

    # Convert all posteriors to common 4D form once
    x4s: List[np.ndarray] = []
    P4s: List[np.ndarray] = []
    for i, fs in enumerate(filters):
        x4, P4 = _to_4d(fs.x, fs.P, i)
        x4s.append(x4)
        P4s.append(P4)

    mixed: List[_FilterState] = []
    for j in range(n):
        # Weighted mean in 4D common space
        x4_mix = sum(mu_ij[i, j] * x4s[i] for i in range(n))  # type: ignore[arg-type]

        # Weighted mixture covariance in 4D common space
        P4_mix = sum(
            mu_ij[i, j] * (
                P4s[i] + np.outer(x4s[i] - x4_mix, x4s[i] - x4_mix)
            )
            for i in range(n)
        )  # type: ignore[arg-type]

        # Project mixed state to model j's native space
        x_j, P_j = _from_4d(x4_mix, P4_mix, j)
        mixed.append(_FilterState(x_j, P_j))

    return mixed, c


def _mode_filter_update(
    fs: _FilterState,
    z: np.ndarray,
    model_idx: int,
    dt: float,
    sigma_a: float,
    sigma_yaw_dd: float,
    R: np.ndarray,
) -> Tuple[_FilterState, float]:
    """Predict then update a single Kalman/EKF filter for one step.

    Args:
        fs: Mixed initial filter state.
        z: Measurement ``[px, py]``.
        model_idx: Index into ``MODEL_NAMES`` (0=cv, 1=ctrv, 2=stop).
        dt: Time step in seconds.
        sigma_a: Acceleration noise std.
        sigma_yaw_dd: Yaw-rate acceleration noise std (CTRV only).
        R: 2×2 measurement noise covariance.

    Returns:
        Tuple of (updated FilterState, measurement likelihood).
    """
    H = _H[model_idx]

    if model_idx == _IDX["ctrv"]:
        F = _ctrv_jacobian(fs.x, dt)
        Q = _ctrv_Q(dt, sigma_a, sigma_yaw_dd)
        x_pred = _ctrv_predict(fs.x, dt)
    elif model_idx == _IDX["stop"]:
        F = _cv_F(dt)
        Q = _stop_Q(dt, sigma_a)
        x_pred = F @ fs.x
    else:  # cv
        F = _cv_F(dt)
        Q = _cv_Q(dt, sigma_a)
        x_pred = F @ fs.x

    P_pred = F @ fs.P @ F.T + Q

    # Innovation
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R

    # Kalman gain
    K = P_pred @ H.T @ np.linalg.inv(S)

    # Joseph form for numerical stability: P = (I-KH)P(I-KH)^T + KRK^T
    I_KH = np.eye(len(x_pred)) - K @ H
    x_post = x_pred + K @ y
    P_post = I_KH @ P_pred @ I_KH.T + K @ R @ K.T

    # Normalise yaw for CTRV
    if model_idx == _IDX["ctrv"]:
        x_post[3] = (x_post[3] + math.pi) % (2.0 * math.pi) - math.pi

    lh = _gaussian_likelihood(y, S)
    return _FilterState(x_post, P_post), lh


def _fuse_estimates(
    filters: List[_FilterState], mu: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse three filter posteriors into a single 4D estimate.

    Args:
        filters: Updated filter states (length 3).
        mu: Updated model weights, shape (3,).

    Returns:
        Tuple of (x_fused shape (4,), P_fused shape (4,4)).
    """
    x4s = [_to_4d(filters[i].x, filters[i].P, i)[0] for i in range(3)]
    P4s = [_to_4d(filters[i].x, filters[i].P, i)[1] for i in range(3)]

    x_fused = sum(mu[i] * x4s[i] for i in range(3))  # type: ignore[arg-type]
    P_fused  = sum(
        mu[i] * (P4s[i] + np.outer(x4s[i] - x_fused, x4s[i] - x_fused))
        for i in range(3)
    )  # type: ignore[arg-type]
    return x_fused, P_fused


# ---------------------------------------------------------------------------
# Greedy nearest-neighbour assignment (module-level helper)
# ---------------------------------------------------------------------------


def _greedy_nn_assign(
    pred_pos: np.ndarray,   # (N, 2) predicted track positions
    det_pos: np.ndarray,    # (M, 2) detection positions
    max_dist: float,
) -> List[Tuple[int, int]]:
    """Greedy nearest-neighbour assignment between tracks and detections.

    Args:
        pred_pos: (N, 2) array of predicted track ``[px, py]`` positions.
        det_pos: (M, 2) array of detection ``[px, py]`` positions.
        max_dist: Maximum allowable assignment distance in metres.

    Returns:
        List of ``(track_idx, det_idx)`` pairs.
    """
    if len(pred_pos) == 0 or len(det_pos) == 0:
        return []

    diff = pred_pos[:, np.newaxis, :] - det_pos[np.newaxis, :, :]  # (N, M, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))                         # (N, M)

    used_t: set = set()
    used_d: set = set()
    assignments: List[Tuple[int, int]] = []

    dist_work = dist.copy()
    dist_work[dist_work > max_dist] = np.inf

    while True:
        if used_t:
            dist_work[list(used_t), :] = np.inf
        if used_d:
            dist_work[:, list(used_d)] = np.inf
        if np.all(np.isinf(dist_work)):
            break
        t_idx, d_idx = np.unravel_index(dist_work.argmin(), dist_work.shape)
        if np.isinf(dist_work[t_idx, d_idx]):
            break
        assignments.append((int(t_idx), int(d_idx)))
        used_t.add(int(t_idx))
        used_d.add(int(d_idx))

    return assignments


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class IMMTracker:
    """Track multiple pedestrians using an Interacting Multiple Model filter.

    Provides the same public interface as ``KalmanTracker`` (``update()``
    and ``all_tracks``) with additional IMM-specific fields on each track.

    Args:
        max_distance: Gate distance for detection-to-track assignment (metres).
        max_misses: Consecutive missed frames before a track is deleted.
        min_hits: Minimum confirmed hits before a track is returned publicly.
        sigma_a: Acceleration noise std for CV/Stop models (m/s²).
        sigma_r: Position measurement noise std (metres).
        sigma_yaw_dd: Yaw-rate acceleration noise std for CTRV (rad/s²).
        transition_matrix: Optional 3×3 Markov transition matrix (rows=from,
            cols=to).  Defaults to :data:`DEFAULT_TRANSITION`.
    """

    def __init__(
        self,
        max_distance: float = 1.5,
        max_misses: int = 5,
        min_hits: int = 2,
        sigma_a: float = 0.5,
        sigma_r: float = 0.15,
        sigma_yaw_dd: float = 0.3,
        transition_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self._max_distance = max_distance
        self._max_misses = max_misses
        self._min_hits = min_hits
        self._sigma_a = sigma_a
        self._sigma_yaw_dd = sigma_yaw_dd
        self._R: np.ndarray = (sigma_r ** 2) * np.eye(2)
        self._P_trans: np.ndarray = (
            transition_matrix
            if transition_matrix is not None
            else DEFAULT_TRANSITION.copy()
        )
        self._tracks: Dict[int, _TrackInternal] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Tuple[float, float, float]],
        now: float,
    ) -> List[IMMTrack]:
        """Process a new set of detections and return confirmed tracks.

        Args:
            detections: List of ``(x, y, confidence)`` tuples in the world
                (odom) frame.
            now: Current ROS time in seconds (``node.get_clock().now()`` as
                float via ``stamp.sec + stamp.nanosec * 1e-9``).

        Returns:
            List of :class:`IMMTrack` objects whose ``hits`` ≥ ``min_hits``.
        """
        track_list = list(self._tracks.values())

        # --- Compute per-track predicted positions for assignment --------
        if track_list:
            pred_positions = np.array([
                self._predicted_position(t, max(now - t.last_seen, 1e-3))
                for t in track_list
            ])  # (N, 2)
        else:
            pred_positions = np.empty((0, 2))

        det_positions = (
            np.array([[d[0], d[1]] for d in detections])
            if detections
            else np.empty((0, 2))
        )

        # --- Greedy assignment ------------------------------------------
        assignments = _greedy_nn_assign(pred_positions, det_positions, self._max_distance)

        assigned_t: set = set()
        assigned_d: set = set()

        for t_idx, d_idx in assignments:
            track = track_list[t_idx]
            det = detections[d_idx]
            z = np.array([det[0], det[1]])
            dt = max(now - track.last_seen, 1e-3)

            self._imm_step(track, z, dt)
            track.last_seen = now
            track.hits += 1
            track.misses = 0
            track.confidence = 0.9 * track.confidence + 0.1 * det[2]
            assigned_t.add(t_idx)
            assigned_d.add(d_idx)

        # --- Predict-only for unmatched tracks --------------------------
        for t_idx, track in enumerate(track_list):
            if t_idx not in assigned_t:
                dt = max(now - track.last_seen, 1e-3)
                self._predict_only(track, dt)
                track.misses += 1

        # --- New tracks for unmatched detections -----------------------
        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_d:
                self._create_track(det, now)

        # --- Prune stale tracks ----------------------------------------
        stale = [tid for tid, t in self._tracks.items() if t.misses > self._max_misses]
        for tid in stale:
            del self._tracks[tid]

        # --- Return confirmed tracks ------------------------------------
        return [
            self._build_output(t)
            for t in self._tracks.values()
            if t.hits >= self._min_hits
        ]

    @property
    def all_tracks(self) -> List[IMMTrack]:
        """All active tracks, including unconfirmed ones.

        Returns:
            List of :class:`IMMTrack` for every track regardless of hit count.
        """
        return [self._build_output(t) for t in self._tracks.values()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predicted_position(self, track: _TrackInternal, dt: float) -> np.ndarray:
        """Return approximate predicted [px, py] using the CV filter only.

        Args:
            track: Internal track state.
            dt: Elapsed time in seconds.

        Returns:
            Shape (2,) array.
        """
        F = _cv_F(dt)
        x4, _ = _to_4d(track.filters[0].x, track.filters[0].P, 0)
        return (F @ x4)[:2]

    def _imm_step(
        self, track: _TrackInternal, z: np.ndarray, dt: float
    ) -> None:
        """Run a full IMM update step (mix → predict → update → fuse weights).

        Args:
            track: Internal track to update in-place.
            z: Measurement ``[px, py]``.
            dt: Time step in seconds.
        """
        # 1. Mixing on posteriors
        mixed_filters, c = _mix(track.filters, track.mu, self._P_trans)

        # 2. Mode-conditioned predict + update, collect likelihoods
        updated: List[_FilterState] = []
        likelihoods = np.zeros(3)
        for j in range(3):
            fs_new, lh = _mode_filter_update(
                mixed_filters[j], z, j, dt,
                self._sigma_a, self._sigma_yaw_dd, self._R,
            )
            updated.append(fs_new)
            likelihoods[j] = lh

        # 3. Mode probability update:  μ_j ∝ Λ_j · c̄_j
        mu_new = likelihoods * c
        total = mu_new.sum()
        track.mu = mu_new / total if total > _EPS else np.ones(3) / 3.0
        track.filters = updated

    def _predict_only(self, track: _TrackInternal, dt: float) -> None:
        """Predict all filters independently without a measurement update.

        Model weights are unchanged (no likelihood information available).

        Args:
            track: Internal track to update in-place.
            dt: Time step in seconds.
        """
        for j, fs in enumerate(track.filters):
            if j == _IDX["ctrv"]:
                F = _ctrv_jacobian(fs.x, dt)
                Q = _ctrv_Q(dt, self._sigma_a, self._sigma_yaw_dd)
                x_pred = _ctrv_predict(fs.x, dt)
            elif j == _IDX["stop"]:
                F = _cv_F(dt)
                Q = _stop_Q(dt, self._sigma_a)
                x_pred = F @ fs.x
            else:  # cv
                F = _cv_F(dt)
                Q = _cv_Q(dt, self._sigma_a)
                x_pred = F @ fs.x
            track.filters[j] = _FilterState(x_pred, F @ fs.P @ F.T + Q)

    def _create_track(
        self, det: Tuple[float, float, float], now: float
    ) -> None:
        """Initialise a new track from a single detection.

        Args:
            det: ``(x, y, confidence)`` detection tuple.
            now: Timestamp in seconds.
        """
        px, py, conf = det[0], det[1], det[2]
        x4 = np.array([px, py, 0.0, 0.0])
        P4 = np.diag([0.5 ** 2, 0.5 ** 2, 2.0 ** 2, 2.0 ** 2])

        x_ctrv = _cv_to_ctrv(x4)
        P_ctrv = _P_cv_to_ctrv(P4, x4)

        filters = [
            _FilterState(x4.copy(), P4.copy()),   # cv
            _FilterState(x_ctrv, P_ctrv),          # ctrv
            _FilterState(x4.copy(), P4.copy()),   # stop
        ]
        self._tracks[self._next_id] = _TrackInternal(
            track_id=self._next_id,
            filters=filters,
            mu=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            last_seen=now,
            hits=1,
            misses=0,
            confidence=float(conf),
        )
        self._next_id += 1

    def _build_output(self, track: _TrackInternal) -> IMMTrack:
        """Convert an internal track to a public :class:`IMMTrack`.

        Args:
            track: Internal track state.

        Returns:
            Fully populated :class:`IMMTrack`.
        """
        x_fused, P_fused = _fuse_estimates(track.filters, track.mu)
        weights = {name: float(track.mu[i]) for i, name in enumerate(MODEL_NAMES)}
        dominant = max(weights, key=lambda k: weights[k])
        return IMMTrack(
            track_id=track.track_id,
            x=x_fused,
            P=P_fused,
            last_seen=track.last_seen,
            hits=track.hits,
            misses=track.misses,
            confidence=track.confidence,
            model_weights=weights,
            dominant_model=dominant,
        )
