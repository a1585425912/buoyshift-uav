"""Distributed per-buoy estimation of a 3-D constant-velocity UAV trajectory.

Each buoy maintains sufficient statistics from its own position observations.
The state exchanged by DPC is

    [bx, vx, by, vy, bz, vz, dx, dy, dz]

where ``b`` and ``v`` parameterize ``p(t) = b + v*t`` and ``d`` is the unit
flight-direction vector.  Keeping direction explicit makes the quantity being
agreed upon unambiguous, while the Cartesian velocity components retain speed
information and avoid angle wrap-around at +/-pi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INTERCEPT_COLUMNS = np.array([0, 2, 4])
VELOCITY_COLUMNS = np.array([1, 3, 5])
DIRECTION_COLUMNS = np.array([6, 7, 8])


@dataclass
class LineEstimatorState:
    """Least-squares sufficient statistics maintained independently by buoys."""

    count: int
    sum_t: float
    sum_t2: float
    sum_xyz: np.ndarray
    sum_txyz: np.ndarray


@dataclass(frozen=True)
class TrajectoryPrediction:
    """Current UAV position/velocity prediction and its uncertainty."""

    state: np.ndarray
    covariance: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self.state[:, :3]

    @property
    def velocities(self) -> np.ndarray:
        return self.state[:, 3:]


def init_line_estimator(n: int) -> LineEstimatorState:
    return LineEstimatorState(
        count=0,
        sum_t=0.0,
        sum_t2=0.0,
        sum_xyz=np.zeros((n, 3), dtype=np.float64),
        sum_txyz=np.zeros((n, 3), dtype=np.float64),
    )


def _unit_directions(velocity_xyz: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    velocity = np.asarray(velocity_xyz, dtype=np.float64)
    norms = np.linalg.norm(velocity, axis=1, keepdims=True)
    direction = np.zeros_like(velocity)
    moving = norms[:, 0] > 1e-12
    direction[moving] = velocity[moving] / norms[moving]
    if fallback is not None:
        old = np.asarray(fallback, dtype=np.float64)
        old_norm = np.linalg.norm(old, axis=1, keepdims=True)
        valid_old = (~moving) & (old_norm[:, 0] > 1e-12)
        direction[valid_old] = old[valid_old] / old_norm[valid_old]
    return direction


def update_local_line_estimates(
    state: LineEstimatorState,
    time_s: float,
    observations_xyz: np.ndarray,
) -> np.ndarray:
    """Add one local observation and return each buoy's 3-D trajectory state."""
    xyz = np.asarray(observations_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape != state.sum_xyz.shape:
        raise ValueError(f"observations_xyz must have shape {state.sum_xyz.shape}")
    t = float(time_s)
    state.count += 1
    state.sum_t += t
    state.sum_t2 += t * t
    state.sum_xyz += xyz
    state.sum_txyz += t * xyz

    n_obs = float(state.count)
    denominator = n_obs * state.sum_t2 - state.sum_t * state.sum_t
    if denominator <= 1e-12:
        velocity = np.zeros_like(state.sum_xyz)
        intercept = state.sum_xyz / n_obs
    else:
        velocity = (n_obs * state.sum_txyz - state.sum_t * state.sum_xyz) / denominator
        intercept = (state.sum_xyz - state.sum_t * velocity) / n_obs
    direction = _unit_directions(velocity)
    return np.column_stack(
        [
            intercept[:, 0], velocity[:, 0],
            intercept[:, 1], velocity[:, 1],
            intercept[:, 2], velocity[:, 2],
            direction,
        ]
    )


def normalize_trajectory_directions(trajectory_states: np.ndarray) -> np.ndarray:
    """Normalize direction after linear consensus, with velocity as fallback."""
    states = np.asarray(trajectory_states, dtype=np.float64).copy()
    if states.ndim != 2 or states.shape[1] != 9:
        raise ValueError("trajectory_states must have shape (N, 9)")
    directions = states[:, DIRECTION_COLUMNS]
    velocities = states[:, VELOCITY_COLUMNS]
    direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    valid = direction_norm[:, 0] > 1e-12
    directions[valid] /= direction_norm[valid]
    directions[~valid] = _unit_directions(velocities[~valid])
    states[:, DIRECTION_COLUMNS] = directions
    return states


def trajectory_velocities(trajectory_states: np.ndarray) -> np.ndarray:
    """Return consensus Cartesian velocities ``[vx, vy, vz]``."""
    states = np.asarray(trajectory_states, dtype=np.float64)
    return states[:, VELOCITY_COLUMNS]


def trajectory_directions(trajectory_states: np.ndarray) -> np.ndarray:
    """Return consensus unit flight directions ``[dx, dy, dz]``."""
    states = np.asarray(trajectory_states, dtype=np.float64)
    return states[:, DIRECTION_COLUMNS]


def trajectory_state_at_time(trajectory_states: np.ndarray, time_s: float) -> np.ndarray:
    """Convert line parameters to the current state [px,py,pz,vx,vy,vz]."""
    states = np.asarray(trajectory_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 9:
        raise ValueError("trajectory_states must have shape (N, 9)")
    position = predict_positions(states, time_s)
    velocity = trajectory_velocities(states)
    return np.column_stack([position, velocity])


def trajectory_prediction_covariance(
    estimator_state: LineEstimatorState,
    time_s: float,
    measurement_std: float,
    initial_velocity_std: float,
) -> np.ndarray:
    """Return OLS covariance for [position, velocity] at time_s.

    The three coordinate axes use the same observation times and independent
    isotropic measurement noise.  Before velocity becomes observable, retain
    the configured initial velocity uncertainty instead of fabricating a
    zero-variance velocity measurement.
    """
    if estimator_state.count < 1:
        raise ValueError("at least one trajectory observation is required")
    sigma2 = max(float(measurement_std), 1e-9) ** 2
    velocity_var = max(float(initial_velocity_std), 1e-9) ** 2
    count = float(estimator_state.count)
    determinant = count * estimator_state.sum_t2 - estimator_state.sum_t**2

    if determinant <= 1e-12:
        covariance_pv = np.diag([sigma2, velocity_var])
    else:
        covariance_bv = sigma2 / determinant * np.array(
            [
                [estimator_state.sum_t2, -estimator_state.sum_t],
                [-estimator_state.sum_t, count],
            ],
            dtype=np.float64,
        )
        transform = np.array([[1.0, float(time_s)], [0.0, 1.0]], dtype=np.float64)
        covariance_pv = transform @ covariance_bv @ transform.T

    n = estimator_state.sum_xyz.shape[0]
    covariance = np.zeros((n, 6, 6), dtype=np.float64)
    for axis in range(3):
        indices = np.array([axis, axis + 3])
        covariance[:, indices[:, None], indices] = covariance_pv
    return covariance


def predict_trajectory(
    trajectory_states: np.ndarray,
    estimator_state: LineEstimatorState,
    time_s: float,
    measurement_std: float,
    initial_velocity_std: float,
) -> TrajectoryPrediction:
    """Build a trajectory-aware UAV pseudo-measurement for Kalman fusion."""
    return TrajectoryPrediction(
        state=trajectory_state_at_time(trajectory_states, time_s),
        covariance=trajectory_prediction_covariance(
            estimator_state,
            time_s,
            measurement_std,
            initial_velocity_std,
        ),
    )


def consensus_independent_covariance(
    weight_matrix: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Propagate independent local-estimate covariance through one consensus step."""
    weights = np.asarray(weight_matrix, dtype=np.float64)
    values = np.asarray(covariance, dtype=np.float64)
    if weights.shape != (values.shape[0], values.shape[0]):
        raise ValueError("weight_matrix and covariance must have matching node dimensions")
    return np.einsum("ij,jkl->ikl", weights**2, values)


def trajectory_state_from_position_velocity(
    position_xyz: np.ndarray,
    velocity_xyz: np.ndarray,
    time_s: float,
) -> np.ndarray:
    """Express a position/velocity estimate as a 3-D trajectory state."""
    position = np.asarray(position_xyz, dtype=np.float64)
    velocity = np.asarray(velocity_xyz, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape:
        raise ValueError("position_xyz and velocity_xyz must both have shape (N, 3)")
    t = float(time_s)
    intercept = position - t * velocity
    direction = _unit_directions(velocity)
    return np.column_stack(
        [
            intercept[:, 0], velocity[:, 0],
            intercept[:, 1], velocity[:, 1],
            intercept[:, 2], velocity[:, 2],
            direction,
        ]
    )


def predict_positions(line_params: np.ndarray, time_s: float, height: float | None = None) -> np.ndarray:
    """Predict UAV positions from 3-D states (or legacy 2-D four-column states)."""
    params = np.asarray(line_params, dtype=np.float64)
    t = float(time_s)
    if params.ndim != 2 or params.shape[1] not in (4, 9):
        raise ValueError("line_params must have shape (N, 9), or legacy shape (N, 4)")
    x = params[:, 0] + params[:, 1] * t
    y = params[:, 2] + params[:, 3] * t
    if params.shape[1] == 9:
        z = params[:, 4] + params[:, 5] * t
    elif height is not None:
        z = np.full(params.shape[0], float(height), dtype=np.float64)
    else:
        raise ValueError("height is required for legacy four-column line parameters")
    return np.column_stack([x, y, z])
