"""Core trajectory-consensus operations for DPC and KF-DPC.

Communication exchanges straight-line parameters ``[b, v, direction]``.
The current state ``[position, velocity]`` is reconstructed only after the
communication update; direction remains derived from Cartesian velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import backend
from .kalman import BatchCVKalman3D
from .trajectory import (
    TrajectoryPrediction,
    normalize_trajectory_directions,
    trajectory_state_at_time,
    trajectory_state_from_position_velocity,
)


def _validated_weight_matrix(weight_matrix: np.ndarray, node_count: int) -> np.ndarray:
    weights = np.asarray(weight_matrix, dtype=np.float64)
    if weights.shape != (node_count, node_count):
        raise ValueError(f"weight_matrix must have shape {(node_count, node_count)}")
    if np.any(weights < -1e-12):
        raise ValueError("weight_matrix cannot contain negative weights")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("weight_matrix must be row stochastic")
    return weights


def conservative_consensus_covariance(
    weight_matrix: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Mix posterior covariances without assuming independent estimates.

    Neighboring posteriors are correlated because they repeatedly exchange
    information.  A weighted covariance average is therefore used instead of
    the over-confident squared-weight formula.
    """
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("covariance must have shape (N, D, D)")
    weights = _validated_weight_matrix(weight_matrix, values.shape[0])
    mixed = np.einsum("ij,jkl->ikl", weights, values)
    return 0.5 * (mixed + np.swapaxes(mixed, 1, 2))


def independent_consensus_covariance(
    weight_matrix: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Propagate independent current-round trajectory measurements through W."""
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (6, 6):
        raise ValueError("covariance must have shape (N, 6, 6)")
    weights = _validated_weight_matrix(weight_matrix, values.shape[0])
    mixed = np.einsum("ij,jkl->ikl", weights**2, values)
    return 0.5 * (mixed + np.swapaxes(mixed, 1, 2))


def consensus_trajectory_parameters(
    local_parameters: np.ndarray,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object = None,
) -> np.ndarray:
    """Communicate complete straight-line trajectory parameters through ``W``."""
    parameters = np.asarray(local_parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 9:
        raise ValueError("trajectory parameters must have shape (N, 9)")
    _validated_weight_matrix(weight_matrix, parameters.shape[0])
    fused = backend.consensus_linear_accel(
        weight_matrix,
        weight_matrix_gpu,
        parameters,
        1,
    )
    return normalize_trajectory_directions(fused)


def consensus_trajectory_state(
    local_state: np.ndarray,
    time_s: float,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[position, velocity]`` to a trajectory, communicate, and predict."""
    states = np.asarray(local_state, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("local trajectory state must have shape (N, 6)")
    local_parameters = trajectory_state_from_position_velocity(
        states[:, :3],
        states[:, 3:],
        time_s,
    )
    fused_parameters = consensus_trajectory_parameters(
        local_parameters,
        weight_matrix,
        weight_matrix_gpu,
    )
    return trajectory_state_at_time(fused_parameters, time_s), fused_parameters


def dpc_state_update(
    local_trajectory: TrajectoryPrediction,
    time_s: float,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object = None,
    position_model_std: float = 0.0,
    velocity_model_std: float = 0.0,
) -> TrajectoryPrediction:
    """Apply DPC consensus to full UAV trajectory parameters ``[b, v]``."""
    local_state = np.asarray(local_trajectory.state, dtype=np.float64)
    if local_state.ndim != 2 or local_state.shape[1] != 6:
        raise ValueError("local trajectory state must have shape (N, 6)")
    weights = _validated_weight_matrix(weight_matrix, local_state.shape[0])
    if local_trajectory.trajectory_parameters is None:
        raise ValueError("DPC requires explicit local trajectory parameters")
    fused_parameters = consensus_trajectory_parameters(
        local_trajectory.trajectory_parameters,
        weights,
        weight_matrix_gpu,
    )
    state = trajectory_state_at_time(fused_parameters, time_s)
    covariance = independent_consensus_covariance(weights, local_trajectory.covariance)
    covariance[:, :3, :3] += max(float(position_model_std), 0.0) ** 2 * np.eye(3)
    covariance[:, 3:, 3:] += max(float(velocity_model_std), 0.0) ** 2 * np.eye(3)
    return TrajectoryPrediction(
        state=state,
        covariance=covariance,
        trajectory_parameters=fused_parameters,
    )


@dataclass(frozen=True)
class KFDPCStep:
    """Observable states from one KF-DPC round, before and after communication."""

    predicted_state: np.ndarray
    local_posterior_state: np.ndarray
    fused_posterior_state: np.ndarray
    fused_trajectory_parameters: np.ndarray


def kf_dpc_state_update(
    kf: BatchCVKalman3D,
    local_trajectory: TrajectoryPrediction,
    time_s: float,
    weight_matrix: np.ndarray,
    weight_matrix_gpu: object = None,
) -> KFDPCStep:
    """Run one persistent KF-DPC round in the required order.

    The order is strictly: predict from the previous posterior, update each
    node with its own current trajectory observation, then communicate the
    resulting UAV posterior through ``W``.
    """
    local_state = np.asarray(local_trajectory.state, dtype=np.float64)
    if local_state.shape != kf.x.shape:
        raise ValueError(f"local trajectory state must have shape {kf.x.shape}")
    _validated_weight_matrix(weight_matrix, kf.x.shape[0])

    kf.predict()
    predicted_state = kf.x.copy()
    kf.update_trajectory_measurement(local_state, local_trajectory.covariance)
    local_posterior_state = kf.x.copy()
    kf.x, fused_parameters = consensus_trajectory_state(
        local_posterior_state,
        time_s,
        weight_matrix,
        weight_matrix_gpu,
    )
    kf.P = conservative_consensus_covariance(weight_matrix, kf.P)
    return KFDPCStep(
        predicted_state,
        local_posterior_state,
        kf.x.copy(),
        fused_parameters,
    )


@dataclass
class ConsensusTracker:
    """Require spatial state agreement for several consecutive physical rounds."""

    position_tolerance_m: float
    velocity_tolerance_mps: float
    hold_steps: int = 3
    consecutive_steps: int = 0
    converged: bool = False

    def update(self, uav_states: np.ndarray) -> tuple[float, float, bool]:
        states = np.asarray(uav_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 6:
            raise ValueError("uav_states must have shape (N, 6)")
        center = np.mean(states, axis=0)
        position_error = float(np.max(np.linalg.norm(states[:, :3] - center[:3], axis=1)))
        velocity_error = float(np.max(np.linalg.norm(states[:, 3:] - center[3:], axis=1)))
        inside = (
            position_error <= max(float(self.position_tolerance_m), 0.0)
            and velocity_error <= max(float(self.velocity_tolerance_mps), 0.0)
        )
        self.consecutive_steps = self.consecutive_steps + 1 if inside else 0
        self.converged = self.converged or self.consecutive_steps >= max(int(self.hold_steps), 1)
        return position_error, velocity_error, self.converged


def velocity_directions(uav_states: np.ndarray) -> np.ndarray:
    """Derive unit flight directions from the consensus Cartesian velocities."""
    states = np.asarray(uav_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("uav_states must have shape (N, 6)")
    velocity = states[:, 3:]
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    direction = np.zeros_like(velocity)
    moving = speed[:, 0] > 1e-12
    direction[moving] = velocity[moving] / speed[moving]
    return direction


def spatial_phase_corrections(
    node_phases: np.ndarray,
    node_positions: np.ndarray,
    uav_states: np.ndarray,
    wave_number: float,
    reference_phase: float = 0.0,
) -> np.ndarray:
    """Compute per-node spatial phase corrections from a converged UAV state."""
    positions = np.asarray(node_positions, dtype=np.float64)
    phases = np.asarray(node_phases, dtype=np.float64)
    states = np.asarray(uav_states, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("node_positions must have shape (N, 3)")
    if phases.shape != (positions.shape[0],):
        raise ValueError("node_phases must have shape (N,)")
    if states.shape == (6,):
        uav_positions = np.broadcast_to(states[:3], positions.shape)
    elif states.shape == (positions.shape[0], 6):
        uav_positions = states[:, :3]
    else:
        raise ValueError("uav_states must have shape (6,) or (N, 6)")
    distance = np.linalg.norm(uav_positions - positions, axis=1)
    correction = float(reference_phase) - phases + float(wave_number) * distance
    return (correction + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class DPCOutput:
    """Outputs released after the distributed UAV-state estimate converges."""

    spatial_phases: np.ndarray
    uav_states: np.ndarray
    flight_directions: np.ndarray


def finalize_dpc_output(
    node_phases: np.ndarray,
    node_positions: np.ndarray,
    uav_states: np.ndarray,
    wave_number: float,
    converged: bool,
    reference_phase: float = 0.0,
) -> DPCOutput:
    """Calculate coherent-transmission phases only after state convergence."""
    if not converged:
        raise RuntimeError("DPC state has not converged; spatial phases are not ready")
    states = np.asarray(uav_states, dtype=np.float64)
    return DPCOutput(
        spatial_phases=spatial_phase_corrections(
            node_phases,
            node_positions,
            states,
            wave_number,
            reference_phase,
        ),
        uav_states=states.copy(),
        flight_directions=velocity_directions(states),
    )
