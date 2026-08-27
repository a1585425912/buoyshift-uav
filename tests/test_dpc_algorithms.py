from __future__ import annotations

import copy
import unittest

import numpy as np

from dfpc_experiment.dpc import (
    ConsensusTracker,
    dpc_state_update,
    finalize_dpc_output,
    kf_dpc_state_update,
    spatial_phase_corrections,
    velocity_directions,
)
from dfpc_experiment.kalman import make_cv3d_filter
from dfpc_experiment.trajectory import TrajectoryPrediction


class DPCAlgorithmTests(unittest.TestCase):
    def test_dpc_exchanges_position_and_velocity_state_through_w(self) -> None:
        states = np.array(
            [
                [0.0, 1.0, 120.0, 2.0, 0.0, 0.0],
                [10.0, 3.0, 120.0, 0.0, 4.0, 0.0],
            ]
        )
        covariance = np.stack([np.eye(6), 4.0 * np.eye(6)])
        local = TrajectoryPrediction(states, covariance)
        weights = np.array([[0.75, 0.25], [0.5, 0.5]])

        fused = dpc_state_update(local, weights)

        np.testing.assert_allclose(fused.state, weights @ states)
        expected_covariance = np.einsum("ij,jkl->ikl", weights**2, covariance)
        np.testing.assert_allclose(fused.covariance, expected_covariance)
        np.testing.assert_allclose(
            velocity_directions(fused.state),
            fused.velocities / np.linalg.norm(fused.velocities, axis=1, keepdims=True),
        )

    def test_kf_dpc_updates_locally_before_communication(self) -> None:
        weights = np.array([[0.8, 0.2], [0.35, 0.65]], dtype=float)
        initial_positions = np.array([[0.0, 0.0, 120.0], [8.0, 0.0, 120.0]])
        kf = make_cv3d_filter(
            initial_positions,
            dt=0.5,
            measurement_std=2.0,
            acceleration_std=0.1,
            initial_velocity_std=5.0,
        )
        local_states = np.array(
            [
                [2.0, 1.0, 120.0, 4.0, 2.0, 0.0],
                [12.0, -2.0, 120.0, 1.0, -4.0, 0.0],
            ]
        )
        local_covariance = np.stack([0.1 * np.eye(6), 8.0 * np.eye(6)])
        observation = TrajectoryPrediction(local_states, local_covariance)
        manual = copy.deepcopy(kf)
        manual.predict()
        predicted = manual.x.copy()
        manual.update_trajectory_measurement(local_states, local_covariance)
        local_posterior = manual.x.copy()
        expected_fused = weights @ local_posterior

        step = kf_dpc_state_update(kf, observation, weights)

        np.testing.assert_allclose(step.predicted_state, predicted)
        np.testing.assert_allclose(step.local_posterior_state, local_posterior)
        np.testing.assert_allclose(step.fused_posterior_state, expected_fused)
        np.testing.assert_allclose(kf.x, expected_fused)

    def test_convergence_means_network_agreement_for_consecutive_rounds(self) -> None:
        tracker = ConsensusTracker(0.2, 0.1, hold_steps=2)
        agreed = np.array(
            [
                [10.0, 0.0, 120.0, 4.0, 0.0, 0.0],
                [10.1, 0.0, 120.0, 4.02, 0.0, 0.0],
            ]
        )
        self.assertFalse(tracker.update(agreed)[2])
        self.assertTrue(tracker.update(agreed + np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0]))[2])
        self.assertTrue(tracker.converged)

    def test_spatial_phase_uses_input_node_phase_and_estimated_geometry(self) -> None:
        node_phases = np.array([0.2, -0.3])
        node_positions = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        uav_state = np.array([3.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        wave_number = 0.5

        corrections = spatial_phase_corrections(
            node_phases,
            node_positions,
            uav_state,
            wave_number,
        )

        distances = np.array([3.0, np.sqrt(13.0)])
        expected = -node_phases + wave_number * distances
        expected = (expected + np.pi) % (2.0 * np.pi) - np.pi
        np.testing.assert_allclose(corrections, expected)

        with self.assertRaisesRegex(RuntimeError, "has not converged"):
            finalize_dpc_output(
                node_phases,
                node_positions,
                np.tile(uav_state, (2, 1)),
                wave_number,
                converged=False,
            )
        output = finalize_dpc_output(
            node_phases,
            node_positions,
            np.tile(uav_state, (2, 1)),
            wave_number,
            converged=True,
        )
        np.testing.assert_allclose(output.spatial_phases, expected)
        np.testing.assert_allclose(output.flight_directions, [[1.0, 0.0, 0.0]] * 2)


if __name__ == "__main__":
    unittest.main()
