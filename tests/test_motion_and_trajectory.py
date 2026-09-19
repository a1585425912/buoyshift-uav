from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.constants import METHOD_DPC
from dfpc_experiment.experiment import run_experiment
from dfpc_experiment.scenario import (
    advance_truth_one_long_block,
    advance_truth_one_short_step,
    geometry_terms,
    init_scene,
)
from dfpc_experiment.trajectory import (
    consensus_independent_covariance,
    init_line_estimator,
    normalize_trajectory_directions,
    predict_positions,
    predict_trajectory,
    trajectory_directions,
    trajectory_state_from_position_velocity,
    trajectory_velocities,
    update_local_line_estimates,
)


class MotionModelTests(unittest.TestCase):
    def test_default_scene_represents_moderate_current_without_base_station(self) -> None:
        cfg = ExperimentConfig()
        self.assertAlmostEqual(cfg.fc_mhz, 20.0)
        self.assertEqual(cfg.warmup_steps, 0)
        self.assertAlmostEqual(cfg.buoy_wave_speed, 0.3)
        self.assertAlmostEqual(cfg.buoy_random_displacement_std, 0.03)
        self.assertAlmostEqual(cfg.buoy_center_accumulation_ratio, 0.05)
        self.assertAlmostEqual(cfg.buoy_kf_accel_std, 0.0)
        self.assertAlmostEqual(cfg.buoy_kf_initial_velocity_std, 0.0)
        self.assertAlmostEqual(cfg.uav_obs_noise, 3.0)
        self.assertAlmostEqual(cfg.buoy_center_obs_noise, 1.0)

    def test_default_geometry_uses_unit_path_loss(self) -> None:
        cfg = ExperimentConfig(N=2, tx_power=5.0)
        _, amp, p_ideal, p_single_mean, p_single_best = geometry_terms(
            cfg,
            np.zeros(3),
            np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            k_const=1.0,
        )
        expected_amplitude = np.sqrt(cfg.tx_power)
        np.testing.assert_allclose(amp, expected_amplitude, atol=1e-12)
        self.assertAlmostEqual(p_ideal, (2.0 * expected_amplitude) ** 2)
        self.assertAlmostEqual(p_single_mean, cfg.tx_power)
        self.assertAlmostEqual(p_single_best, cfg.tx_power)

    def test_center_moves_exactly_with_fixed_current(self) -> None:
        cfg = ExperimentConfig(
            N=400,
            device="cpu",
            buoy_wave_speed=5.0,
            buoy_wave_heading_deg=0.0,
            buoy_random_displacement_std=0.0,
        )
        rng = np.random.default_rng(7)
        state = init_scene(cfg, rng)
        before = state.buoy_center_true.copy()
        advance_truth_one_short_step(cfg, state, rng, dt=0.2)
        np.testing.assert_allclose(
            state.buoy_center_true - before,
            np.tile([1.0, 0.0, 0.0], (cfg.N, 1)),
            atol=1e-12,
        )

    def test_position_offset_is_resampled_around_current_center(self) -> None:
        cfg = ExperimentConfig(
            N=1_000,
            device="cpu",
            buoy_wave_speed=0.0,
            buoy_random_displacement_std=0.4,
            buoy_center_accumulation_ratio=0.0,
        )
        rng = np.random.default_rng(17)
        state = init_scene(cfg, rng)
        previous_offset = state.buoy_offset_true.copy()
        advance_truth_one_short_step(cfg, state, rng, dt=0.2)
        self.assertFalse(np.array_equal(state.buoy_offset_true, previous_offset))
        np.testing.assert_allclose(np.mean(state.buoy_offset_true, axis=0), [0.0, 0.0, 0.0], atol=0.04)
        np.testing.assert_allclose(np.std(state.buoy_offset_true[:, :2], axis=0), [0.4, 0.4], atol=0.03)
        correlation = np.corrcoef(previous_offset[:, 0], state.buoy_offset_true[:, 0])[0, 1]
        self.assertLess(abs(correlation), 0.1)
        np.testing.assert_allclose(state.buoy_offset_true[:, 2], 0.0, atol=0.0)

    def test_random_displacement_partly_accumulates_into_center(self) -> None:
        cfg = ExperimentConfig(
            N=100,
            device="cpu",
            buoy_wave_speed=0.0,
            buoy_random_displacement_std=0.4,
            buoy_center_accumulation_ratio=0.25,
        )
        rng = np.random.default_rng(23)
        state = init_scene(cfg, rng)
        previous_center = state.buoy_center_true.copy()
        advance_truth_one_short_step(cfg, state, rng, dt=0.2)
        center_shift = state.buoy_center_true - previous_center
        np.testing.assert_allclose(
            center_shift,
            state.buoy_offset_true * (0.25 / 0.75),
            atol=1e-12,
        )

    def test_long_block_advances_exactly_K_short_steps(self) -> None:
        cfg = ExperimentConfig(
            N=4,
            device="cpu",
            TL=1.1,
            Ts=0.3,
            K=4,
            uav_speed=20.0,
            heading_deg=0.0,
            buoy_wave_speed=0.0,
            buoy_random_displacement_std=0.0,
        )
        rng = np.random.default_rng(11)
        state = init_scene(cfg, rng)
        before = state.uav_center_true.copy()
        advance_truth_one_long_block(cfg, state, rng)
        np.testing.assert_allclose(state.uav_center_true - before, [24.0, 0.0, 0.0], atol=1e-12)


class TrajectoryEstimatorTests(unittest.TestCase):
    def test_sliding_window_drops_the_oldest_observation(self) -> None:
        estimator = init_line_estimator(1, window_size=3)
        update_local_line_estimates(estimator, 0.0, np.array([[100.0, 0.0, 0.0]]))
        params = None
        for time_s in [1.0, 2.0, 3.0]:
            xyz = np.array([[time_s, 2.0 * time_s, 120.0]])
            params = update_local_line_estimates(estimator, time_s, xyz)
        assert params is not None
        self.assertEqual(estimator.count, 3)
        np.testing.assert_allclose(params[0, :6], [0.0, 1.0, 0.0, 2.0, 120.0, 0.0], atol=1e-12)

    def test_exact_straight_line_is_recovered(self) -> None:
        n = 5
        state = init_line_estimator(n)
        params = None
        for time_s in [0.0, 1.0, 2.0, 3.0]:
            xyz = np.tile([-10.0 + 4.0 * time_s, 3.0 - 2.0 * time_s, 120.0 + time_s], (n, 1))
            params = update_local_line_estimates(state, time_s, xyz)
        assert params is not None
        expected_line = np.tile([-10.0, 4.0, 3.0, -2.0, 120.0, 1.0], (n, 1))
        np.testing.assert_allclose(params[:, :6], expected_line, atol=1e-12)
        expected_direction = np.tile(np.array([4.0, -2.0, 1.0]) / np.sqrt(21.0), (n, 1))
        np.testing.assert_allclose(trajectory_directions(params), expected_direction, atol=1e-12)
        np.testing.assert_allclose(trajectory_velocities(params), np.tile([4.0, -2.0, 1.0], (n, 1)))
        predicted = predict_positions(params, 4.0, 120.0)
        np.testing.assert_allclose(predicted, np.tile([6.0, -5.0, 124.0], (n, 1)), atol=1e-12)

    def test_direction_is_renormalized_after_consensus(self) -> None:
        states = np.zeros((2, 9), dtype=np.float64)
        states[:, [1, 3, 5]] = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
        states[:, 6:] = [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]]
        normalized = normalize_trajectory_directions(states)
        np.testing.assert_allclose(np.linalg.norm(normalized[:, 6:], axis=1), 1.0)
        np.testing.assert_allclose(normalized[1, 6:], [0.0, 1.0, 0.0])

    def test_trajectory_prediction_contains_velocity_and_covariance(self) -> None:
        n = 3
        estimator = init_line_estimator(n)
        params = None
        for time_s in [0.0, 1.0, 2.0]:
            xyz = np.tile([2.0 + 3.0 * time_s, -1.0 + time_s, 120.0], (n, 1))
            params = update_local_line_estimates(estimator, time_s, xyz)
        assert params is not None

        prediction = predict_trajectory(
            params,
            estimator,
            time_s=3.0,
            measurement_std=2.0,
            initial_velocity_std=10.0,
        )
        np.testing.assert_allclose(
            prediction.state,
            np.tile([11.0, 2.0, 120.0, 3.0, 1.0, 0.0], (n, 1)),
            atol=1e-12,
        )
        self.assertEqual(prediction.covariance.shape, (n, 6, 6))
        np.testing.assert_allclose(
            prediction.covariance,
            np.swapaxes(prediction.covariance, 1, 2),
            atol=1e-12,
        )
        self.assertLess(prediction.covariance[0, 3, 3], 10.0**2)

        W = np.full((n, n), 1.0 / n)
        mixed = consensus_independent_covariance(W, prediction.covariance)
        np.testing.assert_allclose(mixed, prediction.covariance / n, atol=1e-12)

    def test_trajectory_state_from_position_velocity(self) -> None:
        position = np.tile([10.0, 20.0, 120.0], (3, 1))
        velocity = np.tile([5.0, -1.0, 2.0], (3, 1))
        states = trajectory_state_from_position_velocity(position, velocity, time_s=2.0)
        np.testing.assert_allclose(states[:, [0, 2, 4]], np.tile([0.0, 22.0, 116.0], (3, 1)), atol=1e-12)
        np.testing.assert_allclose(states[:, [1, 3, 5]], velocity, atol=1e-12)
        np.testing.assert_allclose(
            states[:, 6:],
            np.tile(velocity[0] / np.linalg.norm(velocity[0]), (3, 1)),
            atol=1e-12,
        )

    def test_small_end_to_end_experiment(self) -> None:
        cfg = ExperimentConfig(
            N=16,
            T_long=3,
            K=2,
            TL=1.0,
            Ts=0.3,
            mc_trials=1,
            device="cpu",
            buoy_wave_speed=5.0,
        )
        result = run_experiment(cfg)
        self.assertEqual(result["total_steps"], 6)
        self.assertAlmostEqual(result["physical_duration_s"], 1.5)
        for method_metrics in result["metrics"].values():
            self.assertTrue(np.all(np.isfinite(method_metrics["gain_linear"])))
        velocity_rmse = result["metrics"][METHOD_DPC]["uav_velocity_rmse"]
        self.assertLess(velocity_rmse[-1], velocity_rmse[0])


if __name__ == "__main__":
    unittest.main()
