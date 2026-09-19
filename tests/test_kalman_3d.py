from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.constants import (
    METHOD_DFPC,
    METHOD_KF_DFPC,
    METHOD_NODE_KF_DPC,
    METHOD_NO_ALG,
    METHOD_SHORE_BROADCAST,
)
from dfpc_experiment.experiment import consensus_covariance, run_experiment
from dfpc_experiment.kalman import make_cv3d_filter
from dfpc_experiment.scenario import current_truth, init_scene, observe_positions


class Kalman3DTests(unittest.TestCase):
    def test_position_update_accepts_per_node_measurement_covariance(self) -> None:
        kf = make_cv3d_filter(
            np.zeros((2, 3)),
            dt=0.1,
            measurement_std=1.0,
            acceleration_std=0.0,
            initial_velocity_std=1.0,
        )
        measurements = np.full((2, 3), 10.0)
        covariance = np.stack([np.eye(3), 100.0 * np.eye(3)])

        kf.update_position_covariance(measurements, covariance)

        self.assertGreater(kf.positions[0, 0], kf.positions[1, 0])
        self.assertTrue(np.all(np.isfinite(kf.P)))

    def test_consensus_covariance_is_conservative_for_unknown_correlation(self) -> None:
        W = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
        P = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ],
            dtype=float,
        )
        expected = np.array(
            [
                [[1.5, 0.0], [0.0, 1.5]],
                [[1.5, 0.0], [0.0, 1.5]],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(consensus_covariance(W, P), expected, atol=1e-12)

    def test_paper_scale_kf_consensus_does_not_become_overconfident(self) -> None:
        cfg = ExperimentConfig(
            N=100,
            fc_mhz=30.0,
            T_long=4,
            K=10,
            mc_trials=1,
            device="cpu",
        )
        result = run_experiment(cfg)
        tail = slice(int(0.8 * result["total_steps"]), None)
        dpc = result["metrics"][METHOD_DFPC]
        kf_dpc = result["metrics"][METHOD_KF_DFPC]
        self.assertGreater(np.mean(kf_dpc["gain_linear"][tail]), np.mean(dpc["gain_linear"][tail]))
        self.assertLess(np.mean(kf_dpc["uav_rmse"][tail]), 1.0)

    def test_noiseless_constant_velocity_prediction_in_three_dimensions(self) -> None:
        initial = np.array([[1.0, 2.0, 3.0], [-2.0, 4.0, 1.0]])
        velocity = np.array([4.0, -1.0, 0.25])
        kf = make_cv3d_filter(
            initial,
            dt=0.5,
            measurement_std=0.1,
            acceleration_std=0.0,
            initial_velocity_std=0.1,
            initial_velocity_xyz=velocity,
        )
        kf.predict()
        expected = initial + 0.5 * velocity
        np.testing.assert_allclose(kf.positions, expected, atol=1e-12)
        kf.update(expected)
        np.testing.assert_allclose(kf.positions, expected, atol=1e-12)
        np.testing.assert_allclose(kf.P, np.swapaxes(kf.P, 1, 2), atol=1e-12)

    def test_full_trajectory_measurement_updates_velocity_directly(self) -> None:
        n = 2
        trajectory = np.tile([5.0, -2.0, 120.0, 4.0, -1.0, 0.25], (n, 1))
        kf = make_cv3d_filter(
            trajectory[:, :3],
            dt=0.5,
            measurement_std=1.0,
            acceleration_std=0.0,
            initial_velocity_std=10.0,
        )
        covariance = np.broadcast_to(0.01 * np.eye(6), (n, 6, 6)).copy()
        kf.update_trajectory_measurement(trajectory, covariance)

        np.testing.assert_allclose(kf.positions, trajectory[:, :3], atol=1e-12)
        np.testing.assert_allclose(kf.velocities, trajectory[:, 3:], atol=1e-3)
        np.testing.assert_allclose(kf.P, np.swapaxes(kf.P, 1, 2), atol=1e-12)

    def test_scene_observations_and_truth_are_three_dimensional(self) -> None:
        cfg = ExperimentConfig(N=8, device="cpu")
        scene_rng = np.random.default_rng(1)
        obs_rng = np.random.default_rng(2)
        state = init_scene(cfg, scene_rng)
        uav_true, buoy_true = current_truth(state)
        uav_obs, buoy_obs = observe_positions(cfg, uav_true, buoy_true, obs_rng)
        self.assertEqual(buoy_true.shape, (cfg.N, 3))
        self.assertEqual(uav_obs.shape, (cfg.N, 3))
        self.assertEqual(buoy_obs.shape, (cfg.N, 3))
        np.testing.assert_allclose(buoy_true[:, 2], 0.0)

    def test_kf_dfpc_branch_produces_finite_three_dimensional_metrics(self) -> None:
        cfg = ExperimentConfig(N=20, T_long=5, K=2, mc_trials=1, device="cpu", TL=1.0, Ts=0.2)
        result = run_experiment(cfg)
        metrics = result["metrics"][METHOD_KF_DFPC]
        for key in ["gain_linear", "distance_rmse", "node_rmse", "uav_rmse", "uav_velocity_rmse"]:
            self.assertTrue(np.all(np.isfinite(metrics[key])), key)
        dpc_initial_velocity_rmse = result["metrics"][METHOD_DFPC]["uav_velocity_rmse"][0]
        self.assertLess(metrics["uav_velocity_rmse"][-1], dpc_initial_velocity_rmse)

    def test_uav_kf_initializes_at_step_zero_then_consumes_each_new_observation(self) -> None:
        cfg = ExperimentConfig(
            N=24,
            T_long=3,
            K=4,
            mc_trials=1,
            device="cpu",
            Ts=0.2,
        )
        result = run_experiment(cfg)
        total_steps = cfg.T_long * cfg.K
        self.assertEqual(result["uav_kf_window_updates"], 1)
        self.assertEqual(result["uav_kf_trajectory_initializations"], 1)
        self.assertEqual(result["uav_kf_position_updates"], total_steps - 1)
        self.assertEqual(result["uav_kf_initialization_step"], 0)
        self.assertEqual(result["uav_kf_window_size"], 1)
        np.testing.assert_allclose(
            result["metrics"][METHOD_KF_DFPC]["uav_rmse"][:1],
            result["metrics"][METHOD_DFPC]["uav_rmse"][:1],
            atol=1e-12,
        )
        self.assertLess(
            result["metrics"][METHOD_DFPC]["uav_velocity_rmse"][cfg.K],
            cfg.uav_speed,
        )

    def test_shared_warmup_is_consumed_before_formal_step_zero(self) -> None:
        cfg = ExperimentConfig(
            N=24,
            T_long=2,
            K=3,
            warmup_steps=5,
            mc_trials=1,
            device="cpu",
            Ts=0.2,
        )
        result = run_experiment(cfg)
        total_steps = cfg.T_long * cfg.K
        self.assertEqual(result["uav_kf_trajectory_initializations"], 1)
        self.assertEqual(
            result["uav_kf_position_updates"],
            total_steps + cfg.warmup_steps - 1,
        )
        self.assertEqual(result["uav_kf_initialization_step"], -cfg.warmup_steps)
        self.assertFalse(
            np.allclose(
                result["metrics"][METHOD_KF_DFPC]["uav_rmse"][0],
                result["metrics"][METHOD_DFPC]["uav_rmse"][0],
            )
        )

    def test_dpc_kf_variants_are_exposed_as_algorithms(self) -> None:
        cfg = ExperimentConfig(
            N=40,
            T_long=2,
            K=2,
            mc_trials=1,
            device="cpu",
        )
        result = run_experiment(cfg)
        self.assertEqual(set(result["metrics"]), set(result["methods"]))
        self.assertEqual(
            set(result["methods"]),
            {METHOD_NO_ALG, METHOD_DFPC, METHOD_NODE_KF_DPC, METHOD_KF_DFPC},
        )
        np.testing.assert_allclose(
            result["metrics"][METHOD_NODE_KF_DPC]["uav_rmse"],
            result["metrics"][METHOD_DFPC]["uav_rmse"],
        )
        self.assertLess(
            np.mean(result["metrics"][METHOD_NODE_KF_DPC]["node_rmse"]),
            np.mean(result["metrics"][METHOD_DFPC]["node_rmse"]),
        )
        self.assertTrue(np.all(np.isnan(result["metrics"][METHOD_NO_ALG]["uav_rmse"])))
        self.assertNotIn("cluster_diagnostics", result)
        self.assertFalse(any("Cluster" in method or "Subgraph" in method for method in result["methods"]))

    def test_ideal_shore_broadcast_uses_current_positions(self) -> None:
        cfg = ExperimentConfig(
            N=32,
            T_long=2,
            K=3,
            mc_trials=1,
            device="cpu",
            system_phase_std_deg=0.0,
            shore_broadcast_enabled=True,
            shore_broadcast_uav_noise=0.0,
            shore_broadcast_node_noise=0.0,
        )
        result = run_experiment(cfg)
        metrics = result["metrics"][METHOD_SHORE_BROADCAST]
        self.assertIn(METHOD_SHORE_BROADCAST, result["methods"])
        np.testing.assert_allclose(metrics["distance_rmse"], 0.0, atol=1e-12)
        np.testing.assert_allclose(metrics["node_rmse"], 0.0, atol=1e-12)
        np.testing.assert_allclose(metrics["uav_rmse"], 0.0, atol=1e-12)
        np.testing.assert_allclose(metrics["norm_db"], 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
