from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.constants import METHOD_DFPC, METHOD_KF_DFPC
from dfpc_experiment.experiment import consensus_covariance, run_experiment
from dfpc_experiment.kalman import make_cv3d_filter
from dfpc_experiment.scenario import current_truth, init_scene, observe_positions


class Kalman3DTests(unittest.TestCase):
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

    def test_uav_kf_waits_for_a_complete_trajectory_window(self) -> None:
        cfg = ExperimentConfig(
            N=24,
            T_long=3,
            K=4,
            mc_trials=1,
            device="cpu",
            Ts=0.2,
        )
        result = run_experiment(cfg)
        self.assertEqual(result["uav_kf_window_updates"], cfg.T_long)
        self.assertEqual(result["uav_kf_window_size"], cfg.K)
        np.testing.assert_allclose(
            result["metrics"][METHOD_KF_DFPC]["uav_rmse"][: cfg.K - 1],
            result["metrics"][METHOD_DFPC]["uav_rmse"][: cfg.K - 1],
            atol=1e-12,
        )

    def test_subgraph_mode_and_diagnostics_are_exposed_end_to_end(self) -> None:
        cfg = ExperimentConfig(
            N=40,
            n_clusters=4,
            T_long=2,
            K=2,
            mc_trials=1,
            device="cpu",
            cluster_mode="subgraph",
        )
        result = run_experiment(cfg)
        diagnostics = result["cluster_diagnostics"]
        self.assertEqual(result["args"]["cluster_mode"], "subgraph")
        self.assertTrue(np.all(diagnostics["effective_peer_count_mean"] >= 1.0))
        self.assertTrue(np.all(diagnostics["nodes_with_subgraph_peers"] > 0))


if __name__ == "__main__":
    unittest.main()
