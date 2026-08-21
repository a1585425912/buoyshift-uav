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
        dfpc = result["metrics"][METHOD_DFPC]
        kf_dfpc = result["metrics"][METHOD_KF_DFPC]
        self.assertGreater(np.mean(kf_dfpc["gain_linear"][tail]), np.mean(dfpc["gain_linear"][tail]))
        self.assertLess(np.mean(kf_dfpc["uav_rmse"][tail]), 1.0)

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
        # The KF is initialized from the true velocity plus a small noise, so
        # its first-step velocity error is already small.  The meaningful
        # convergence property is that the steady-state velocity estimate beats
        # the naive DPC trajectory estimate, not that the first step is worse
        # than the last step.
        dfpc_initial_velocity_rmse = result["metrics"][METHOD_DFPC]["uav_velocity_rmse"][0]
        self.assertLess(metrics["uav_velocity_rmse"][-1], dfpc_initial_velocity_rmse)


if __name__ == "__main__":
    unittest.main()
