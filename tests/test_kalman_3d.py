from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.constants import METHOD_KF_DFPC
from dfpc_experiment.experiment import run_experiment
from dfpc_experiment.kalman import make_cv3d_filter
from dfpc_experiment.scenario import current_truth, init_scene, observe_positions


class Kalman3DTests(unittest.TestCase):
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
        self.assertGreater(metrics["uav_velocity_rmse"][0], metrics["uav_velocity_rmse"][-1])


if __name__ == "__main__":
    unittest.main()
