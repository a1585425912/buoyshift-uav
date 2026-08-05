from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.debug_tools import DebugRecorder, parse_index_filter
from dfpc_experiment.kalman import copy_velocity_state, inject_position_state, make_cv3d_filter, predict_n_steps


class DebugToolsTests(unittest.TestCase):
    def test_negative_iteration_maps_to_last_real_index(self) -> None:
        self.assertEqual(parse_index_filter("-1", 5), {4})
        self.assertEqual(parse_index_filter("0,-1", 5), {0, 4})
        self.assertEqual(parse_index_filter("99,-3", 5), {2})

    def test_from_config_uses_K_valid_iteration_range(self) -> None:
        cfg = ExperimentConfig(debug=True, K=5)
        recorder = DebugRecorder.from_config(cfg)
        self.assertEqual(recorder.iter_filter, {0, 4})
        self.assertTrue(recorder.should_capture(0, 0, 4))
        self.assertFalse(recorder.should_capture(0, 0, 5))


class KalmanConsistencyTests(unittest.TestCase):
    def test_predict_n_steps_is_safe_for_zero(self) -> None:
        kf = make_cv3d_filter(np.zeros((2, 3)), dt=0.1, measurement_std=1.0, acceleration_std=1.0, initial_velocity_std=1.0)
        before = kf.x.copy()
        predict_n_steps(kf, 0)
        np.testing.assert_allclose(kf.x, before)

    def test_position_injection_syncs_covariance(self) -> None:
        kf = make_cv3d_filter(np.zeros((3, 3)), dt=0.1, measurement_std=1.0, acceleration_std=1.0, initial_velocity_std=1.0)
        kf.predict()
        self.assertGreater(np.max(np.abs(kf.P[:, :3, 3:])), 0.0)
        inject_position_state(kf, np.ones((3, 3)), position_std=0.7)
        np.testing.assert_allclose(kf.x[:, :3], np.ones((3, 3)))
        expected = 0.7**2 * np.eye(3)
        np.testing.assert_allclose(kf.P[:, :3, :3], np.broadcast_to(expected, (3, 3, 3)))
        np.testing.assert_allclose(kf.P[:, :3, 3:], 0.0)
        np.testing.assert_allclose(kf.P[:, 3:, :3], 0.0)
        np.testing.assert_array_equal(np.linalg.eigvalsh(kf.P[0]) > 0, True)

    def test_velocity_copy_syncs_covariance(self) -> None:
        short = make_cv3d_filter(np.zeros((2, 3)), dt=0.1, measurement_std=1.0, acceleration_std=1.0, initial_velocity_std=1.0)
        long = make_cv3d_filter(np.zeros((2, 3)), dt=0.2, measurement_std=2.0, acceleration_std=1.0, initial_velocity_std=4.0)
        short.predict()
        copy_velocity_state(short, long)
        np.testing.assert_allclose(short.x[:, 3:], long.x[:, 3:])
        np.testing.assert_allclose(short.P[:, 3:, 3:], long.P[:, 3:, 3:])
        np.testing.assert_allclose(short.P[:, :3, 3:], 0.0)
        np.testing.assert_allclose(short.P[:, 3:, :3], 0.0)


if __name__ == "__main__":
    unittest.main()
