import unittest

import numpy as np

from dfpc_experiment.metrics import add_power_reference_metrics, random_phase_reference_metrics
from dfpc_experiment.sim_core import (
    apply_polarity_correction,
    effective_phase_errors,
    phase_rmse,
    power_from_theta,
    sample_phase_std,
    wrap_to_pi,
)


class RandomPhaseReferenceTests(unittest.TestCase):
    def test_iid_uniform_phase_statistics(self) -> None:
        n = 200_000
        rng = np.random.default_rng(12345)
        phi_true = rng.uniform(-np.pi, np.pi, size=n)
        result = random_phase_reference_metrics(
            phi_true=phi_true,
            amp=np.ones(n),
            eps=np.zeros(n),
            p_ideal=float(n**2),
            p_single_mean=1.0,
            p_single_best=1.0,
        )

        theoretical_std_deg = 90.0 / np.sqrt(3.0)
        self.assertAlmostEqual(
            result["phase_std_deg"], theoretical_std_deg, delta=0.35
        )

    def test_reference_uses_scene_phase_without_redrawing(self) -> None:
        n = 64
        seed = 77
        amp = np.linspace(0.5, 1.5, n)
        rng = np.random.default_rng(seed)
        phi_true = rng.uniform(-np.pi, np.pi, size=n)
        eps = rng.normal(0.0, 0.02, size=n)
        residual = effective_phase_errors(np.zeros_like(phi_true), phi_true, eps)
        expected_power = np.abs(np.sum(amp * np.exp(1j * residual))) ** 2

        result = random_phase_reference_metrics(
            phi_true=phi_true,
            amp=amp,
            eps=eps,
            p_ideal=float(np.sum(amp) ** 2),
            p_single_mean=float(np.mean(amp**2)),
            p_single_best=float(np.max(amp**2)),
        )

        self.assertAlmostEqual(result["power_linear"], expected_power, places=12)

    def test_default_power_path_applies_ideal_polarity(self) -> None:
        phi_true = np.deg2rad(np.array([120.0, 170.0, -135.0]))
        amp = np.ones(3)
        actual = power_from_theta(np.zeros(3), phi_true, amp, np.zeros(3))
        residual = effective_phase_errors(np.zeros(3), phi_true, np.zeros(3))
        expected = np.abs(np.sum(np.exp(1j * residual))) ** 2
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertTrue(np.all(residual >= -np.pi / 2.0))
        self.assertTrue(np.all(residual < np.pi / 2.0))

    def test_normalized_power_is_distinct_from_single_node_gain(self) -> None:
        n = 100
        coherent_fraction = 0.8
        result = add_power_reference_metrics(
            {}, coherent_fraction * n**2, float(n**2), 1.0, 1.0
        )
        self.assertAlmostEqual(result["gain_linear"], coherent_fraction)
        self.assertAlmostEqual(
            result["gain_over_single_mean_db"],
            result["norm_db"] + 20.0 * np.log10(n),
            places=12,
        )

    def test_phase_rmse_counts_common_phase_bias(self) -> None:
        residual = np.deg2rad(np.array([10.0, 10.0, 10.0]))
        self.assertAlmostEqual(np.rad2deg(sample_phase_std(residual)), 0.0, places=12)
        self.assertAlmostEqual(np.rad2deg(phase_rmse(residual)), 10.0, places=12)

    def test_polarity_correction_folds_instead_of_clips(self) -> None:
        raw_deg = np.array([120.0, 170.0, -135.0, 89.0, -90.0])
        expected_deg = np.array([-60.0, -10.0, 45.0, 89.0, -90.0])
        actual_deg = np.rad2deg(apply_polarity_correction(np.deg2rad(raw_deg)))
        np.testing.assert_allclose(actual_deg, expected_deg, atol=1e-12)
        self.assertTrue(np.all(actual_deg >= -90.0))
        self.assertTrue(np.all(actual_deg < 90.0))


if __name__ == "__main__":
    unittest.main()
