from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.plots import causal_exponential_average


class PlotSmoothingTests(unittest.TestCase):
    def test_causal_ema_preserves_constant_series_and_input(self) -> None:
        values = np.full(21, 3.5)
        original = values.copy()
        np.testing.assert_allclose(causal_exponential_average(values, 11), values)
        np.testing.assert_array_equal(values, original)

    def test_causal_ema_ignores_nan_and_keeps_length(self) -> None:
        values = np.array([1.0, np.nan, 3.0, 5.0, 7.0])
        result = causal_exponential_average(values, 3)
        np.testing.assert_allclose(result, [1.0, 1.0, 2.0, 3.5, 5.25])
        self.assertEqual(result.shape, values.shape)

    def test_causal_ema_does_not_use_future_samples(self) -> None:
        original = np.array([1.0, 1.0, 1.0, 1.0])
        changed_future = np.array([1.0, 1.0, 100.0, 100.0])
        result = causal_exponential_average(original, 4)
        changed = causal_exponential_average(changed_future, 4)
        np.testing.assert_allclose(result[:2], changed[:2])
        self.assertEqual(result[0], original[0])
