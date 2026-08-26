from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.constants import METHOD_DFPC, METHOD_KF_DFPC
from dfpc_experiment.plots import metric_values_for_plot


class PlotTests(unittest.TestCase):
    def test_uav_kf_curve_starts_when_trajectory_initialization_is_ready(self) -> None:
        cfg = ExperimentConfig(T_long=2, K=4)
        values = np.arange(8, dtype=np.float64)

        plotted = metric_values_for_plot(METHOD_KF_DFPC, values, cfg)

        self.assertTrue(np.all(np.isnan(plotted[:3])))
        np.testing.assert_array_equal(plotted[3:], values[3:])
        np.testing.assert_array_equal(values, np.arange(8, dtype=np.float64))

    def test_non_uav_kf_curve_is_not_masked(self) -> None:
        cfg = ExperimentConfig(T_long=2, K=4)
        values = np.arange(8, dtype=np.float64)

        plotted = metric_values_for_plot(METHOD_DFPC, values, cfg)

        np.testing.assert_array_equal(plotted, values)


if __name__ == "__main__":
    unittest.main()
