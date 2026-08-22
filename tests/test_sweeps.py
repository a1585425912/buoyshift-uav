from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.constants import METHOD_CLUSTER_KF_DFPC
from dfpc_experiment.experiment import run_experiment
from dfpc_experiment.sweeps import (
    controlled_node_settings,
    convergence_summary,
    node_count_scaling_diagnostics,
    summarize_result,
)


class SweepTests(unittest.TestCase):
    def test_controlled_node_settings_hold_degree_and_cluster_size(self) -> None:
        for node_count in [20, 100, 2000]:
            settings = controlled_node_settings(node_count, 8.0, 10)
            expected_degree = 2.0 + (node_count - 3) * settings["global_connectivity"]
            self.assertAlmostEqual(expected_degree, 8.0)
            self.assertEqual(settings["n_clusters"], int(np.ceil(node_count / 10)))

    def test_summary_contains_subgraph_cluster_methods(self) -> None:
        result = run_experiment(
            ExperimentConfig(
                N=20,
                n_clusters=4,
                T_long=2,
                K=2,
                mc_trials=1,
                device="cpu",
                cluster_mode="subgraph",
            )
        )
        row = summarize_result(result, "node_count", 20.0, "N=20")
        for key in [
            "cluster_dfpc_tail_power_db",
            "cluster_kf_dfpc_tail_power_db",
            "node_kf_dfpc_tail_power_db",
            "cluster_dfpc_tail_node_rmse_m",
            "cluster_kf_dfpc_tail_node_rmse_m",
        ]:
            self.assertTrue(np.isfinite(row[key]), key)
        convergence = convergence_summary(result, METHOD_CLUSTER_KF_DFPC)
        self.assertGreaterEqual(convergence["convergence_step_05db"], 0)
        self.assertTrue(np.isfinite(convergence["transient_deficit_mean_db"]))

    def test_node_scaling_diagnostic_flags_linear_normalized_growth(self) -> None:
        rows = []
        for node_count in [100, 200, 300, 400]:
            power = node_count / 100.0
            gain = 20.0 * np.log10(node_count)
            rows.append(
                {
                    "factor_value": node_count,
                    "dfpc_tail_power_db": power,
                    "node_kf_dfpc_tail_power_db": power,
                    "kf_dfpc_tail_power_db": power,
                    "cluster_dfpc_tail_power_db": power,
                    "cluster_kf_dfpc_tail_power_db": power,
                    "dfpc_tail_gain_over_single_mean_db": gain,
                    "node_kf_dfpc_gain_over_single_mean_db": gain,
                    "kf_dfpc_gain_over_single_mean_db": gain,
                    "cluster_dfpc_gain_over_single_mean_db": gain,
                    "cluster_kf_dfpc_gain_over_single_mean_db": gain,
                }
            )
        diagnostics = node_count_scaling_diagnostics(rows)
        self.assertEqual(len(diagnostics), 5)
        self.assertTrue(all(row["suspicious_linear_growth"] for row in diagnostics))
        self.assertTrue(
            all(abs(row["gain_slope_error_from_20_db"]) < 1e-10 for row in diagnostics)
        )


if __name__ == "__main__":
    unittest.main()
