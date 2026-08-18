from __future__ import annotations

import unittest

import numpy as np

from dfpc_experiment.cluster import (
    ClusterTrajectoryLocalizer,
    _doubly_stochastic_submatrix,
    cluster_trajectory_consensus,
    cluster_node_localization,
    communication_graph_clusters,
    rotating_cluster_position_correction,
    rotating_cluster_trajectory_correction,
)
from dfpc_experiment.sim_core import metropolis_hastings_W


class ClusterTests(unittest.TestCase):
    def test_doubly_stochastic_submatrix_preserves_row_and_column_sums(self) -> None:
        W = np.array(
            [
                [0.5, 0.25, 0.25, 0.0],
                [0.25, 0.5, 0.0, 0.25],
                [0.25, 0.0, 0.5, 0.25],
                [0.0, 0.25, 0.25, 0.5],
            ],
            dtype=float,
        )
        local = _doubly_stochastic_submatrix(W, np.array([0, 1, 2]))
        np.testing.assert_allclose(local.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(local.sum(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(local, local.T, atol=1e-12)

    def test_graph_partition_assigns_every_node(self) -> None:
        W = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5],
            ],
            dtype=float,
        )
        labels = communication_graph_clusters(W, 2, np.random.default_rng(3))
        self.assertEqual(labels.shape, (4,))
        self.assertTrue(np.all(labels >= 0))
        self.assertEqual(np.unique(labels).size, 2)

    def test_rotating_votes_preserve_consistent_relative_geometry(self) -> None:
        truth = np.column_stack([np.arange(6.0), np.zeros(6), np.zeros(6)])
        prior = truth + np.array([1.0, -0.5, 0.25])
        labels = np.zeros(6, dtype=int)
        corrected = rotating_cluster_position_correction(
            prior,
            truth,
            labels,
            np.random.default_rng(4),
            alpha=1.0,
            relative_noise_std=0.0,
        )
        np.testing.assert_allclose(corrected, prior, atol=1e-12)

    def test_rotating_trajectory_votes_preserve_prior_with_alpha_one(self) -> None:
        states = np.tile(np.array([1.0, 20.0, 2.0, -1.0, 3.0, 0.5, 0.8, 0.0, 0.6]), (6, 1))
        labels = np.zeros(6, dtype=int)
        corrected = rotating_cluster_trajectory_correction(
            states,
            labels,
            np.random.default_rng(5),
            alpha=1.0,
            trajectory_noise_std=0.0,
        )
        np.testing.assert_allclose(corrected, states, atol=1e-12)

    def test_rotating_trajectory_votes_are_finite_and_bounded(self) -> None:
        base = np.array([1.0, 20.0, 2.0, -1.0, 3.0, 0.5, 0.8, 0.0, 0.6])
        states = np.tile(base, (24, 1))
        rng = np.random.default_rng(6)
        states[:, [0, 2, 4]] += rng.normal(0.0, 0.1, size=(24, 3))
        states[:, 6:] = states[:, 1:4] / np.linalg.norm(states[:, 1:4], axis=1, keepdims=True)
        labels = np.repeat([0, 1, 2], 8)
        corrected = rotating_cluster_trajectory_correction(
            states,
            labels,
            np.random.default_rng(7),
            alpha=0.8,
            trajectory_noise_std=0.1,
        )
        self.assertTrue(np.all(np.isfinite(corrected)))
        self.assertLess(np.max(np.abs(corrected - states)), 1.0)

    def test_rotating_trajectory_votes_use_other_peers_only(self) -> None:
        states = np.zeros((6, 9), dtype=np.float64)
        states[:, 0] = [0.0, 2.0, 4.0, 10.0, 20.0, 30.0]
        states[:, 1] = 10.0
        states[:, 6] = 1.0
        labels = np.array([0, 0, 0, 1, 1, 1])
        corrected = rotating_cluster_trajectory_correction(
            states,
            labels,
            np.random.default_rng(8),
            alpha=1.0,
            trajectory_noise_std=0.0,
        )
        np.testing.assert_allclose(corrected[:3, 0], [3.0, 2.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(corrected[3:, 0], [25.0, 20.0, 15.0], atol=1e-12)

    def test_cluster_node_localization_vector_constraint_reduces_error(self) -> None:
        rng = np.random.default_rng(20)
        truth = np.column_stack([np.arange(20.0), np.sin(np.arange(20.0)), np.zeros(20)])
        motion = truth + rng.normal(0.0, 1.0, size=(20, 3))
        labels = np.repeat([0, 1, 2, 3], 5)
        corrected = cluster_node_localization(
            motion,
            labels,
            truth,
            np.random.default_rng(21),
            alpha=0.8,
            relative_noise_std=0.0,
            constraint_type="vector",
        )
        self.assertLess(
            float(np.sqrt(np.mean(np.sum((corrected - truth) ** 2, axis=1)))),
            float(np.sqrt(np.mean(np.sum((motion - truth) ** 2, axis=1)))),
        )

    def test_cluster_node_localization_range_constraint_reduces_relative_error(self) -> None:
        rng = np.random.default_rng(22)
        truth = np.column_stack([np.arange(20.0), np.sin(np.arange(20.0)), np.zeros(20)])
        motion = truth + rng.normal(0.0, 1.0, size=(20, 3))
        labels = np.repeat([0, 1, 2, 3], 5)
        corrected = cluster_node_localization(
            motion,
            labels,
            truth,
            np.random.default_rng(23),
            alpha=0.8,
            relative_noise_std=0.0,
            constraint_type="range",
            iterations=10,
        )
        truth_ranges = np.linalg.norm(truth[:, None, :] - truth[None, :, :], axis=2)
        motion_ranges = np.linalg.norm(motion[:, None, :] - motion[None, :, :], axis=2)
        corrected_ranges = np.linalg.norm(corrected[:, None, :] - corrected[None, :, :], axis=2)
        self.assertLess(
            float(np.sqrt(np.mean((corrected_ranges - truth_ranges) ** 2))),
            float(np.sqrt(np.mean((motion_ranges - truth_ranges) ** 2))),
        )

    def test_cluster_node_localization_vector_constraint_recovers_relative_geometry(self) -> None:
        truth = np.column_stack([np.arange(8.0), np.zeros(8), np.zeros(8)])
        motion = truth + np.array([2.0, -1.0, 0.5])
        corrected = cluster_node_localization(
            motion,
            np.zeros(8, dtype=int),
            truth,
            np.random.default_rng(24),
            alpha=0.9,
            relative_noise_std=0.0,
            constraint_type="vector",
            iterations=1,
        )
        np.testing.assert_allclose(
            corrected[:, None, :] - corrected[None, :, :],
            truth[:, None, :] - truth[None, :, :],
            atol=1e-9,
        )

    def test_cluster_trajectory_localizer_vector_reduces_position_error(self) -> None:
        rng = np.random.default_rng(30)
        n = 36
        W = metropolis_hastings_W(n, 0.08, rng)
        labels = np.repeat(np.arange(3), n // 3)
        truth_pos = np.column_stack(
            [np.arange(n, dtype=float), np.sin(np.arange(n) * 0.4), np.zeros(n)]
        )
        truth_vel = np.tile([1.0, 0.1, 0.0], (n, 1))
        prior_pos = truth_pos + rng.normal(0.0, 1.0, size=(n, 3))
        prior_vel = truth_vel + rng.normal(0.0, 0.2, size=(n, 3))
        states = np.column_stack([prior_pos, prior_vel])
        localizer = ClusterTrajectoryLocalizer(n)
        corrected = localizer.update(
            states,
            W,
            labels,
            truth_pos,
            np.random.default_rng(31),
            alpha=0.8,
            relative_noise_std=0.0,
            constraint_type="vector",
            consensus_steps=5,
            dt=0.05,
        )
        self.assertLess(
            float(np.sqrt(np.mean(np.sum((corrected[:, :3] - truth_pos) ** 2, axis=1)))),
            float(np.sqrt(np.mean(np.sum((prior_pos - truth_pos) ** 2, axis=1)))),
        )

    def test_cluster_trajectory_localizer_keeps_observer_consensus_finite(self) -> None:
        rng = np.random.default_rng(32)
        n = 24
        W = metropolis_hastings_W(n, 0.06, rng)
        labels = np.repeat(np.arange(4), 6)
        truth_pos = np.column_stack(
            [np.arange(n, dtype=float), np.cos(np.arange(n) * 0.3), np.zeros(n)]
        )
        states = np.column_stack([truth_pos, np.tile([0.5, -0.2, 0.0], (n, 1))])
        localizer = ClusterTrajectoryLocalizer(n)
        corrected = localizer.update(
            states,
            W,
            labels,
            truth_pos,
            np.random.default_rng(33),
            alpha=0.8,
            relative_noise_std=0.3,
            constraint_type="range",
            consensus_steps=3,
            dt=0.05,
        )
        self.assertTrue(np.all(np.isfinite(corrected)))

    def test_cluster_trajectory_consensus_reduces_local_estimate_variance(self) -> None:
        rng = np.random.default_rng(34)
        n = 36
        W = metropolis_hastings_W(n, 0.08, rng)
        labels = np.repeat(np.arange(3), n // 3)
        base = np.tile(np.array([1.0, 20.0, 2.0, -1.0, 3.0, 0.5, 0.8, 0.0, 0.6]), (n, 1))
        states = base + rng.normal(0.0, 0.2, size=base.shape)
        prior_std = np.std(states, axis=0)
        consensus = cluster_trajectory_consensus(
            states,
            W,
            labels,
            consensus_steps=8,
            noise_std=0.0,
        )
        consensus_std = np.std(consensus, axis=0)
        self.assertTrue(np.all(consensus_std < prior_std))
        self.assertTrue(np.all(np.isfinite(consensus)))


if __name__ == "__main__":
    unittest.main()
