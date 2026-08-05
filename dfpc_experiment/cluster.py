"""Communication-graph clustering and rotating intra-cluster estimation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


def communication_graph_clusters(
    weight_matrix: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Partition nodes by multi-source BFS over communication edges."""
    W = np.asarray(weight_matrix, dtype=np.float64)
    n = W.shape[0]
    k = int(np.clip(n_clusters, 1, n))
    adjacency = [
        np.flatnonzero((W[i] > 1e-12) & (np.arange(n) != i))
        for i in range(n)
    ]

    seeds = rng.choice(n, k, replace=False)
    labels = np.full(n, -1, dtype=np.int64)
    queues = [deque([int(seed)]) for seed in seeds]
    for cluster_id, seed in enumerate(seeds):
        labels[seed] = cluster_id

    unassigned = n - k
    while unassigned:
        progressed = False
        for cluster_id, queue in enumerate(queues):
            if not queue:
                continue
            node = queue.popleft()
            neighbors = adjacency[node].copy()
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                if labels[neighbor] >= 0:
                    continue
                labels[neighbor] = cluster_id
                queue.append(int(neighbor))
                unassigned -= 1
                progressed = True
                if not unassigned:
                    break
            if not unassigned:
                break

        if not progressed:
            counts = np.bincount(labels[labels >= 0], minlength=k)
            for node in np.flatnonzero(labels < 0):
                neighbor_labels = labels[adjacency[int(node)]]
                neighbor_labels = neighbor_labels[neighbor_labels >= 0]
                cluster_id = (
                    int(np.bincount(neighbor_labels, minlength=k).argmax())
                    if neighbor_labels.size
                    else int(np.argmin(counts))
                )
                labels[node] = cluster_id
                counts[cluster_id] += 1
                unassigned -= 1

    return labels


def rotating_cluster_position_correction(
    prior_positions: np.ndarray,
    true_positions_for_relative_measurement: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.8,
    relative_noise_std: float = 0.2,
) -> np.ndarray:
    """Fuse rotating peer votes for every target inside each cluster.

    This reproduces the legacy cluster mechanism.  Peer ``m`` votes for
    target ``q`` using its own prior plus a noisy relative displacement from
    ``m`` to ``q``.  The target prior and cluster vote are blended by alpha.
    """
    prior = np.asarray(prior_positions, dtype=np.float64)
    truth = np.asarray(true_positions_for_relative_measurement, dtype=np.float64)
    corrected = prior.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    sigma = max(float(relative_noise_std), 0.0)

    for cluster_id in np.unique(labels):
        ids = np.flatnonzero(labels == cluster_id)
        count = ids.size
        if count <= 2:
            continue
        local_prior = prior[ids]
        local_truth = truth[ids]
        noise = rng.normal(0.0, sigma, size=(count, count, prior.shape[1]))
        votes = (
            local_prior[:, None, :]
            + local_truth[None, :, :]
            - local_truth[:, None, :]
            + noise
        )
        mask = ~np.eye(count, dtype=bool)
        consensus = np.sum(np.where(mask[:, :, None], votes, 0.0), axis=0) / (count - 1)
        corrected[ids] = (1.0 - alpha) * local_prior + alpha * consensus

    return corrected


def rotating_cluster_trajectory_correction(
    prior_trajectory_states: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.8,
    trajectory_noise_std: float = 0.2,
) -> np.ndarray:
    """Fuse rotating peer trajectory votes for every target inside each cluster.

    Every node estimates the same UAV trajectory, so peers vote by sharing
    their own 9-column trajectory states directly.  The target prior and the
    intra-cluster peer consensus are blended by alpha.  This is the trajectory
    analogue of the legacy position correction: nodes inside the same
    communication cluster refine their UAV trajectory estimate using only
    intra-cluster traffic, before the final position prediction.
    """
    prior = np.asarray(prior_trajectory_states, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != 9:
        raise ValueError("prior_trajectory_states must have shape (N, 9)")
    corrected = prior.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    sigma = max(float(trajectory_noise_std), 0.0)

    for cluster_id in np.unique(labels):
        ids = np.flatnonzero(labels == cluster_id)
        count = ids.size
        if count <= 2:
            continue
        local_prior = prior[ids]
        noise = rng.normal(0.0, sigma, size=(count, count, 9))
        votes = local_prior[:, None, :] + noise
        mask = ~np.eye(count, dtype=bool)
        consensus = np.sum(np.where(mask[:, :, None], votes, 0.0), axis=0) / (count - 1)
        corrected[ids] = (1.0 - alpha) * local_prior + alpha * consensus

    return corrected


def cluster_trajectory_consensus(
    trajectory_states: np.ndarray,
    weight_matrix: np.ndarray,
    labels: np.ndarray,
    consensus_steps: int = 5,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run subgraph DPC consensus on UAV trajectory states.

    Unlike ``rotating_cluster_trajectory_correction``, this is a direct local
    DPC pass over each cluster subgraph.  The result is intended to be used as
    an extra prior for the global UAV trajectory estimator, not as the final
    fused trajectory.
    """
    states = np.asarray(trajectory_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 9:
        raise ValueError("trajectory_states must have shape (N, 9)")
    W = np.asarray(weight_matrix, dtype=np.float64)
    if W.shape != (states.shape[0], states.shape[0]):
        raise ValueError("weight_matrix must match trajectory_states")
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.shape != (states.shape[0],):
        raise ValueError("labels must have shape (N,)")
    steps = max(int(consensus_steps), 1)
    sigma = max(float(noise_std), 0.0)

    corrected = states.copy()
    for cluster_id in np.unique(labels_arr):
        ids = np.flatnonzero(labels_arr == cluster_id)
        count = ids.size
        if count < 2:
            continue
        local_W = W[np.ix_(ids, ids)].copy()
        local_W /= np.maximum(local_W.sum(axis=1, keepdims=True), 1e-12)
        local_states = corrected[ids].copy()
        if sigma > 0.0 and rng is not None:
            local_states += rng.normal(0.0, sigma, size=local_states.shape)
        for _ in range(steps):
            local_states = local_W @ local_states
        corrected[ids] = local_states
    return corrected


def cluster_node_localization(
    motion_positions: np.ndarray,
    labels: np.ndarray,
    true_positions_for_relative_measurement: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.8,
    relative_noise_std: float = 0.2,
    constraint_type: str = "vector",
    iterations: int = 5,
    dpc_prior_positions: np.ndarray | None = None,
    dpc_prior_weight: float = 0.0,
) -> np.ndarray:
    """Locate nodes inside each cluster with relative pilot constraints.

    The motion prior comes from each node's own filtered trajectory.  Inside
    every cluster, noisy pilot observations constrain the relative geometry:

    - ``vector``: peer ``j`` observes ``p_j - p_i`` plus Gaussian noise.
    - ``range``: peer ``j`` observes ``||p_j - p_i||`` plus Gaussian noise.

    The optional DPC prior is only a weak absolute anchor; by default the
    motion prior provides the absolute reference and the relative constraints
    refine the intra-cluster geometry.
    """
    motion = np.asarray(motion_positions, dtype=np.float64)
    truth = np.asarray(true_positions_for_relative_measurement, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != 3:
        raise ValueError("motion_positions must have shape (N, 3)")
    if truth.shape != motion.shape:
        raise ValueError("true_positions_for_relative_measurement must match motion_positions")
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.shape != (motion.shape[0],):
        raise ValueError("labels must have shape (N,)")
    if constraint_type not in {"vector", "range"}:
        raise ValueError("constraint_type must be 'vector' or 'range'")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    sigma = max(float(relative_noise_std), 0.0)
    iterations = max(int(iterations), 1)
    dpc_weight = max(float(dpc_prior_weight), 0.0)
    if dpc_prior_positions is not None:
        dpc = np.asarray(dpc_prior_positions, dtype=np.float64)
        if dpc.shape != motion.shape:
            raise ValueError("dpc_prior_positions must match motion_positions")
    else:
        dpc = None

    corrected = motion.copy()
    for cluster_id in np.unique(labels_arr):
        ids = np.flatnonzero(labels_arr == cluster_id)
        count = ids.size
        if count < 2:
            continue

        local_motion = motion[ids]
        local_truth = truth[ids]
        if dpc is not None:
            local_dpc = dpc[ids]
        else:
            local_dpc = local_motion

        local_corrected = _relative_constraint_consensus(
            local_motion,
            local_dpc,
            local_truth,
            rng,
            alpha,
            dpc_weight,
            sigma,
            constraint_type,
            iterations,
        )
        corrected[ids] = local_corrected

    return corrected


def _relative_constraint_consensus(
    motion: np.ndarray,
    dpc: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    dpc_weight: float,
    sigma: float,
    constraint_type: str,
    iterations: int,
) -> np.ndarray:
    """Fuse motion, DPC, and noisy pilot-relative peer votes."""
    count = motion.shape[0]
    motion_weight = max(1.0 - alpha, 1e-9)
    relative_weight = alpha
    normalizer = motion_weight + dpc_weight + relative_weight

    if constraint_type == "vector":
        relative = truth[None, :, :] - truth[:, None, :]
        relative += rng.normal(0.0, sigma, size=relative.shape)
    else:
        measured_ranges = np.linalg.norm(
            truth[:, None, :] - truth[None, :, :],
            axis=2,
        )
        measured_ranges += rng.normal(0.0, sigma, size=measured_ranges.shape)

    corrected = motion.copy()
    mask = ~np.eye(count, dtype=bool)
    for _ in range(iterations):
        if constraint_type == "vector":
            votes = corrected[:, None, :] + relative
        else:
            delta = corrected[None, :, :] - corrected[:, None, :]
            distance = np.maximum(np.linalg.norm(delta, axis=2, keepdims=True), 1e-12)
            directions = delta / distance
            votes = corrected[:, None, :] + directions * measured_ranges[:, :, None]
        consensus = np.sum(np.where(mask[:, :, None], votes, 0.0), axis=0) / (count - 1)
        corrected = (
            motion_weight * motion
            + dpc_weight * dpc
            + relative_weight * consensus
        ) / normalizer
    return corrected


@dataclass
class ClusterTrajectoryLocalizer:
    """Locate node trajectories with cluster-internal observer DPC.

    For each target node ``i`` in a cluster, every node in that cluster acts
    as an observer.  Observer ``j`` forms an estimate of target ``i``'s
    trajectory from its own trajectory plus the pilot-relative position
    ``p_i - p_j``.  A local DPC pass over the cluster subgraph makes all
    observer estimates of the same target agree, and the agreed trajectory is
    blended with the target's own trajectory prior by ``alpha``.
    """

    n: int
    _relative_history: np.ndarray = field(init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.n = int(self.n)
        if self.n < 1:
            raise ValueError("n must be positive")
        self._relative_history = np.zeros((self.n, self.n, 3), dtype=np.float64)

    def update(
        self,
        node_states: np.ndarray,
        weight_matrix: np.ndarray,
        labels: np.ndarray,
        true_positions_for_relative_measurement: np.ndarray,
        rng: np.random.Generator,
        alpha: float = 0.8,
        relative_noise_std: float = 0.2,
        constraint_type: str = "vector",
        consensus_steps: int = 5,
        dt: float = 1.0,
        blend_with_own: bool = True,
    ) -> np.ndarray:
        """Run one step of cluster-internal trajectory localization."""
        states = np.asarray(node_states, dtype=np.float64)
        truth = np.asarray(true_positions_for_relative_measurement, dtype=np.float64)
        if states.ndim != 2 or states.shape != (self.n, 6):
            raise ValueError(f"node_states must have shape ({self.n}, 6)")
        if truth.shape != (self.n, 3):
            raise ValueError(f"true_positions must have shape ({self.n}, 3)")
        W = np.asarray(weight_matrix, dtype=np.float64)
        if W.shape != (self.n, self.n):
            raise ValueError(f"weight_matrix must have shape ({self.n}, {self.n})")
        labels_arr = np.asarray(labels, dtype=np.int64)
        if labels_arr.shape != (self.n,):
            raise ValueError(f"labels must have shape ({self.n},)")
        if constraint_type not in {"vector", "range"}:
            raise ValueError("constraint_type must be 'vector' or 'range'")
        alpha = float(np.clip(alpha, 0.0, 1.0))
        sigma = max(float(relative_noise_std), 0.0)
        steps = max(int(consensus_steps), 1)
        step_dt = max(float(dt), 1e-12)

        corrected = states.copy()
        consensus_states = states.copy()
        for cluster_id in np.unique(labels_arr):
            ids = np.flatnonzero(labels_arr == cluster_id)
            count = ids.size
            if count < 2:
                continue

            local_states = states[ids]
            local_truth = truth[ids]
            local_W = W[np.ix_(ids, ids)].copy()
            local_W /= np.maximum(local_W.sum(axis=1, keepdims=True), 1e-12)

            if constraint_type == "vector":
                relative = local_truth[None, :, :] - local_truth[:, None, :]
                relative += rng.normal(0.0, sigma, size=relative.shape)
            else:
                measured_ranges = np.linalg.norm(
                    local_truth[:, None, :] - local_truth[None, :, :],
                    axis=2,
                )
                measured_ranges += rng.normal(0.0, sigma, size=measured_ranges.shape)
                delta = local_states[None, :, :3] - local_states[:, None, :3]
                distance = np.maximum(np.linalg.norm(delta, axis=2, keepdims=True), 1e-12)
                directions = delta / distance
                relative = directions * measured_ranges[:, :, None]

            if self._initialized:
                previous = self._relative_history[np.ix_(ids, ids)]
                relative_velocity = (relative - previous) / step_dt
            else:
                relative_velocity = np.zeros_like(relative)

            estimates = np.zeros((count, count, 6), dtype=np.float64)
            estimates[:, :, :3] = local_states[:, None, :3] + relative
            estimates[:, :, 3:] = local_states[:, None, 3:] + relative_velocity

            for target in range(count):
                observer_estimates = estimates[:, target, :].copy()
                # The target itself is not an observer.  Seed its row with the
                # mean of the other observers so the DPC iteration still runs
                # on the full cluster subgraph without injecting the target's
                # own trajectory into the observation consensus directly.
                observer_estimates[target] = (
                    np.sum(observer_estimates, axis=0) - observer_estimates[target]
                ) / max(count - 1, 1)
                for _ in range(steps):
                    observer_estimates = local_W @ observer_estimates
                consensus = observer_estimates[target]
                consensus_states[ids[target]] = consensus
                corrected[ids[target]] = (
                    (1.0 - alpha) * local_states[target]
                    + alpha * consensus
                )

            self._relative_history[np.ix_(ids, ids)] = relative
        self._initialized = True
        return corrected if blend_with_own else consensus_states


def cluster_node_trajectory_localization(
    node_states: np.ndarray,
    weight_matrix: np.ndarray,
    labels: np.ndarray,
    true_positions_for_relative_measurement: np.ndarray,
    rng: np.random.Generator,
    alpha: float = 0.8,
    relative_noise_std: float = 0.2,
    constraint_type: str = "vector",
    consensus_steps: int = 5,
    dt: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper for one-shot cluster trajectory localization."""
    states = np.asarray(node_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("node_states must have shape (N, 6)")
    localizer = ClusterTrajectoryLocalizer(states.shape[0])
    return localizer.update(
        states,
        weight_matrix,
        labels,
        true_positions_for_relative_measurement,
        rng,
        alpha=alpha,
        relative_noise_std=relative_noise_std,
        constraint_type=constraint_type,
        consensus_steps=consensus_steps,
        dt=dt,
    )
