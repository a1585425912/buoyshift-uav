"""Nonlinear range filters for UAV state refinement after DPC.

Both filters track one UAV state per network node.  The DPC position estimate
is treated as a Cartesian pseudo-measurement and the physical UAV-to-buoy range
is treated with its nonlinear measurement function ``h(x)=||p-anchor||``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cv_matrices(dt: float, acceleration_std: float) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    f = np.eye(6, dtype=np.float64)
    f[:3, 3:] = dt * np.eye(3)
    g = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
    q = max(float(acceleration_std), 1e-12) ** 2 * (g @ g.T)
    return f, q


def _validate_positions(value: np.ndarray, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or (n is not None and array.shape[0] != n):
        expected = "(N, 3)" if n is None else f"({n}, 3)"
        raise ValueError(f"positions must have shape {expected}, got {array.shape}")
    return array


@dataclass
class BatchRangeEKF3D:
    """Batched constant-velocity EKF with Cartesian-prior and range updates."""

    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    position_prior_var: float
    range_var: float

    @property
    def positions(self) -> np.ndarray:
        return self.x[:, :3]

    @property
    def velocities(self) -> np.ndarray:
        return self.x[:, 3:]

    def predict(self) -> None:
        self.x = self.x @ self.F.T
        self.P = np.einsum("ij,njk,lk->nil", self.F, self.P, self.F) + self.Q[None, :, :]

    def update_position_prior(self, dfpc_positions_xyz: np.ndarray) -> None:
        z = _validate_positions(dfpc_positions_xyz, self.x.shape[0])
        h = np.zeros((3, 6), dtype=np.float64)
        h[:, :3] = np.eye(3)
        r = self.position_prior_var * np.eye(3)
        innovation = z - self.x[:, :3]
        s = self.P[:, :3, :3] + r[None, :, :]
        pht = self.P[:, :, :3]
        gain = np.einsum("nij,njk->nik", pht, np.linalg.inv(s))
        self.x += np.einsum("nij,nj->ni", gain, innovation)
        self._joseph_update(gain, h, r)

    def update_range(self, ranges_m: np.ndarray, anchors_xyz: np.ndarray) -> None:
        anchors = _validate_positions(anchors_xyz, self.x.shape[0])
        z = np.asarray(ranges_m, dtype=np.float64)
        if z.shape != (self.x.shape[0],):
            raise ValueError(f"ranges must have shape {(self.x.shape[0],)}, got {z.shape}")
        delta = self.x[:, :3] - anchors
        predicted = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        h = np.zeros((self.x.shape[0], 6), dtype=np.float64)
        h[:, :3] = delta / predicted[:, None]
        ph = np.einsum("nij,nj->ni", self.P, h)
        innovation_var = np.einsum("ni,ni->n", h, ph) + self.range_var
        gain = ph / np.maximum(innovation_var[:, None], 1e-12)
        self.x += gain * (z - predicted)[:, None]

        identity = np.eye(6, dtype=np.float64)[None, :, :]
        kh = np.einsum("ni,nj->nij", gain, h)
        ikh = identity - kh
        left = np.einsum("nij,njk->nik", ikh, self.P)
        self.P = np.einsum("nij,nkj->nik", left, ikh) + self.range_var * np.einsum(
            "ni,nj->nij", gain, gain
        )
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))

    def _joseph_update(self, gain: np.ndarray, h: np.ndarray, r: np.ndarray) -> None:
        identity = np.eye(6, dtype=np.float64)[None, :, :]
        ikh = identity - np.einsum("nij,jk->nik", gain, h)
        left = np.einsum("nij,njk->nik", ikh, self.P)
        joseph = np.einsum("nij,nkj->nik", left, ikh)
        kr = np.einsum("nij,jk->nik", gain, r)
        self.P = joseph + np.einsum("nij,nkj->nik", kr, gain)
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))


def make_range_ekf(
    initial_positions_xyz: np.ndarray,
    initial_velocities_xyz: np.ndarray,
    dt: float,
    position_prior_std: float,
    range_std: float,
    acceleration_std: float,
    initial_velocity_std: float,
) -> BatchRangeEKF3D:
    positions = _validate_positions(initial_positions_xyz)
    velocities = _validate_positions(initial_velocities_xyz, positions.shape[0])
    f, q = _cv_matrices(dt, acceleration_std)
    position_var = max(float(position_prior_std), 1e-6) ** 2
    velocity_var = max(float(initial_velocity_std), 1e-6) ** 2
    p0 = np.diag([position_var] * 3 + [velocity_var] * 3)
    return BatchRangeEKF3D(
        x=np.column_stack([positions, velocities]),
        P=np.broadcast_to(p0, (positions.shape[0], 6, 6)).copy(),
        F=f,
        Q=q,
        position_prior_var=position_var,
        range_var=max(float(range_std), 1e-6) ** 2,
    )


@dataclass
class BatchRangeParticleFilter3D:
    """Bootstrap particle filter with DPC-prior and nonlinear range weights."""

    particles: np.ndarray
    dt: float
    acceleration_std: float
    position_prior_std: float
    range_std: float
    _positions: np.ndarray
    _velocities: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    def predict(self, rng: np.random.Generator) -> None:
        accel = rng.normal(0.0, self.acceleration_std, size=self.particles[:, :, :3].shape)
        self.particles[:, :, :3] += self.dt * self.particles[:, :, 3:] + 0.5 * self.dt**2 * accel
        self.particles[:, :, 3:] += self.dt * accel

    def update(
        self,
        dfpc_positions_xyz: np.ndarray,
        ranges_m: np.ndarray,
        anchors_xyz: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        n, particle_count, _ = self.particles.shape
        dfpc = _validate_positions(dfpc_positions_xyz, n)
        anchors = _validate_positions(anchors_xyz, n)
        ranges = np.asarray(ranges_m, dtype=np.float64)
        if ranges.shape != (n,):
            raise ValueError(f"ranges must have shape {(n,)}, got {ranges.shape}")

        position_residual = self.particles[:, :, :3] - dfpc[:, None, :]
        predicted_ranges = np.linalg.norm(self.particles[:, :, :3] - anchors[:, None, :], axis=2)
        log_weights = -0.5 * np.sum(position_residual**2, axis=2) / self.position_prior_std**2
        log_weights -= 0.5 * (predicted_ranges - ranges[:, None]) ** 2 / self.range_std**2
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-300)
        self._positions = np.sum(weights[:, :, None] * self.particles[:, :, :3], axis=1)
        self._velocities = np.sum(weights[:, :, None] * self.particles[:, :, 3:], axis=1)

        # Per-node systematic resampling keeps memory bounded and deterministic
        # for a fixed RNG seed.
        cdf = np.cumsum(weights, axis=1)
        cdf[:, -1] = 1.0
        targets = (rng.random((n, 1)) + np.arange(particle_count)[None, :]) / particle_count
        indices = np.empty((n, particle_count), dtype=np.int64)
        for node in range(n):
            indices[node] = np.searchsorted(cdf[node], targets[node], side="left")
        self.particles = self.particles[np.arange(n)[:, None], indices]

    def apply_consensus_state(self, positions_xyz: np.ndarray, velocities_xyz: np.ndarray) -> None:
        positions = _validate_positions(positions_xyz, self.particles.shape[0])
        velocities = _validate_positions(velocities_xyz, self.particles.shape[0])
        self.particles[:, :, :3] += (positions - self._positions)[:, None, :]
        self.particles[:, :, 3:] += (velocities - self._velocities)[:, None, :]
        self._positions = positions.copy()
        self._velocities = velocities.copy()


def make_range_particle_filter(
    initial_positions_xyz: np.ndarray,
    initial_velocities_xyz: np.ndarray,
    dt: float,
    particle_count: int,
    position_prior_std: float,
    range_std: float,
    acceleration_std: float,
    initial_position_std: float,
    initial_velocity_std: float,
    rng: np.random.Generator,
) -> BatchRangeParticleFilter3D:
    positions = _validate_positions(initial_positions_xyz)
    velocities = _validate_positions(initial_velocities_xyz, positions.shape[0])
    if particle_count < 8:
        raise ValueError("particle_count must be at least 8")
    particles = np.empty((positions.shape[0], int(particle_count), 6), dtype=np.float64)
    particles[:, :, :3] = positions[:, None, :] + rng.normal(
        0.0, initial_position_std, size=particles[:, :, :3].shape
    )
    particles[:, :, 3:] = velocities[:, None, :] + rng.normal(
        0.0, initial_velocity_std, size=particles[:, :, 3:].shape
    )
    return BatchRangeParticleFilter3D(
        particles=particles,
        dt=float(dt),
        acceleration_std=max(float(acceleration_std), 1e-9),
        position_prior_std=max(float(position_prior_std), 1e-6),
        range_std=max(float(range_std), 1e-6),
        _positions=positions.copy(),
        _velocities=velocities.copy(),
    )
