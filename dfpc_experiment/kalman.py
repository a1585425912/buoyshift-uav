"""用于 UAV 与浮标位置跟踪的批量三维恒速 Kalman 滤波器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatchCVKalman3D:
    """并行跟踪多个三维目标，状态排列为 [x,y,z,vx,vy,vz]。"""

    x: np.ndarray
    P: np.ndarray
    F: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self.x[:, :3]

    @property
    def velocities(self) -> np.ndarray:
        return self.x[:, 3:]

    def predict(self) -> None:
        self.x = self.x @ self.F.T
        self.P = np.einsum("ij,njk,lk->nil", self.F, self.P, self.F) + self.Q[None, :, :]

    def update(self, measurements_xyz: np.ndarray) -> None:
        z = np.asarray(measurements_xyz, dtype=np.float64)
        if z.shape != (self.x.shape[0], 3):
            raise ValueError(f"measurements must have shape {(self.x.shape[0], 3)}, got {z.shape}")
        self._update_linear_measurement(z, self.H, self.R)

    def _update_linear_measurement(
        self,
        measurements: np.ndarray,
        observation_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> None:
        """Update from a shared linear observation model and per-node covariance."""
        z = np.asarray(measurements, dtype=np.float64)
        H = np.asarray(observation_matrix, dtype=np.float64)
        if H.ndim != 2 or H.shape[1] != 6:
            raise ValueError("observation_matrix must have shape (M, 6)")
        expected_shape = (self.x.shape[0], H.shape[0])
        if z.shape != expected_shape:
            raise ValueError(f"measurements must have shape {expected_shape}, got {z.shape}")

        covariance = np.asarray(measurement_covariance, dtype=np.float64)
        if covariance.shape == (H.shape[0], H.shape[0]):
            covariance = np.broadcast_to(
                covariance,
                (self.x.shape[0], H.shape[0], H.shape[0]),
            )
        elif covariance.shape != (self.x.shape[0], H.shape[0], H.shape[0]):
            raise ValueError(
                "measurement_covariance must be shared (M, M) or batched (N, M, M)"
            )
        covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))

        innovation = z - self.x @ H.T
        innovation_cov = np.einsum("ij,njk,lk->nil", H, self.P, H) + covariance
        pht = np.einsum("nij,kj->nik", self.P, H)
        gain = np.einsum("nij,njk->nik", pht, np.linalg.inv(innovation_cov))
        self.x = self.x + np.einsum("nij,nj->ni", gain, innovation)

        # Joseph 形式能在长时间和高频 sweep 中更好地保持协方差对称半正定。
        identity = np.eye(6, dtype=np.float64)[None, :, :]
        ikh = identity - np.einsum("nij,jk->nik", gain, H)
        left = np.einsum("nij,njk->nik", ikh, self.P)
        joseph = np.einsum("nij,nkj->nik", left, ikh)
        kr = np.einsum("nij,njk->nik", gain, covariance)
        self.P = joseph + np.einsum("nij,nkj->nik", kr, gain)
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))

    def update_position_measurement(
        self,
        positions_xyz: np.ndarray,
        measurement_std: float,
    ) -> None:
        """Update the position block with a custom pseudo-measurement."""
        z = np.asarray(positions_xyz, dtype=np.float64)
        if z.shape != (self.x.shape[0], 3):
            raise ValueError(f"positions must have shape {(self.x.shape[0], 3)}, got {z.shape}")
        r = max(float(measurement_std), 1e-9) ** 2
        self._update_linear_measurement(z, self.H, r * np.eye(3))

    def update_position_covariance(
        self,
        positions_xyz: np.ndarray,
        measurement_covariances: np.ndarray,
    ) -> None:
        """Update positions using a shared or per-node 3-D covariance."""
        z = np.asarray(positions_xyz, dtype=np.float64)
        if z.shape != (self.x.shape[0], 3):
            raise ValueError(f"positions must have shape {(self.x.shape[0], 3)}, got {z.shape}")
        self._update_linear_measurement(z, self.H, measurement_covariances)

    def update_trajectory_measurement(
        self,
        trajectory_states: np.ndarray,
        measurement_covariances: np.ndarray,
    ) -> None:
        """Fuse full UAV trajectory states [px,py,pz,vx,vy,vz]."""
        states = np.asarray(trajectory_states, dtype=np.float64)
        if states.shape != self.x.shape:
            raise ValueError(f"trajectory_states must have shape {self.x.shape}, got {states.shape}")
        self._update_linear_measurement(states, np.eye(6), measurement_covariances)


def make_cv3d_filter(
    initial_positions_xyz: np.ndarray,
    dt: float,
    measurement_std: float,
    acceleration_std: float,
    initial_velocity_std: float,
    initial_velocity_xyz: np.ndarray | None = None,
    position_diffusion: float = 0.0,
    velocity_propagate: bool = True,
) -> BatchCVKalman3D:
    """由首帧三维位置观测构造恒速 KF；z 轴保留但使用较小过程噪声。

    velocity_propagate=False 时转移矩阵不把速度积分到位置上（F 的位置—速度耦合块置零），
    用于短时间尺度 KF：block 内 UAV 块中心不随速度前进，只跟踪局部抖动。
    """
    positions = np.asarray(initial_positions_xyz, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("initial_positions_xyz must have shape (N, 3)")
    n = positions.shape[0]
    step = float(dt)
    if step <= 0.0:
        raise ValueError("dt must be positive")

    if initial_velocity_xyz is None:
        velocity = np.zeros((n, 3), dtype=np.float64)
    else:
        velocity = np.broadcast_to(np.asarray(initial_velocity_xyz, dtype=np.float64), (n, 3)).copy()
    x = np.column_stack([positions, velocity])

    F = np.eye(6, dtype=np.float64)
    if velocity_propagate:
        F[:3, 3:] = step * np.eye(3)
    H = np.zeros((3, 6), dtype=np.float64)
    H[:, :3] = np.eye(3)

    axis_scale = np.diag([1.0, 1.0, 0.05])
    G = np.vstack([0.5 * step**2 * axis_scale, step * axis_scale])
    Q = max(float(acceleration_std), 0.0) ** 2 * (G @ G.T)
    diffusion_scale = np.diag([1.0, 1.0, 0.05])
    Q[:3, :3] += (
        max(float(position_diffusion), 0.0) ** 2 * step * (diffusion_scale @ diffusion_scale.T)
    )
    measurement_var = max(float(measurement_std), 1e-9) ** 2
    R = measurement_var * np.eye(3)
    initial_var = max(float(initial_velocity_std), 1e-9) ** 2
    P0 = np.diag([measurement_var] * 3 + [initial_var] * 3)
    P = np.broadcast_to(P0, (n, 6, 6)).copy()
    return BatchCVKalman3D(x=x, P=P, F=F, H=H, Q=Q, R=R)


def predict_n_steps(kf: BatchCVKalman3D, steps: int) -> None:
    """Propagate a Kalman filter by ``steps`` fixed-dt steps."""
    for _ in range(max(int(steps), 0)):
        kf.predict()


def inject_position_state(
    kf: BatchCVKalman3D,
    positions_xyz: np.ndarray,
    position_std: float,
) -> None:
    """Overwrite the position state and set a matching covariance block.

    The DPC consensus estimate is treated as a position pseudo-measurement of
    the given standard deviation.  The position-velocity cross covariance is
    cleared because the old cross terms no longer describe the injected state;
    keeping them makes P indefinite and corrupts later velocity updates.
    """
    kf.x[:, :3] = np.asarray(positions_xyz, dtype=np.float64)
    var = max(float(position_std), 1e-9) ** 2
    kf.P[:, :3, :3] = var * np.eye(3)
    kf.P[:, :3, 3:] = 0.0
    kf.P[:, 3:, :3] = 0.0


def copy_velocity_state(
    target: BatchCVKalman3D,
    source: BatchCVKalman3D,
) -> None:
    """Copy velocity mean and covariance from the long-timescale filter.

    The old position-velocity cross covariance of ``target`` is cleared
    because it describes a different velocity posterior.
    """
    target.x[:, 3:] = source.x[:, 3:].copy()
    target.P[:, 3:, 3:] = source.P[:, 3:, 3:].copy()
    target.P[:, :3, 3:] = 0.0
    target.P[:, 3:, :3] = 0.0
