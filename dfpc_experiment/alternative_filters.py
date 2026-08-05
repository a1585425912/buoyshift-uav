"""Lightweight position filters used to compare filtering choices in DPC.

All filters expose the same batched interface.  They consume one ``(N, 3)``
position sample per physical short-time step and expose filtered positions and
finite-difference/model velocities.  Keeping these filters in their own module
lets the DPC simulation remain responsible only for producing observations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class BatchPositionFilter(Protocol):
    """Common interface implemented by every alternative filter."""

    @property
    def positions(self) -> np.ndarray: ...

    @property
    def velocities(self) -> np.ndarray: ...

    def update(self, measurements_xyz: np.ndarray) -> np.ndarray: ...


def _positions(value: np.ndarray, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {array.shape}")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"positions must have shape {expected_shape}, got {array.shape}")
    return array


@dataclass
class ExponentialMovingAverage3D:
    """First-order exponential smoother, ``x = alpha*z + (1-alpha)*x``."""

    _positions: np.ndarray
    dt: float
    alpha: float
    _velocities: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._positions = _positions(self._positions).copy()
        self._velocities = np.zeros_like(self._positions)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    def update(self, measurements_xyz: np.ndarray) -> np.ndarray:
        sample = _positions(measurements_xyz, self._positions.shape)
        previous = self._positions.copy()
        self._positions = self.alpha * sample + (1.0 - self.alpha) * previous
        self._velocities = (self._positions - previous) / self.dt
        return self._positions


@dataclass
class WindowPositionFilter3D:
    """Causal moving mean or moving median over the most recent samples."""

    _positions: np.ndarray
    dt: float
    window: int
    statistic: str
    _velocities: np.ndarray = field(init=False)
    _history: deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self._positions = _positions(self._positions).copy()
        self._velocities = np.zeros_like(self._positions)
        self._history = deque(maxlen=int(self.window))
        self._history.append(self._positions.copy())
        if self.statistic not in {"mean", "median"}:
            raise ValueError("statistic must be 'mean' or 'median'")

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    def update(self, measurements_xyz: np.ndarray) -> np.ndarray:
        sample = _positions(measurements_xyz, self._positions.shape)
        previous = self._positions.copy()
        self._history.append(sample.copy())
        stack = np.stack(tuple(self._history), axis=0)
        self._positions = (
            np.mean(stack, axis=0) if self.statistic == "mean" else np.median(stack, axis=0)
        )
        self._velocities = (self._positions - previous) / self.dt
        return self._positions


@dataclass
class AlphaBetaFilter3D:
    """Constant-velocity alpha-beta tracker for batched 3-D positions."""

    _positions: np.ndarray
    dt: float
    alpha: float
    beta: float
    _velocities: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._positions = _positions(self._positions).copy()
        self._velocities = np.zeros_like(self._positions)

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    def update(self, measurements_xyz: np.ndarray) -> np.ndarray:
        sample = _positions(measurements_xyz, self._positions.shape)
        predicted = self._positions + self.dt * self._velocities
        residual = sample - predicted
        self._positions = predicted + self.alpha * residual
        self._velocities = self._velocities + (self.beta / self.dt) * residual
        return self._positions


def make_position_filter(
    name: str,
    initial_positions_xyz: np.ndarray,
    dt: float,
    *,
    ema_alpha: float = 0.35,
    window: int = 5,
    alpha_beta_alpha: float = 0.85,
    alpha_beta_beta: float = 0.05,
) -> BatchPositionFilter:
    """Create an alternative filter by its stable command-line name."""
    initial = _positions(initial_positions_xyz)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if name == "ema":
        return ExponentialMovingAverage3D(initial, dt, ema_alpha)
    if name == "moving_average":
        return WindowPositionFilter3D(initial, dt, window, "mean")
    if name == "median":
        return WindowPositionFilter3D(initial, dt, window, "median")
    if name == "alpha_beta":
        return AlphaBetaFilter3D(initial, dt, alpha_beta_alpha, alpha_beta_beta)
    raise ValueError(f"unknown alternative filter: {name}")
