"""数值后端工具。

这个文件只放和计算后端有关的底层函数：
1. 选择 CPU / GPU；
2. 加速共识矩阵乘法；
3. 生成和限制节点短时间尺度位移。

如果你后续想把共识从 CuPy 改成 PyTorch，主要改这里。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import sim_core as sim

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional acceleration path
    cp = None


def resolve_device(cfg: Any) -> str:
    """根据配置和本机环境决定实际使用 CPU 还是 CUDA。"""
    requested = getattr(cfg, "device", "auto")
    cuda_ok = False
    if cp is not None:
        try:
            cuda_ok = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            cuda_ok = False
    if requested == "cuda" and not cuda_ok:
        print("CUDA requested but CuPy CUDA is unavailable; falling back to CPU consensus.", flush=True)
        return "cpu"
    if requested == "auto":
        return "cuda" if cuda_ok else "cpu"
    return requested


def make_gpu_consensus_matrix(W: np.ndarray, cfg: Any) -> Any:
    """把共识矩阵 W 放到 GPU 上。

    如果当前使用 CPU，则返回 None；上层函数会自动走 CPU 分支。
    """
    device = resolve_device(cfg)
    if device != "cuda":
        return None
    dtype = cp.float64 if getattr(cfg, "gpu_dtype", "float32") == "float64" else cp.float32
    return cp.asarray(W, dtype=dtype)


def consensus_linear_accel(W_cpu: np.ndarray, W_gpu: Any, X: np.ndarray, steps: int) -> np.ndarray:
    """执行线性共识 `X <- W @ X`。

    X 可以是 UAV 位置估计矩阵，形状通常是 N x 3。
    steps 表示连续做几次共识更新；当前真实迭代模型中每个 iteration 做 1 次。
    """
    if W_gpu is None:
        return sim.consensus_linear(W_cpu, X, steps)
    Y = cp.asarray(X, dtype=W_gpu.dtype)
    for _ in range(max(steps, 0)):
        Y = W_gpu @ Y
    return cp.asnumpy(Y).astype(np.float64, copy=False)


def gaussian_drift_displacement(
    n: int,
    mean_velocity: np.ndarray,
    diffusion: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """采样漂移—扩散位移 N(mean_velocity*dt, diffusion^2*dt*I)。"""
    mean = np.asarray(mean_velocity, dtype=np.float64)
    if mean.shape != (2,):
        raise ValueError("mean_velocity must have shape (2,)")
    safe_dt = max(float(dt), 0.0)
    mean_displacement = mean * safe_dt
    displacement_std = max(float(diffusion), 0.0) * np.sqrt(safe_dt)
    return rng.normal(mean_displacement, displacement_std, size=(n, 2))


def gaussian_short_offset_step(
    n: int,
    sigma_per_sqrt_second: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成零均值小尺度位移；标准差按 sqrt(dt) 缩放以保持时间步长一致性。"""
    sigma = max(float(sigma_per_sqrt_second), 0.0) * np.sqrt(max(float(dt), 0.0))
    return rng.normal(0.0, sigma, size=(n, 2))


def clip_offsets(offset: np.ndarray, radius: float) -> np.ndarray:
    """把节点短时间尺度偏移限制在半径 radius 内，避免随机游走无限发散。"""
    if radius <= 0.0:
        return offset
    norms = np.linalg.norm(offset, axis=1)
    over = norms > radius
    if np.any(over):
        clipped = offset.copy()
        clipped[over] *= (radius / np.maximum(norms[over], 1e-12))[:, None]
        return clipped
    return offset
