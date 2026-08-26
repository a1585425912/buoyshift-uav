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


def gaussian_position_offsets(
    n: int,
    position_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """独立采样相对当前位移中心的二维零均值位置偏移。"""
    sigma = max(float(position_std), 0.0)
    return rng.normal(0.0, sigma, size=(n, 2))
