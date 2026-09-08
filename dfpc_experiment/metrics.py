"""功率、相位误差和定位误差指标。

这个文件负责“怎么评价一次实验结果”：
1. 相对于理想多节点相干功率的归一化功率；
2. 相对于单节点发射的增益；
3. 相位残差标准差；
4. 距离、节点位置、UAV 位置 RMSE。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import sim_core as sim


def add_power_reference_metrics(
    result: dict[str, float],
    power: float,
    p_ideal: float,
    p_single_mean: float,
    p_single_best: float,
) -> dict[str, float]:
    """把同一个接收功率 power 转换成多种评价指标。

    p_ideal 是所有节点理想相干时的功率，所以 norm_db 越接近 0 dB 越好。
    p_single_mean / p_single_best 用于回答“比单个节点发射多多少增益”。
    """
    gain_linear = power / max(p_ideal, 1e-30)
    gain_single_mean = power / max(p_single_mean, 1e-30)
    gain_single_best = power / max(p_single_best, 1e-30)
    result.update(
        {
            "gain_linear": float(gain_linear),
            "norm_db": float(10.0 * np.log10(max(gain_linear, 1e-30))),
            "power_linear": float(power),
            "ideal_power_linear": float(p_ideal),
            "single_node_mean_power_linear": float(p_single_mean),
            "single_node_best_power_linear": float(p_single_best),
            "gain_over_single_mean_linear": float(gain_single_mean),
            "gain_over_single_mean_db": float(10.0 * np.log10(max(gain_single_mean, 1e-30))),
            "gain_over_single_best_linear": float(gain_single_best),
            "gain_over_single_best_db": float(10.0 * np.log10(max(gain_single_best, 1e-30))),
        }
    )
    return result


def random_phase_reference_metrics(
    phi_true: np.ndarray,
    amp: np.ndarray,
    eps: np.ndarray,
    p_ideal: float,
    p_single_mean: float,
    p_single_best: float,
) -> dict[str, float]:
    """随机相位理论参考：所有节点 ``theta=0``，不使用位置观测。"""
    theta = np.zeros_like(phi_true)
    residual = sim.effective_phase_errors(theta, phi_true, eps)
    power = sim.power_from_theta(theta, phi_true, amp, eps)
    return {
        **add_power_reference_metrics({}, power, p_ideal, p_single_mean, p_single_best),
        "phase_std_deg": float(np.rad2deg(sim.sample_phase_std(residual))),
        "phase_rmse_deg": float(np.rad2deg(sim.phase_rmse(residual))),
        "distance_rmse": np.nan,
        "node_rmse": np.nan,
        "uav_rmse": np.nan,
    }


def evaluate_dfpc(
    p_u_est: np.ndarray,
    node_est_xyz: np.ndarray,
    p_u_true: np.ndarray,
    node_true_xyz: np.ndarray,
    phi_true: np.ndarray,
    amp: np.ndarray,
    eps: np.ndarray,
    k_const: float,
    p_ideal: float,
    p_single_mean: float,
    p_single_best: float,
) -> dict[str, float]:
    """评价 DPC。

    DPC 的核心是用估计距离 d_hat 得到补偿相位 theta。
    如果 d_hat 与真实距离 d_true 有误差，残余相位会变大，功率下降。
    """
    node_est_xyz = np.asarray(node_est_xyz, dtype=np.float64)
    node_true_xyz = np.asarray(node_true_xyz, dtype=np.float64)
    d_hat = np.linalg.norm(p_u_est - node_est_xyz, axis=1)
    d_true = np.linalg.norm(p_u_true[None, :] - node_true_xyz, axis=1)
    phi_hat = k_const * d_hat
    # The physical phase command is k*d_hat.  Reducing it modulo 2*pi is only
    # a hardware representation choice and is not part of the DPC algorithm.
    theta = phi_hat
    residual = sim.effective_phase_errors(theta, phi_true, eps)
    power = sim.power_from_theta(theta, phi_true, amp, eps)
    result = {
        "phase_std_deg": float(np.rad2deg(sim.sample_phase_std(residual))),
        "phase_rmse_deg": float(np.rad2deg(sim.phase_rmse(residual))),
        "distance_rmse": float(np.sqrt(np.mean((d_hat - d_true) ** 2))),
        "node_rmse": float(np.sqrt(np.mean(np.sum((node_est_xyz - node_true_xyz) ** 2, axis=1)))),
        "uav_rmse": float(np.sqrt(np.mean(np.sum((p_u_est - p_u_true[None, :]) ** 2, axis=1)))),
    }
    return add_power_reference_metrics(result, power, p_ideal, p_single_mean, p_single_best)


def mean_stack(arrays: list[np.ndarray]) -> np.ndarray:
    """对 Monte Carlo trial 的数组求平均，并逐元素处理全 NaN 情况。"""
    stack = np.stack(arrays, axis=0)
    valid = ~np.isnan(stack)
    counts = np.sum(valid, axis=0)
    means = np.full(stack.shape[1:], np.nan, dtype=np.float64)
    np.divide(
        np.nansum(stack, axis=0),
        counts,
        out=means,
        where=counts > 0,
    )
    return means


def tail_summary(res: dict[str, Any], method: str, tail_ratio: float = 0.8) -> dict[str, float | str]:
    """取后 20% 迭代步作为稳态区域，生成报告中的摘要指标。"""
    metrics = res["metrics"]
    total_steps = int(res["total_steps"])
    tail = slice(int(tail_ratio * total_steps), None)
    row: dict[str, float | str] = {
        "method": method,
        "tail_mean_power_db": float(10.0 * np.log10(np.maximum(np.mean(metrics[method]["gain_linear"][tail]), 1e-30))),
        "tail_mean_gain_over_single_mean_db": float(
            10.0
            * np.log10(
                np.maximum(
                    np.mean(metrics[method]["power_linear"][tail])
                    / max(np.mean(metrics[method]["single_node_mean_power_linear"][tail]), 1e-30),
                    1e-30,
                )
            )
        ),
        "tail_mean_gain_over_single_best_db": float(
            10.0
            * np.log10(
                np.maximum(
                    np.mean(metrics[method]["power_linear"][tail])
                    / max(np.mean(metrics[method]["single_node_best_power_linear"][tail]), 1e-30),
                    1e-30,
                )
            )
        ),
        "tail_mean_phase_std_deg": float(np.nanmean(metrics[method]["phase_std_deg"][tail])),
        "tail_mean_phase_rmse_deg": float(np.nanmean(metrics[method]["phase_rmse_deg"][tail])),
        "final_power_db": float(metrics[method]["norm_db"][-1]),
    }
    for key, label in [
        ("distance_rmse", "tail_mean_distance_rmse_m"),
        ("node_rmse", "tail_mean_node_rmse_m"),
        ("uav_rmse", "tail_mean_uav_rmse_m"),
        ("uav_line_intercept_rmse", "tail_mean_uav_line_intercept_rmse_m"),
        ("uav_velocity_rmse", "tail_mean_uav_velocity_rmse_mps"),
    ]:
        vals = metrics[method][key][tail]
        row[label] = "" if np.all(np.isnan(vals)) else float(np.nanmean(vals))
    return row
