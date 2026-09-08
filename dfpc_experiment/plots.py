"""绘图、CSV/NPZ 输出和 Markdown 报告。

实验主循环只负责算数据；这个文件负责把数据保存成你看得见的结果。
如果你想改图名、横轴标签、报告表格内容，优先改这里。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .constants import METHOD_DFPC, METHODS, METRIC_KEYS
from .io_utils import write_rows
from .metrics import tail_summary


PLOT_SMOOTHING_WINDOW = 11


def causal_exponential_average(
    values: np.ndarray,
    span: int = PLOT_SMOOTHING_WINDOW,
) -> np.ndarray:
    """Return a NaN-aware causal EMA without modifying the raw series."""
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if series.size == 0:
        return series.copy()
    width = max(int(span), 1)
    if width == 1:
        return series.copy()
    alpha = 2.0 / (width + 1.0)
    smoothed = np.full(series.shape, np.nan, dtype=np.float64)
    previous = np.nan
    for index, value in enumerate(series):
        if np.isfinite(value):
            previous = value if not np.isfinite(previous) else alpha * value + (1.0 - alpha) * previous
        smoothed[index] = previous
    return smoothed


def plot_metric(
    metrics: dict[str, dict[str, np.ndarray]],
    cfg: ExperimentConfig,
    out_dir: Path,
    key: str,
    ylabel: str,
    filename: str,
    *,
    log_y: bool = False,
) -> None:
    """按全局迭代步画曲线。

    横轴是 global iteration = long block 和 iteration 合并后的迭代序号。
    PNG 使用因果指数平滑突出趋势，并以浅色细线保留原始波动。CSV/NPZ
    始终保存未平滑数据。K 只是连续物理时间轴的输出分段，因此不再画
    容易被误解为状态重置的 block 边界线。
    """
    iter_count = cfg.K
    steps = np.arange(cfg.T_long * iter_count)
    plt.figure(figsize=(11, 6))
    power_linear_keys = {
        "norm_db": "gain_linear",
        "gain_over_single_mean_db": "gain_over_single_mean_linear",
        "gain_over_single_best_db": "gain_over_single_best_linear",
    }
    for method_index, method in enumerate(metrics):
        vals = metrics[method][key]
        if np.all(np.isnan(vals)):
            continue
        color = f"C{method_index}"
        plt.plot(steps, vals, color=color, linewidth=0.65, alpha=0.16)
        if key in power_linear_keys:
            linear = causal_exponential_average(metrics[method][power_linear_keys[key]])
            trend = 10.0 * np.log10(np.maximum(linear, 1e-30))
        else:
            trend = causal_exponential_average(vals)
        plt.plot(steps, trend, color=color, linewidth=2.2, label=method)
    if key == "norm_db":
        plt.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    if log_y:
        plt.yscale("log")
    plt.xlabel("Iteration index k")
    plt.ylabel(ylabel)
    plt.title(
        f"{cfg.fc_mhz:.0f} MHz continuous DPC: {ylabel}\n"
        f"causal EMA, span={PLOT_SMOOTHING_WINDOW} (raw trace shown faintly)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()
def save_outputs(res: dict[str, Any], out_dir: Path) -> None:
    """保存一次完整实验的所有输出。

    输出包括：
    - 全局迭代步曲线；
    - curves/block_final/summary CSV；
    - NPZ 原始数组；
    - Markdown 简短报告；
    - debug CSV。
    """
    cfg = ExperimentConfig(**res["args"])
    metrics = res["metrics"]
    methods = res.get("methods") or METHODS
    out_dir.mkdir(parents=True, exist_ok=True)
    freq_tag = f"{res['frequency_MHz']:.0f}MHz"

    plot_metric(metrics, cfg, out_dir, "norm_db", r"$10\log_{10}(P_J/P_{ideal})$ (dB)", "基准实验_归一化相干功率_迭代曲线.png")
    plot_metric(metrics, cfg, out_dir, "gain_over_single_mean_db", r"$10\log_{10}(P_J/P_{single,mean})$ (dB)", "基准实验_相对平均单节点增益_迭代曲线.png")
    plot_metric(metrics, cfg, out_dir, "gain_over_single_best_db", r"$10\log_{10}(P_J/P_{single,best})$ (dB)", "基准实验_相对最佳单节点增益_迭代曲线.png")
    plot_metric(metrics, cfg, out_dir, "phase_std_deg", "residual phase std (deg)", "基准实验_残余相位标准差_迭代曲线.png")
    plot_metric(metrics, cfg, out_dir, "phase_rmse_deg", "residual phase RMSE (deg)", "基准实验_残余相位RMSE_迭代曲线.png")
    plot_metric(metrics, cfg, out_dir, "distance_rmse", "distance RMSE (m)", "基准实验_距离RMSE_迭代曲线.png")
    plot_metric(metrics, cfg, out_dir, "node_rmse", "node position RMSE (m)", "基准实验_浮标位置RMSE_迭代曲线.png")
    plot_metric(metrics, cfg, out_dir, "uav_rmse", "UAV position RMSE (m)", "基准实验_UAV位置RMSE_迭代曲线.png")
    # The line-intercept RMSE remains in CSV/NPZ for diagnosis, but it is not a
    # primary performance curve: b = p - t*v amplifies tiny velocity errors as
    # physical time grows and can look like current-position divergence.
    plot_metric(
        metrics,
        cfg,
        out_dir,
        "uav_velocity_rmse",
        "UAV velocity RMSE (m/s)",
        "基准实验_UAV速度RMSE_迭代曲线.png",
        log_y=True,
    )

    iter_count = cfg.K
    rows = []
    # 每个 global iteration 的完整曲线数据，适合调试异常点。
    for sidx in range(res["total_steps"]):
        long_block = sidx // iter_count
        row: dict[str, Any] = {
            "global_step": sidx,
            "long_block": long_block,
            "physical_time_s": (long_block * cfg.K + sidx % iter_count) * float(cfg.Ts),
            "iteration_step": sidx % iter_count,
        }
        for method in methods:
            prefix = method.lower().replace(" ", "_").replace("-", "_")
            for key in METRIC_KEYS:
                val = metrics[method][key][sidx]
                row[f"{prefix}_{key}"] = "" if np.isnan(val) else float(val)
        rows.append(row)
    write_rows(out_dir / f"基准实验_{freq_tag}_逐迭代曲线.csv", rows)
    write_rows(out_dir / f"基准实验_{freq_tag}_长窗口末值.csv", res["block_final_rows"])
    write_rows(out_dir / f"基准实验_{freq_tag}_MC逐次汇总.csv", res["trial_summary_rows"])
    write_rows(out_dir / f"基准实验_{freq_tag}_MC分窗口数据.csv", res["trial_block_rows"])

    summary_rows = []
    # 后 20% 迭代步的稳态摘要。
    dfpc_reference = tail_summary(res, METHOD_DFPC)
    for method in methods:
        row = {
            "frequency_MHz": res["frequency_MHz"],
            "wavelength_m": res["wavelength_m"],
            "T_long": cfg.T_long,
            "K": cfg.K,
            "TL_s": cfg.K * cfg.Ts,
            "Ts_s": cfg.Ts,
            "physical_duration_s": res["physical_duration_s"],
            "buoy_wave_speed_mps": cfg.buoy_wave_speed,
            "buoy_wave_heading_deg": cfg.buoy_wave_heading_deg,
            "buoy_random_displacement_std_m": cfg.buoy_random_displacement_std,
            "buoy_center_accumulation_ratio": cfg.buoy_center_accumulation_ratio,
            "mc_trials": res["mc_trials"],
            "compute_device": res["compute_device"],
        }
        method_summary = tail_summary(res, method)
        row.update(method_summary)
        # Positive power change is better; negative phase/RMSE change is better.
        row["power_change_vs_dfpc_db"] = float(
            method_summary["tail_mean_power_db"] - dfpc_reference["tail_mean_power_db"]
        )
        row["phase_std_change_vs_dfpc_deg"] = float(
            method_summary["tail_mean_phase_std_deg"]
            - dfpc_reference["tail_mean_phase_std_deg"]
        )
        row["phase_rmse_change_vs_dfpc_deg"] = float(
            method_summary["tail_mean_phase_rmse_deg"]
            - dfpc_reference["tail_mean_phase_rmse_deg"]
        )
        distance = method_summary["tail_mean_distance_rmse_m"]
        reference_distance = dfpc_reference["tail_mean_distance_rmse_m"]
        row["distance_rmse_change_vs_dfpc_m"] = (
            "" if distance == "" or reference_distance == "" else float(distance - reference_distance)
        )
        summary_rows.append(row)
    write_rows(out_dir / f"基准实验_{freq_tag}_方法汇总.csv", summary_rows)

    block_times = (
        np.repeat(np.arange(cfg.T_long, dtype=np.float64) * cfg.K, iter_count)
        + np.tile(np.arange(iter_count, dtype=np.float64), cfg.T_long)
    ) * float(cfg.Ts)
    npz_data: dict[str, Any] = {
        "steps": np.arange(res["total_steps"]),
        "physical_time_s": block_times,
        "methods": np.array(methods, dtype=object),
    }
    # NPZ 保存原始数组，方便后续不用重跑实验就能重新画图。
    for method in methods:
        prefix = method.lower().replace(" ", "_").replace("-", "_")
        for key, arr in metrics[method].items():
            npz_data[f"{prefix}_{key}"] = arr
    np.savez(out_dir / f"基准实验_{freq_tag}_原始数组.npz", **npz_data)

    table = []
    for row in summary_rows:
        table.append(
            f"| {row['method']} | {row['tail_mean_power_db']:.2f} dB | "
            f"{row['power_change_vs_dfpc_db']:+.2f} dB | "
            f"{row['tail_mean_gain_over_single_mean_db']:.2f} dB | "
            f"{row['tail_mean_phase_std_deg']:.2f} deg | "
            f"{row['tail_mean_phase_rmse_deg']:.2f} deg | "
            f"{row['tail_mean_node_rmse_m']} | {row['tail_mean_distance_rmse_m']} | {row['tail_mean_uav_rmse_m']} | "
            f"{row['tail_mean_uav_velocity_rmse_mps']} |"
        )
    report = f"""# {res['frequency_MHz']:.0f} MHz Modular Dual-Timescale DPC Experiment

This is the modular DPC experiment.

Monte Carlo trials: {res['mc_trials']}. Global consensus device: `{res['compute_device']}`.
Physical duration: {res['physical_duration_s']:.3f} s; block duration K*Ts={cfg.K * cfg.Ts:g} s; Ts={cfg.Ts:g} s.
Buoy centers move at {cfg.buoy_wave_speed:g} m/s in the {cfg.buoy_wave_heading_deg:g} deg direction. At every iteration, separated by Ts={cfg.Ts:g} s, a random displacement with std={cfg.buoy_random_displacement_std:g} m per horizontal axis is sampled; {cfg.buoy_center_accumulation_ratio:g} of it accumulates into the center and the remainder is the instantaneous offset.
Each buoy independently estimates the UAV trajectory. DPC reaches consensus on
`[bx,vx,by,vy,bz,vz]` and unit flight direction `[dx,dy,dz]`.
UAV+Node-KF DPC filters both UAV and buoy states as `[x,y,z,vx,vy,vz]` and uses 3-D ranges.
The only filtering branch retained is the linear 3-D constant-velocity Kalman filter.
    Power is averaged as linear gain before conversion to dB. Positive power change versus DPC is better.
    PNG curves use a causal EMA with span {PLOT_SMOOTHING_WINDOW} and show the raw trace faintly; CSV and NPZ remain unsmoothed.
    K is only an output grouping on one continuous physical timeline, so no modulo-K "iteration-axis" figures are generated.
Every method uses an ideal 0/pi polarity choice. Effective residual phases are folded modulo pi into [-90, 90) deg; they are not clipped.
`No Algorithm` is the pure-random reference: it transmits with theta=0 and uses no observations, consensus, or filtering.

| method | tail normalized power | power change vs DPC | gain over mean single node | tail phase std | tail phase RMSE | tail node RMSE | tail distance RMSE | tail UAV RMSE | tail UAV velocity RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}
"""
    (out_dir / f"基准实验_{freq_tag}_结果报告.md").write_text(report, encoding="utf-8")

    debug_recorder = res.get("debug_recorder")
    if debug_recorder is not None:
        # 如果没有打开 debug，DebugRecorder.write 会直接返回。
        debug_recorder.write(out_dir)
