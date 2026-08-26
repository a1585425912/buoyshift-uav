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


def plot_metric(metrics: dict[str, dict[str, np.ndarray]], cfg: ExperimentConfig, out_dir: Path, key: str, ylabel: str, filename: str) -> None:
    """按全局时间步画曲线。

    横轴是 global step = long block 和 iteration 混合后的时间序号。
    图中的细竖线表示长时间尺度 block 的边界。
    """
    iter_count = cfg.K
    steps = np.arange(cfg.T_long * iter_count)
    plt.figure(figsize=(11, 6))
    for method in metrics:
        vals = metrics[method][key]
        if np.all(np.isnan(vals)):
            continue
        plt.plot(steps, vals, linewidth=1.8, label=method)
    if key == "norm_db":
        plt.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    for block in range(1, cfg.T_long):
        plt.axvline(block * cfg.K, color="black", linewidth=0.4, alpha=0.12)
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    plt.title(f"{cfg.fc_mhz:.0f} MHz step-0 recursive-KF DPC: {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()


def iteration_profiles(metrics: dict[str, dict[str, np.ndarray]], cfg: ExperimentConfig) -> dict[str, dict[str, np.ndarray]]:
    """把所有长 block 按 iteration 对齐后求平均。

    这个函数生成“横轴为迭代步数”的图，用来观察算法是否逐步收敛。
    """
    iter_count = cfg.K
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for method in metrics:
        profiles[method] = {}
        for key in METRIC_KEYS:
            vals = metrics[method][key].reshape(cfg.T_long, iter_count)
            profiles[method][key] = np.full(iter_count, np.nan, dtype=np.float64) if np.all(np.isnan(vals)) else np.nanmean(vals, axis=0)
        profiles[method]["norm_db"] = 10.0 * np.log10(np.maximum(profiles[method]["gain_linear"], 1e-30))
    return profiles


def plot_iteration_profile(
    profiles: dict[str, dict[str, np.ndarray]],
    cfg: ExperimentConfig,
    out_dir: Path,
    key: str,
    ylabel: str,
    filename: str,
) -> None:
    """画横轴为 iteration step 的平均收敛曲线。"""
    iters = np.arange(cfg.K)
    plt.figure(figsize=(9.5, 5.6))
    for method in profiles:
        vals = profiles[method][key]
        if np.all(np.isnan(vals)):
            continue
        plt.plot(iters, vals, marker="o", markersize=3.2, linewidth=1.9, label=method)
    if key == "norm_db":
        plt.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.xlabel("Short-time step")
    plt.ylabel(ylabel)
    plt.title(f"{cfg.fc_mhz:.0f} MHz DPC by iteration step: {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()


def save_outputs(res: dict[str, Any], out_dir: Path) -> None:
    """保存一次完整实验的所有输出。

    输出包括：
    - 全局时间步曲线；
    - iteration 轴曲线；
    - curves/block_final/summary CSV；
    - NPZ 原始数组；
    - Markdown 简短报告；
    - debug CSV。
    """
    cfg = ExperimentConfig(**res["args"])
    metrics = res["metrics"]
    methods = res.get("methods") or METHODS
    out_dir.mkdir(parents=True, exist_ok=True)
    freq_tag = f"{res['frequency_MHz']:.0f}mhz"

    plot_metric(metrics, cfg, out_dir, "norm_db", r"$10\log_{10}(P_J/P_{ideal})$ (dB)", "dpc_power_db.png")
    plot_metric(metrics, cfg, out_dir, "gain_over_single_mean_db", r"$10\log_{10}(P_J/P_{single,mean})$ (dB)", "dpc_gain_over_single_mean_db.png")
    plot_metric(metrics, cfg, out_dir, "gain_over_single_best_db", r"$10\log_{10}(P_J/P_{single,best})$ (dB)", "dpc_gain_over_single_best_db.png")
    plot_metric(metrics, cfg, out_dir, "phase_std_deg", "residual phase std (deg)", "dpc_phase_std_deg.png")
    plot_metric(metrics, cfg, out_dir, "phase_rmse_deg", "residual phase RMSE (deg)", "dpc_phase_rmse_deg.png")
    plot_metric(metrics, cfg, out_dir, "distance_rmse", "distance RMSE (m)", "dpc_distance_rmse.png")
    plot_metric(metrics, cfg, out_dir, "node_rmse", "node position RMSE (m)", "dpc_node_rmse.png")
    plot_metric(metrics, cfg, out_dir, "uav_rmse", "UAV position RMSE (m)", "dpc_uav_rmse.png")
    plot_metric(metrics, cfg, out_dir, "uav_line_intercept_rmse", "UAV line intercept RMSE (m)", "dpc_uav_line_intercept_rmse.png")
    plot_metric(metrics, cfg, out_dir, "uav_velocity_rmse", "UAV velocity RMSE (m/s)", "dpc_uav_velocity_rmse.png")

    profiles = iteration_profiles(metrics, cfg)
    plot_iteration_profile(profiles, cfg, out_dir, "norm_db", r"$10\log_{10}(P_J/P_{ideal})$ (dB)", "dpc_iteration_axis_power_db.png")
    plot_iteration_profile(profiles, cfg, out_dir, "gain_over_single_mean_db", r"$10\log_{10}(P_J/P_{single,mean})$ (dB)", "dpc_iteration_axis_gain_over_single_mean_db.png")
    plot_iteration_profile(profiles, cfg, out_dir, "phase_std_deg", "residual phase std (deg)", "dpc_iteration_axis_phase_std_deg.png")
    plot_iteration_profile(profiles, cfg, out_dir, "phase_rmse_deg", "residual phase RMSE (deg)", "dpc_iteration_axis_phase_rmse_deg.png")
    plot_iteration_profile(profiles, cfg, out_dir, "distance_rmse", "distance RMSE (m)", "dpc_iteration_axis_distance_rmse.png")
    plot_iteration_profile(profiles, cfg, out_dir, "node_rmse", "node position RMSE (m)", "dpc_iteration_axis_node_rmse.png")
    plot_iteration_profile(profiles, cfg, out_dir, "uav_rmse", "UAV position RMSE (m)", "dpc_iteration_axis_uav_rmse.png")
    plot_iteration_profile(profiles, cfg, out_dir, "uav_line_intercept_rmse", "UAV line intercept RMSE (m)", "dpc_iteration_axis_uav_line_intercept_rmse.png")
    plot_iteration_profile(profiles, cfg, out_dir, "uav_velocity_rmse", "UAV velocity RMSE (m/s)", "dpc_iteration_axis_uav_velocity_rmse.png")

    iter_count = cfg.K
    rows = []
    # 每个 global step 的完整曲线数据，适合调试异常点。
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
    write_rows(out_dir / f"dual_timescale_dpc_{freq_tag}_curves.csv", rows)
    write_rows(out_dir / f"dual_timescale_dpc_{freq_tag}_block_final.csv", res["block_final_rows"])
    write_rows(out_dir / f"dual_timescale_dpc_{freq_tag}_trial_summary.csv", res["trial_summary_rows"])
    write_rows(out_dir / f"dual_timescale_dpc_{freq_tag}_trial_block.csv", res["trial_block_rows"])

    iter_rows = []
    # 按 iteration 对齐后的平均曲线数据，适合论文图表。
    for it in range(iter_count):
        row = {"iteration_step": it}
        for method in methods:
            prefix = method.lower().replace(" ", "_").replace("-", "_")
            for key in METRIC_KEYS:
                val = profiles[method][key][it]
                row[f"{prefix}_{key}"] = "" if np.isnan(val) else float(val)
        iter_rows.append(row)
    write_rows(out_dir / f"dual_timescale_dpc_{freq_tag}_iteration_axis.csv", iter_rows)

    summary_rows = []
    # 后 20% 时间步的稳态摘要。
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
    write_rows(out_dir / f"dual_timescale_dpc_{freq_tag}_summary.csv", summary_rows)

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
    np.savez(out_dir / f"dual_timescale_dpc_{freq_tag}_data.npz", **npz_data)

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
Buoy centers move at {cfg.buoy_wave_speed:g} m/s in the {cfg.buoy_wave_heading_deg:g} deg direction. At every time step, a random displacement with std={cfg.buoy_random_displacement_std:g} m per horizontal axis is sampled; {cfg.buoy_center_accumulation_ratio:g} of it accumulates into the center and the remainder is the instantaneous offset.
Each buoy independently estimates the UAV trajectory. DPC reaches consensus on
`[bx,vx,by,vy,bz,vz]` and unit flight direction `[dx,dy,dz]`.
UAV+Node-KF DPC filters both UAV and buoy states as `[x,y,z,vx,vy,vz]` and uses 3-D ranges.
The only filtering branch retained is the linear 3-D constant-velocity Kalman filter.
Power is averaged as linear gain before conversion to dB. Positive power change versus DPC is better.
Every method uses an ideal 0/pi polarity choice. Effective residual phases are folded modulo pi into [-90, 90) deg; they are not clipped.
`Random Phase Reference` transmits with theta=0 and uses no observations. `No Algorithm` uses each node's instantaneous noisy UAV/node positions without consensus or filtering.

| method | tail normalized power | power change vs DPC | gain over mean single node | tail phase std | tail phase RMSE | tail node RMSE | tail distance RMSE | tail UAV RMSE | tail UAV velocity RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}
"""
    (out_dir / f"dual_timescale_dpc_{freq_tag}_report.md").write_text(report, encoding="utf-8")

    debug_recorder = res.get("debug_recorder")
    if debug_recorder is not None:
        # 如果没有打开 debug，DebugRecorder.write 会直接返回。
        debug_recorder.write(out_dir)
