"""敏感性分析工具。

这个文件负责“改变一个因素并重复跑实验”，例如：
节点数、连通性、频率、位置误差、海浪均值速度。

单次实验逻辑仍然在 `experiment.py`，这里不重复实现物理模型。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .constants import (
    METHOD_DFPC,
    METHOD_KF_DFPC,
    METHOD_NODE_KF_DPC,
    METHOD_NO_ALG,
)
from .experiment import run_experiment
from .io_utils import write_rows


PERFORMANCE_METHODS = [
    (METHOD_DFPC, "dfpc"),
    (METHOD_NODE_KF_DPC, "node_kf_dpc"),
    (METHOD_KF_DFPC, "kf_dfpc"),
]

FACTOR_FILE_LABELS = {
    "node_count": "节点数扫描",
    "frequency": "载波频率扫描",
    "connectivity": "连通度扫描",
    "node_position_noise": "浮标位置误差扫描",
    "uav_position_error": "UAV位置误差扫描",
    "wave_speed": "海流速度扫描",
}


def factor_file_label(factor: str) -> str:
    """Return a readable Chinese filename prefix for a sweep factor."""
    return FACTOR_FILE_LABELS.get(factor, factor.replace("_", "-"))


def parse_float_list(text: str) -> list[float]:
    """把命令行里的逗号分隔浮点数解析成列表。"""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    """把命令行里的逗号分隔整数解析成列表。"""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def controlled_node_settings(
    node_count: int,
    target_mean_degree: float,
) -> dict[str, int | float]:
    """Keep the expected communication-graph degree stable while N changes."""
    n = int(node_count)
    if n < 4:
        raise ValueError("controlled node-count sweeps require at least 4 nodes")
    random_edge_probability = float(
        np.clip((float(target_mean_degree) - 2.0) / (n - 3.0), 0.0, 1.0)
    )
    return {
        "global_connectivity": random_edge_probability,
    }


def summarize_result(res: dict[str, Any], factor: str, value: float, label: str) -> dict[str, Any]:
    """把一次 sweep 点的完整结果压缩成一行 summary。

    这里同样遵守一个原则：功率先在线性域取平均，再转 dB。
    """
    metrics = res["metrics"]
    steps = res["total_steps"]
    tail = slice(int(0.8 * steps), None)
    dfpc_tail_gain = float(np.mean(metrics[METHOD_DFPC]["gain_linear"][tail]))
    no_alg_tail_gain = float(np.mean(metrics[METHOD_NO_ALG]["gain_linear"][tail]))
    dfpc_tail_power = float(np.mean(metrics[METHOD_DFPC]["power_linear"][tail]))
    kf_tail_gain = float(np.mean(metrics[METHOD_KF_DFPC]["gain_linear"][tail]))
    kf_tail_power = float(np.mean(metrics[METHOD_KF_DFPC]["power_linear"][tail]))
    node_kf_tail_gain = float(np.mean(metrics[METHOD_NODE_KF_DPC]["gain_linear"][tail]))
    node_kf_tail_power = float(np.mean(metrics[METHOD_NODE_KF_DPC]["power_linear"][tail]))
    no_alg_tail_power = float(np.mean(metrics[METHOD_NO_ALG]["power_linear"][tail]))
    single_mean_tail_power = float(np.mean(metrics[METHOD_DFPC]["single_node_mean_power_linear"][tail]))
    single_best_tail_power = float(np.mean(metrics[METHOD_DFPC]["single_node_best_power_linear"][tail]))
    row = {
        "factor": factor,
        "factor_value": value,
        "factor_label": label,
        "frequency_MHz": float(res["frequency_MHz"]),
        "wavelength_m": float(res["wavelength_m"]),
        "N": int(res["args"]["N"]),
        "connectivity": float(res["args"]["global_connectivity"]),
        "actual_mean_degree": float(res["graph_diagnostics"]["mean_degree"]),
        "uav_position_noise_m": float(res["args"]["uav_obs_noise"]),
        "node_position_noise_m": float(res["args"]["buoy_center_obs_noise"]),
        "mc_trials": int(res["mc_trials"]),
        "T_long": int(res["args"]["T_long"]),
        "K": int(res["args"]["K"]),
        "TL_s": float(res["args"]["K"]) * float(res["args"]["Ts"]),
        "Ts_s": float(res["args"]["Ts"]),
        "buoy_wave_speed_mps": float(res["args"]["buoy_wave_speed"]),
        "buoy_random_displacement_std_m": float(res["args"]["buoy_random_displacement_std"]),
        "buoy_center_accumulation_ratio": float(res["args"]["buoy_center_accumulation_ratio"]),
        "no_algorithm_tail_power_db": float(10.0 * np.log10(max(no_alg_tail_gain, 1e-30))),
        "dfpc_tail_power_db": float(10.0 * np.log10(max(dfpc_tail_gain, 1e-30))),
        "dfpc_gain_over_no_algorithm_db": float(10.0 * np.log10(max(dfpc_tail_gain / max(no_alg_tail_gain, 1e-30), 1e-30))),
        "no_algorithm_tail_gain_over_single_mean_db": float(10.0 * np.log10(max(no_alg_tail_power / max(single_mean_tail_power, 1e-30), 1e-30))),
        "dfpc_tail_gain_over_single_mean_db": float(10.0 * np.log10(max(dfpc_tail_power / max(single_mean_tail_power, 1e-30), 1e-30))),
        "dfpc_tail_gain_over_single_best_db": float(10.0 * np.log10(max(dfpc_tail_power / max(single_best_tail_power, 1e-30), 1e-30))),
        "no_algorithm_tail_phase_std_deg": float(np.nanmean(metrics[METHOD_NO_ALG]["phase_std_deg"][tail])),
        "no_algorithm_tail_phase_rmse_deg": float(np.nanmean(metrics[METHOD_NO_ALG]["phase_rmse_deg"][tail])),
        "dfpc_tail_phase_std_deg": float(np.nanmean(metrics[METHOD_DFPC]["phase_std_deg"][tail])),
        "dfpc_tail_phase_rmse_deg": float(np.nanmean(metrics[METHOD_DFPC]["phase_rmse_deg"][tail])),
        "dfpc_tail_distance_rmse_m": float(np.nanmean(metrics[METHOD_DFPC]["distance_rmse"][tail])),
        "dfpc_tail_node_rmse_m": float(np.nanmean(metrics[METHOD_DFPC]["node_rmse"][tail])),
        "dfpc_tail_uav_rmse_m": float(np.nanmean(metrics[METHOD_DFPC]["uav_rmse"][tail])),
        "dfpc_tail_uav_line_intercept_rmse_m": float(np.nanmean(metrics[METHOD_DFPC]["uav_line_intercept_rmse"][tail])),
        "dfpc_tail_uav_velocity_rmse_mps": float(np.nanmean(metrics[METHOD_DFPC]["uav_velocity_rmse"][tail])),
        "dfpc_final_power_db": float(metrics[METHOD_DFPC]["norm_db"][-1]),
        "dfpc_final_phase_std_deg": float(metrics[METHOD_DFPC]["phase_std_deg"][-1]),
        "dfpc_final_phase_rmse_deg": float(metrics[METHOD_DFPC]["phase_rmse_deg"][-1]),
        "dfpc_final_distance_rmse_m": float(metrics[METHOD_DFPC]["distance_rmse"][-1]),
        "node_kf_dpc_tail_power_db": float(10.0 * np.log10(max(node_kf_tail_gain, 1e-30))),
        "node_kf_improvement_over_dfpc_db": float(10.0 * np.log10(max(node_kf_tail_gain / max(dfpc_tail_gain, 1e-30), 1e-30))),
        "node_kf_dpc_gain_over_single_mean_db": float(10.0 * np.log10(max(node_kf_tail_power / max(single_mean_tail_power, 1e-30), 1e-30))),
        "node_kf_dpc_phase_std_deg": float(np.nanmean(metrics[METHOD_NODE_KF_DPC]["phase_std_deg"][tail])),
        "node_kf_dpc_phase_rmse_deg": float(np.nanmean(metrics[METHOD_NODE_KF_DPC]["phase_rmse_deg"][tail])),
        "node_kf_dpc_distance_rmse_m": float(np.nanmean(metrics[METHOD_NODE_KF_DPC]["distance_rmse"][tail])),
        "node_kf_dpc_node_rmse_m": float(np.nanmean(metrics[METHOD_NODE_KF_DPC]["node_rmse"][tail])),
        "node_kf_dpc_uav_rmse_m": float(np.nanmean(metrics[METHOD_NODE_KF_DPC]["uav_rmse"][tail])),
        "node_kf_dpc_final_power_db": float(metrics[METHOD_NODE_KF_DPC]["norm_db"][-1]),
        "node_kf_dpc_final_phase_std_deg": float(metrics[METHOD_NODE_KF_DPC]["phase_std_deg"][-1]),
        "node_kf_dpc_final_phase_rmse_deg": float(metrics[METHOD_NODE_KF_DPC]["phase_rmse_deg"][-1]),
        "node_kf_dpc_final_distance_rmse_m": float(metrics[METHOD_NODE_KF_DPC]["distance_rmse"][-1]),
        "kf_dfpc_tail_power_db": float(10.0 * np.log10(max(kf_tail_gain, 1e-30))),
        "kf_improvement_over_dfpc_db": float(10.0 * np.log10(max(kf_tail_gain / max(dfpc_tail_gain, 1e-30), 1e-30))),
        "kf_dfpc_gain_over_single_mean_db": float(10.0 * np.log10(max(kf_tail_power / max(single_mean_tail_power, 1e-30), 1e-30))),
        "kf_dfpc_phase_std_deg": float(np.nanmean(metrics[METHOD_KF_DFPC]["phase_std_deg"][tail])),
        "kf_dfpc_phase_rmse_deg": float(np.nanmean(metrics[METHOD_KF_DFPC]["phase_rmse_deg"][tail])),
        "kf_dfpc_distance_rmse_m": float(np.nanmean(metrics[METHOD_KF_DFPC]["distance_rmse"][tail])),
        "kf_dfpc_node_rmse_m": float(np.nanmean(metrics[METHOD_KF_DFPC]["node_rmse"][tail])),
        "kf_dfpc_uav_rmse_m": float(np.nanmean(metrics[METHOD_KF_DFPC]["uav_rmse"][tail])),
        "kf_dfpc_final_power_db": float(metrics[METHOD_KF_DFPC]["norm_db"][-1]),
        "kf_dfpc_final_phase_std_deg": float(metrics[METHOD_KF_DFPC]["phase_std_deg"][-1]),
        "kf_dfpc_final_phase_rmse_deg": float(metrics[METHOD_KF_DFPC]["phase_rmse_deg"][-1]),
        "kf_dfpc_final_distance_rmse_m": float(metrics[METHOD_KF_DFPC]["distance_rmse"][-1]),
    }
    return row


def convergence_summary(
    res: dict[str, Any],
    method: str,
    tolerance_db: float = 0.5,
) -> dict[str, float | int]:
    """Measure sustained convergence to a method's own steady-state band."""
    gain = np.asarray(res["metrics"][method]["gain_linear"], dtype=np.float64)
    total = gain.size
    block_window = max(1, min(int(res["args"]["K"]), total))
    settling_window = max(1, min(5, total))
    tail = slice(int(0.8 * total), None)
    tail_db = float(10.0 * np.log10(max(float(np.mean(gain[tail])), 1e-30)))
    rolling = np.convolve(gain, np.ones(settling_window) / settling_window, mode="valid")
    rolling_db = 10.0 * np.log10(np.maximum(rolling, 1e-30))
    hold = max(1, min(5, rolling_db.size))

    def sustained_step(tolerance: float) -> int:
        threshold = tail_db - tolerance
        for index in range(rolling_db.size - hold + 1):
            if np.all(rolling_db[index : index + hold] >= threshold):
                return index + settling_window - 1
        return total

    convergence_step = sustained_step(float(tolerance_db))
    strict_step = sustained_step(0.1)
    norm_db = 10.0 * np.log10(np.maximum(gain, 1e-30))
    transient_end = max(1, total // 2)
    transient_deficit = np.maximum(tail_db - norm_db[:transient_end], 0.0)
    return {
        "tail_power_db": tail_db,
        "convergence_step_05db": convergence_step,
        "convergence_time_s_05db": convergence_step * float(res["args"]["Ts"]),
        "settling_step_01db_w5": strict_step,
        "settling_time_s_01db_w5": strict_step * float(res["args"]["Ts"]),
        "transient_deficit_mean_db": float(np.mean(transient_deficit)),
        "first_block_power_db": float(
            10.0 * np.log10(max(float(np.mean(gain[:block_window])), 1e-30))
        ),
    }


def distance_rmse_convergence_summary(
    res: dict[str, Any],
    method: str,
) -> dict[str, float | int]:
    """Return when the configured distance-RMSE convergence rule first holds."""
    key = {
        METHOD_DFPC: "dpc_phase_ready",
        METHOD_NODE_KF_DPC: "node_kf_dpc_phase_ready",
        METHOD_KF_DFPC: "kf_dpc_phase_ready",
    }[method]
    ready = np.asarray(res["consensus_diagnostics"][key], dtype=bool)
    indices = np.flatnonzero(ready)
    step = int(indices[0]) if indices.size else int(ready.size)
    return {
        "distance_rmse_convergence_step": step,
        "distance_rmse_convergence_time_s": step * float(res["args"]["Ts"]),
    }


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if total <= 1e-15 else 1.0 - residual / total
    return float(slope), float(r_squared)


def node_count_scaling_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinguish suspicious linear-in-N growth from expected log scaling."""
    group = sorted(rows, key=lambda row: float(row["factor_value"]))
    nodes = np.array([float(row["factor_value"]) for row in group], dtype=np.float64)
    log_nodes = np.log10(nodes)
    diagnostics = []
    for method, prefix in PERFORMANCE_METHODS:
        power_key = f"{prefix}_tail_power_db"
        gain_key = (
            "dfpc_tail_gain_over_single_mean_db"
            if prefix == "dfpc"
            else f"{prefix}_gain_over_single_mean_db"
        )
        power = np.array([float(row[power_key]) for row in group], dtype=np.float64)
        gain = np.array([float(row[gain_key]) for row in group], dtype=np.float64)
        linear_slope, linear_r2 = _fit_line(nodes, power)
        log_slope, log_r2 = _fit_line(log_nodes, power)
        gain_slope, gain_r2 = _fit_line(log_nodes, gain)
        span = float(np.max(power) - np.min(power))
        diagnostics.append(
            {
                "method": method,
                "node_count_min": int(nodes[0]),
                "node_count_max": int(nodes[-1]),
                "normalized_power_span_db": span,
                "linear_slope_db_per_1000_nodes": 1000.0 * linear_slope,
                "linear_fit_r2": linear_r2,
                "log_fit_slope_db_per_decade": log_slope,
                "log_fit_r2": log_r2,
                "gain_slope_db_per_decade": gain_slope,
                "gain_log_fit_r2": gain_r2,
                "gain_slope_error_from_20_db": gain_slope - 20.0,
                "suspicious_linear_growth": bool(
                    span > 1.0 and linear_r2 >= 0.95 and linear_r2 > log_r2 + 0.02
                ),
            }
        )
    return diagnostics


def sweep_points(factor: str, values: str) -> list[tuple[float, str, dict[str, Any]]]:
    """把某个因素的取值转换成配置覆盖项。

    返回的 overrides 会覆盖 ExperimentConfig 中的对应字段。
    """
    if factor == "node_count":
        return [(float(n), f"N={n}", {"N": n}) for n in parse_int_list(values)]
    if factor == "connectivity":
        return [(p, f"p={p:g}", {"global_connectivity": p}) for p in parse_float_list(values)]
    if factor == "frequency":
        return [(f, f"{f:g} MHz", {"fc_mhz": f}) for f in parse_float_list(values)]
    if factor == "node_position_noise":
        return [(x, f"{x:g} m", {"buoy_center_obs_noise": x}) for x in parse_float_list(values)]
    if factor == "uav_position_error":
        return [(x, f"{x:g} m", {"uav_obs_noise": x}) for x in parse_float_list(values)]
    if factor == "wave_speed":
        return [(x, f"{x:g} m/s", {"buoy_wave_speed": x}) for x in parse_float_list(values)]
    raise ValueError(f"unknown factor: {factor}")


def run_factor(
    cfg: ExperimentConfig,
    factor: str,
    values: str,
    out_dir: Path,
    rebound_threshold_db: float = 0.5,
    controlled_design: bool = False,
    target_mean_degree: float = 8.0,
) -> list[dict[str, Any]]:
    """运行一个因素的完整 sweep。"""
    rows = []
    trial_diagnostics: list[dict[str, Any]] = []
    block_diagnostics: list[dict[str, Any]] = []
    connectivity_diagnostics: list[dict[str, Any]] = []
    connectivity_time_rows: list[dict[str, Any]] = []
    for idx, (value, label, overrides) in enumerate(sweep_points(factor, values)):
        point_dir = out_dir / factor / f"point_{idx:02d}"
        point_cfg = replace(cfg, out_dir=str(point_dir))
        if controlled_design and factor == "node_count":
            overrides.update(
                controlled_node_settings(
                    int(value),
                    target_mean_degree,
                )
            )
        for key, val in overrides.items():
            setattr(point_cfg, key, val)
        print(f"[{factor}] {idx + 1} {label}", flush=True)
        res = run_experiment(point_cfg)
        summary = summarize_result(res, factor, value, label)
        summary.update(
            {
                "controlled_design": bool(controlled_design),
                "target_mean_degree": target_mean_degree if factor == "node_count" else "",
                "point_index": idx,
                "point_out_dir": str(point_dir),
            }
        )
        rows.append(summary)
        common = {
            "factor": factor,
            "factor_value": value,
            "factor_label": label,
            "point_index": idx,
            "point_out_dir": str(point_dir),
        }
        trial_diagnostics.extend([{**common, **row} for row in res["trial_summary_rows"]])
        block_diagnostics.extend([{**common, **row} for row in res["trial_block_rows"]])
        if factor == "connectivity":
            for method, _ in PERFORMANCE_METHODS:
                connectivity_diagnostics.append(
                    {
                        **common,
                        "method": method,
                        **convergence_summary(res, method),
                        **distance_rmse_convergence_summary(res, method),
                    }
                )
                for step, power_db in enumerate(res["metrics"][method]["norm_db"]):
                    connectivity_time_rows.append(
                        {
                            **common,
                            "method": method,
                            "global_step": step,
                            "physical_time_s": step * float(res["args"]["Ts"]),
                            "normalized_power_db": float(power_db),
                            "distance_rmse_m": float(
                                res["metrics"][method]["distance_rmse"][step]
                            ),
                        }
                    )
        if point_cfg.debug:
            res["debug_recorder"].write(point_dir)
    file_prefix = factor_file_label(factor)
    write_rows(out_dir / f"{file_prefix}_实验汇总.csv", rows)
    write_rows(out_dir / f"{file_prefix}_MC逐次诊断.csv", trial_diagnostics)
    write_rows(out_dir / f"{file_prefix}_MC分窗口诊断.csv", block_diagnostics)
    if factor == "node_count":
        write_rows(
            out_dir / "节点数扫描_规模变化诊断.csv",
            node_count_scaling_diagnostics(rows),
        )
    if factor == "connectivity":
        write_rows(out_dir / "连通度扫描_稳态迭代步汇总.csv", connectivity_diagnostics)
        write_rows(out_dir / "连通度扫描_逐迭代曲线数据.csv", connectivity_time_rows)
        plot_connectivity_convergence(connectivity_diagnostics, out_dir)
        plot_connectivity_time_curves(connectivity_time_rows, out_dir)
    if factor == "frequency":
        events = detect_frequency_rebounds(rows, trial_diagnostics, rebound_threshold_db)
        write_rows(out_dir / "载波频率扫描_异常回升事件.csv", events)
        write_frequency_rebound_report(events, rebound_threshold_db, out_dir)
    plot_factor(rows, factor, out_dir)
    return rows


def detect_frequency_rebounds(
    rows: list[dict[str, Any]],
    trial_diagnostics: list[dict[str, Any]],
    threshold_db: float,
) -> list[dict[str, Any]]:
    """检测随频率升高却出现的功率回升，并定位到具体 trial。"""
    group = sorted(rows, key=lambda row: float(row["factor_value"]))
    method_specs = [
        (METHOD_DFPC, "dfpc"),
        (METHOD_NODE_KF_DPC, "node_kf_dpc"),
        (METHOD_KF_DFPC, "kf_dfpc"),
    ]
    events: list[dict[str, Any]] = []
    for method, prefix in method_specs:
        for statistic, aggregate_key, phase_key, distance_key in [
            (
                "final",
                f"{prefix}_final_power_db",
                f"{prefix}_final_phase_std_deg",
                f"{prefix}_final_distance_rmse_m",
            ),
            (
                "tail",
                f"{prefix}_tail_power_db",
                (
                    f"{prefix}_phase_std_deg"
                    if prefix != "dfpc"
                    else f"{prefix}_tail_phase_std_deg"
                ),
                (
                    f"{prefix}_distance_rmse_m"
                    if prefix != "dfpc"
                    else f"{prefix}_tail_distance_rmse_m"
                ),
            ),
        ]:
            trial_key = f"{statistic}_power_db"
            for previous, current in zip(group, group[1:]):
                rebound = float(current[aggregate_key]) - float(previous[aggregate_key])
                if rebound < float(threshold_db):
                    continue
                prev_freq = float(previous["factor_value"])
                curr_freq = float(current["factor_value"])
                prev_trials = {
                    int(row["seed"]): float(row[trial_key])
                    for row in trial_diagnostics
                    if row["method"] == method and float(row["factor_value"]) == prev_freq
                }
                curr_trials = {
                    int(row["seed"]): float(row[trial_key])
                    for row in trial_diagnostics
                    if row["method"] == method and float(row["factor_value"]) == curr_freq
                }
                common_seeds = sorted(set(prev_trials) & set(curr_trials))
                deltas = np.array([curr_trials[seed] - prev_trials[seed] for seed in common_seeds], dtype=np.float64)
                max_index = None if deltas.size == 0 else int(np.argmax(deltas))
                dominant_seed = "" if max_index is None else common_seeds[max_index]
                distance_change = float(current[distance_key]) - float(previous[distance_key])
                trial_delta_mean = np.nan if deltas.size == 0 else float(np.mean(deltas))
                if abs(distance_change) < 1e-9 and np.isfinite(trial_delta_mean) and trial_delta_mean < 0.0:
                    diagnosis = "linear-mean dominated by high-power trials; RMSE unchanged"
                elif abs(distance_change) < 1e-9:
                    diagnosis = "phase wrapping / coherent-sum fluctuation; RMSE unchanged"
                elif distance_change < 0.0:
                    diagnosis = "estimation RMSE improved; inspect KF state and phase together"
                else:
                    diagnosis = "RMSE worsened or mixed effect; inspect block/node diagnostics"
                events.append(
                    {
                        "method": method,
                        "statistic": statistic,
                        "previous_frequency_MHz": prev_freq,
                        "current_frequency_MHz": curr_freq,
                        "previous_power_db": float(previous[aggregate_key]),
                        "current_power_db": float(current[aggregate_key]),
                        "rebound_db": rebound,
                        "phase_std_change_deg": float(current[phase_key]) - float(previous[phase_key]),
                        "distance_rmse_change_m": distance_change,
                        "trials_compared": len(common_seeds),
                        "trial_rebound_fraction": "" if deltas.size == 0 else float(np.mean(deltas > 0.0)),
                        "trial_delta_mean_db": "" if deltas.size == 0 else trial_delta_mean,
                        "trial_delta_median_db": "" if deltas.size == 0 else float(np.median(deltas)),
                        "trial_delta_std_db": "" if deltas.size == 0 else float(np.std(deltas)),
                        "max_trial_rebound_db": "" if deltas.size == 0 else float(np.max(deltas)),
                        "dominant_trial_seed": dominant_seed,
                        "dominant_trial_previous_power_db": "" if max_index is None else prev_trials[dominant_seed],
                        "dominant_trial_current_power_db": "" if max_index is None else curr_trials[dominant_seed],
                        "diagnosis": diagnosis,
                        "previous_point_out_dir": previous["point_out_dir"],
                        "current_point_out_dir": current["point_out_dir"],
                    }
                )
    return events


def write_frequency_rebound_report(events: list[dict[str, Any]], threshold_db: float, out_dir: Path) -> None:
    """生成人可读的回升索引；详细 trial/block 数据保留在配套 CSV。"""
    if events:
        lines = [
            f"| {row['method']} | {row['statistic']} | {row['previous_frequency_MHz']:.0f} → "
            f"{row['current_frequency_MHz']:.0f} | {row['rebound_db']:+.3f} | "
            f"{row['trial_rebound_fraction']} | {row['phase_std_change_deg']:+.3f} | "
            f"{row['distance_rmse_change_m']:+.6f} | {row['dominant_trial_seed']} | {row['diagnosis']} |"
            for row in events
        ]
        table = "\n".join(lines)
    else:
        table = "No rebound exceeded the configured threshold."
    report = f"""# Frequency rebound diagnostics

Detection threshold: {threshold_db:g} dB between adjacent frequencies.

| method | statistic | frequency (MHz) | rebound | trial rebound fraction | phase std change | distance RMSE change | dominant seed | diagnosis |
|---|---|---:|---:|---:|---:|---:|---:|---|
{table}

Diagnostic files:

- `载波频率扫描_异常回升事件.csv`: detected aggregate rebound events.
- `frequency_trial_diagnostics.csv`: per-frequency, per-trial final/tail statistics.
- `frequency_block_trial_diagnostics.csv`: per-frequency, per-trial, per-block final statistics.
- `frequency/point_XX/debug/*.csv`: per-node 3-D errors when the sweep is run with `--debug`.

Interpretation: if distance RMSE is unchanged but power and phase change, the rebound is caused by
carrier-phase wrapping/coherent summation. If only a small fraction of trials rebound, it is Monte Carlo
variance. If most trials rebound together with a lower distance RMSE, investigate the estimator state.
"""
    (out_dir / "载波频率扫描_异常趋势诊断报告.md").write_text(report, encoding="utf-8")


def plot_factor(rows: list[dict[str, Any]], factor: str, out_dir: Path) -> None:
    """给一个因素的 sweep 结果画摘要曲线。"""
    if not rows:
        return
    group = sorted(rows, key=lambda row: float(row["factor_value"]))
    xs = np.array([float(row["factor_value"]) for row in group], dtype=np.float64)
    file_prefix = factor_file_label(factor)
    specs = [
        ("dfpc_tail_power_db", "DPC normalized power (dB)", f"{file_prefix}_DPC尾段归一化功率.png"),
        ("dfpc_tail_gain_over_single_mean_db", "DPC gain over mean single node (dB)", f"{file_prefix}_DPC相对平均单节点增益.png"),
        ("dfpc_tail_phase_std_deg", "DPC phase std (deg)", f"{file_prefix}_DPC相位标准差.png"),
        ("dfpc_tail_phase_rmse_deg", "DPC phase RMSE (deg)", f"{file_prefix}_DPC相位RMSE.png"),
        ("no_algorithm_tail_phase_rmse_deg", "No-algorithm phase RMSE (deg)", f"{file_prefix}_无算法相位RMSE.png"),
        ("dfpc_tail_distance_rmse_m", "Distance RMSE (m)", f"{file_prefix}_DPC距离RMSE.png"),
        ("dfpc_tail_uav_rmse_m", "UAV RMSE (m)", f"{file_prefix}_DPC-UAV位置RMSE.png"),
        ("dfpc_tail_uav_velocity_rmse_mps", "UAV velocity RMSE (m/s)", f"{file_prefix}_DPC-UAV速度RMSE.png"),
        ("kf_dfpc_tail_power_db", "UAV+Node-KF DPC normalized power (dB)", f"{file_prefix}_KF-DPC尾段归一化功率.png"),
        ("node_kf_dpc_tail_power_db", "Node-KF DPC normalized power (dB)", f"{file_prefix}_仅节点KF-DPC尾段归一化功率.png"),
        ("kf_improvement_over_dfpc_db", "KF improvement over DPC (dB)", f"{file_prefix}_KF-DPC相对DPC提升.png"),
        ("kf_dfpc_distance_rmse_m", "UAV+Node-KF distance RMSE (m)", f"{file_prefix}_KF-DPC距离RMSE.png"),
    ]
    for key, ylabel, filename in specs:
        ys = np.array([float(row[key]) for row in group], dtype=np.float64)
        plt.figure(figsize=(9.6, 5.5))
        plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.2)
        plt.xlabel(factor.replace("_", " "))
        plt.ylabel(ylabel)
        plt.title(f"Modular DPC factor sweep: {factor.replace('_', ' ')}")
        if factor == "node_count":
            plt.xticks(xs, [str(int(x)) for x in xs], rotation=35)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()

    comparison_specs = [
        (
            [
                "dfpc_tail_power_db",
                "node_kf_dpc_tail_power_db",
                "kf_dfpc_tail_power_db",
            ],
            "Tail normalized power (dB)",
            f"{file_prefix}_DPC与KF-DPC尾段功率总览.png",
        ),
        (
            [
                "dfpc_tail_phase_std_deg",
                "node_kf_dpc_phase_std_deg",
                "kf_dfpc_phase_std_deg",
            ],
            "Tail residual phase std (deg)",
            f"{file_prefix}_DPC与KF-DPC相位标准差对比.png",
        ),
        (
            [
                "dfpc_tail_node_rmse_m",
                "node_kf_dpc_node_rmse_m",
                "kf_dfpc_node_rmse_m",
            ],
            "Tail node RMSE (m)",
            f"{file_prefix}_DPC与KF-DPC浮标位置RMSE对比.png",
        ),
    ]
    labels = [method for method, _ in PERFORMANCE_METHODS]
    for keys, ylabel, filename in comparison_specs:
        plt.figure(figsize=(10.4, 5.8))
        for key, method_label in zip(keys, labels):
            ys = np.array([float(row[key]) for row in group], dtype=np.float64)
            plt.plot(xs, ys, marker="o", linewidth=2.0, label=method_label)
        if factor == "node_count":
            plt.xscale("log")
        plt.xlabel(factor.replace("_", " "))
        plt.ylabel(ylabel)
        plt.title(f"All-method comparison: {factor.replace('_', ' ')}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()

    if factor == "frequency":
        comparison_specs = [
            (
                ["dfpc_tail_power_db", "node_kf_dpc_tail_power_db", "kf_dfpc_tail_power_db"],
                "Tail normalized power (dB)",
                "载波频率扫描_DPC与KF-DPC尾段功率对比.png",
            ),
            (
                ["dfpc_final_power_db", "node_kf_dpc_final_power_db", "kf_dfpc_final_power_db"],
                "Final normalized power (dB)",
                "载波频率扫描_DPC与KF-DPC最终功率对比.png",
            ),
        ]
        for keys, ylabel, filename in comparison_specs:
            plt.figure(figsize=(10.2, 5.8))
            for key, label in zip(keys, [method for method, _ in PERFORMANCE_METHODS]):
                values = np.array([float(row[key]) for row in group], dtype=np.float64)
                plt.plot(xs, values, marker="o", linewidth=2.0, label=label)
            plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            plt.xlabel("carrier frequency (MHz)")
            plt.ylabel(ylabel)
            plt.title(f"3-D KF/DPC frequency sweep: {ylabel}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / filename, dpi=220)
            plt.close()


def plot_connectivity_convergence(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot sustained convergence time and transient deficit versus connectivity."""
    if not rows:
        return
    for key, ylabel, filename in [
        (
            "distance_rmse_convergence_time_s",
            "Distance-RMSE convergence time (s)",
            "连通度扫描_距离RMSE收敛时间.png",
        ),
        (
            "convergence_time_s_05db",
            "Time to sustained 0.5 dB steady-state band (s)",
            "连通度扫描_DPC与KF-DPC功率收敛时间.png",
        ),
        (
            "transient_deficit_mean_db",
            "Mean first-half transient deficit (dB)",
            "连通度扫描_DPC与KF-DPC瞬态功率缺口.png",
        ),
    ]:
        plt.figure(figsize=(10.4, 5.8))
        for method, _ in PERFORMANCE_METHODS:
            method_rows = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: float(row["factor_value"]),
            )
            plt.plot(
                [float(row["factor_value"]) for row in method_rows],
                [float(row[key]) for row in method_rows],
                marker="o",
                linewidth=2.0,
                label=method,
            )
        plt.xscale("log")
        plt.xlabel("communication-edge probability")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()


def plot_connectivity_time_curves(rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot power and distance-RMSE transients for selected connectivities."""
    if not rows:
        return
    connectivities = sorted({float(row["factor_value"]) for row in rows})
    selected = {connectivities[0], connectivities[len(connectivities) // 2], connectivities[-1]}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True)
    for row_index, (method, _) in enumerate(PERFORMANCE_METHODS):
        for connectivity in sorted(selected):
            curve = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method
                    and float(row["factor_value"]) == connectivity
                ),
                key=lambda row: int(row["global_step"]),
            )
            axes[row_index, 0].plot(
                [int(row["global_step"]) for row in curve],
                [float(row["normalized_power_db"]) for row in curve],
                linewidth=1.8,
                label=f"p={connectivity:g}",
            )
            axes[row_index, 1].plot(
                [int(row["global_step"]) for row in curve],
                [float(row["distance_rmse_m"]) for row in curve],
                linewidth=1.8,
                label=f"p={connectivity:g}",
            )
        axes[row_index, 0].set_title(f"{method}: normalized power")
        axes[row_index, 1].set_title(f"{method}: distance RMSE")
        axes[row_index, 0].set_ylabel("normalized power (dB)")
        axes[row_index, 1].set_ylabel("distance RMSE (m)")
        for axis in axes[row_index]:
            axis.grid(True, alpha=0.3)
            axis.legend()
    axes[-1, 0].set_xlabel("iteration index k")
    axes[-1, 1].set_xlabel("iteration index k")
    figure.tight_layout()
    figure.savefig(out_dir / "连通度扫描_典型连通度功率与距离RMSE迭代曲线.png", dpi=220)
    plt.close(figure)
