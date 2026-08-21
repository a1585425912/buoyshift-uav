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
    METHOD_CLUSTER_DFPC,
    METHOD_CLUSTER_KF_DFPC,
    METHOD_DFPC,
    METHOD_KF_DFPC,
    METHOD_NO_ALG,
    METHOD_RANDOM_REFERENCE,
)
from .experiment import run_experiment
from .io_utils import write_rows


PERFORMANCE_METHODS = [
    (METHOD_DFPC, "dfpc"),
    (METHOD_KF_DFPC, "kf_dfpc"),
    (METHOD_CLUSTER_DFPC, "cluster_dfpc"),
    (METHOD_CLUSTER_KF_DFPC, "cluster_kf_dfpc"),
]


def parse_float_list(text: str) -> list[float]:
    """把命令行里的逗号分隔浮点数解析成列表。"""
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    """把命令行里的逗号分隔整数解析成列表。"""
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
    no_alg_tail_power = float(np.mean(metrics[METHOD_NO_ALG]["power_linear"][tail]))
    reference_tail_gain = float(np.mean(metrics[METHOD_RANDOM_REFERENCE]["gain_linear"][tail]))
    reference_tail_power = float(np.mean(metrics[METHOD_RANDOM_REFERENCE]["power_linear"][tail]))
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
        "uav_position_noise_m": float(res["args"]["uav_obs_noise"]),
        "node_position_noise_m": float(res["args"]["buoy_center_obs_noise"]),
        "mc_trials": int(res["mc_trials"]),
        "T_long": int(res["args"]["T_long"]),
        "K": int(res["args"]["K"]),
        "TL_s": float(res["args"]["K"]) * float(res["args"]["Ts"]),
        "Ts_s": float(res["args"]["Ts"]),
        "buoy_wave_speed_mps": float(res["args"]["buoy_wave_speed"]),
        "buoy_wave_diffusion_m_per_sqrt_s": float(res["args"]["buoy_wave_diffusion"]),
        "no_algorithm_tail_power_db": float(10.0 * np.log10(max(no_alg_tail_gain, 1e-30))),
        "dfpc_tail_power_db": float(10.0 * np.log10(max(dfpc_tail_gain, 1e-30))),
        "dfpc_gain_over_no_algorithm_db": float(10.0 * np.log10(max(dfpc_tail_gain / max(no_alg_tail_gain, 1e-30), 1e-30))),
        "no_algorithm_tail_gain_over_single_mean_db": float(10.0 * np.log10(max(no_alg_tail_power / max(single_mean_tail_power, 1e-30), 1e-30))),
        "dfpc_tail_gain_over_single_mean_db": float(10.0 * np.log10(max(dfpc_tail_power / max(single_mean_tail_power, 1e-30), 1e-30))),
        "dfpc_tail_gain_over_single_best_db": float(10.0 * np.log10(max(dfpc_tail_power / max(single_best_tail_power, 1e-30), 1e-30))),
        "no_algorithm_tail_phase_std_deg": float(np.nanmean(metrics[METHOD_NO_ALG]["phase_std_deg"][tail])),
        "no_algorithm_tail_phase_rmse_deg": float(np.nanmean(metrics[METHOD_NO_ALG]["phase_rmse_deg"][tail])),
        "random_reference_tail_power_db": float(10.0 * np.log10(max(reference_tail_gain, 1e-30))),
        "random_reference_gain_over_single_mean_db": float(10.0 * np.log10(max(reference_tail_power / max(single_mean_tail_power, 1e-30), 1e-30))),
        "random_reference_tail_phase_std_deg": float(np.nanmean(metrics[METHOD_RANDOM_REFERENCE]["phase_std_deg"][tail])),
        "random_reference_tail_phase_rmse_deg": float(np.nanmean(metrics[METHOD_RANDOM_REFERENCE]["phase_rmse_deg"][tail])),
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
    for method, prefix in PERFORMANCE_METHODS[2:]:
        method_tail_gain = float(np.mean(metrics[method]["gain_linear"][tail]))
        method_tail_power = float(np.mean(metrics[method]["power_linear"][tail]))
        row.update(
            {
                f"{prefix}_tail_power_db": float(
                    10.0 * np.log10(max(method_tail_gain, 1e-30))
                ),
                f"{prefix}_gain_over_single_mean_db": float(
                    10.0
                    * np.log10(
                        max(method_tail_power / max(single_mean_tail_power, 1e-30), 1e-30)
                    )
                ),
                f"{prefix}_tail_phase_std_deg": float(
                    np.nanmean(metrics[method]["phase_std_deg"][tail])
                ),
                f"{prefix}_tail_phase_rmse_deg": float(
                    np.nanmean(metrics[method]["phase_rmse_deg"][tail])
                ),
                f"{prefix}_tail_distance_rmse_m": float(
                    np.nanmean(metrics[method]["distance_rmse"][tail])
                ),
                f"{prefix}_tail_node_rmse_m": float(
                    np.nanmean(metrics[method]["node_rmse"][tail])
                ),
                f"{prefix}_tail_uav_rmse_m": float(
                    np.nanmean(metrics[method]["uav_rmse"][tail])
                ),
                f"{prefix}_final_power_db": float(metrics[method]["norm_db"][-1]),
                f"{prefix}_final_phase_std_deg": float(metrics[method]["phase_std_deg"][-1]),
                f"{prefix}_final_distance_rmse_m": float(metrics[method]["distance_rmse"][-1]),
            }
        )
    return row


def convergence_summary(
    res: dict[str, Any],
    method: str,
    tolerance_db: float = 0.5,
) -> dict[str, float | int]:
    """Measure sustained convergence to a method's own steady-state band."""
    gain = np.asarray(res["metrics"][method]["gain_linear"], dtype=np.float64)
    total = gain.size
    window = max(1, min(int(res["args"]["K"]), total))
    tail = slice(int(0.8 * total), None)
    tail_db = float(10.0 * np.log10(max(float(np.mean(gain[tail])), 1e-30)))
    rolling = np.convolve(gain, np.ones(window) / window, mode="valid")
    rolling_db = 10.0 * np.log10(np.maximum(rolling, 1e-30))
    threshold = tail_db - float(tolerance_db)
    hold = max(2, min(window // 2, rolling_db.size))
    convergence_step = total
    for index in range(rolling_db.size - hold + 1):
        if np.all(rolling_db[index : index + hold] >= threshold):
            convergence_step = index + window - 1
            break
    norm_db = 10.0 * np.log10(np.maximum(gain, 1e-30))
    transient_end = max(1, total // 2)
    transient_deficit = np.maximum(tail_db - norm_db[:transient_end], 0.0)
    return {
        "tail_power_db": tail_db,
        "convergence_step_05db": convergence_step,
        "convergence_time_s_05db": convergence_step * float(res["args"]["Ts"]),
        "transient_deficit_mean_db": float(np.mean(transient_deficit)),
        "first_block_power_db": float(
            10.0 * np.log10(max(float(np.mean(gain[:window])), 1e-30))
        ),
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
        for key, val in overrides.items():
            setattr(point_cfg, key, val)
        print(f"[{factor}] {idx + 1} {label}", flush=True)
        res = run_experiment(point_cfg)
        summary = summarize_result(res, factor, value, label)
        summary.update({"point_index": idx, "point_out_dir": str(point_dir)})
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
                        }
                    )
        if point_cfg.debug:
            res["debug_recorder"].write(point_dir)
    write_rows(out_dir / f"{factor}_summary.csv", rows)
    write_rows(out_dir / f"{factor}_trial_diagnostics.csv", trial_diagnostics)
    write_rows(out_dir / f"{factor}_block_trial_diagnostics.csv", block_diagnostics)
    if factor == "node_count":
        write_rows(
            out_dir / "node_count_scaling_diagnostics.csv",
            node_count_scaling_diagnostics(rows),
        )
    if factor == "connectivity":
        write_rows(out_dir / "connectivity_convergence_summary.csv", connectivity_diagnostics)
        write_rows(out_dir / "connectivity_time_curves.csv", connectivity_time_rows)
        plot_connectivity_convergence(connectivity_diagnostics, out_dir)
        plot_connectivity_time_curves(connectivity_time_rows, out_dir)
    if factor == "frequency":
        events = detect_frequency_rebounds(rows, trial_diagnostics, rebound_threshold_db)
        write_rows(out_dir / "frequency_rebound_events.csv", events)
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
        (METHOD_KF_DFPC, "kf_dfpc"),
        (METHOD_CLUSTER_DFPC, "cluster_dfpc"),
        (METHOD_CLUSTER_KF_DFPC, "cluster_kf_dfpc"),
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
                    if prefix == "kf_dfpc"
                    else f"{prefix}_tail_phase_std_deg"
                ),
                (
                    f"{prefix}_distance_rmse_m"
                    if prefix == "kf_dfpc"
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

- `frequency_rebound_events.csv`: detected aggregate rebound events.
- `frequency_trial_diagnostics.csv`: per-frequency, per-trial final/tail statistics.
- `frequency_block_trial_diagnostics.csv`: per-frequency, per-trial, per-block final statistics.
- `frequency/point_XX/debug/*.csv`: per-node 3-D errors when the sweep is run with `--debug`.

Interpretation: if distance RMSE is unchanged but power and phase change, the rebound is caused by
carrier-phase wrapping/coherent summation. If only a small fraction of trials rebound, it is Monte Carlo
variance. If most trials rebound together with a lower distance RMSE, investigate the estimator state.
"""
    (out_dir / "frequency_rebound_debug_report.md").write_text(report, encoding="utf-8")


def plot_factor(rows: list[dict[str, Any]], factor: str, out_dir: Path) -> None:
    """给一个因素的 sweep 结果画摘要曲线。"""
    if not rows:
        return
    group = sorted(rows, key=lambda row: float(row["factor_value"]))
    xs = np.array([float(row["factor_value"]) for row in group], dtype=np.float64)
    specs = [
        ("dfpc_tail_power_db", "DPC normalized power (dB)", f"{factor}_dpc_tail_power_db.png"),
        ("random_reference_tail_power_db", "Random-reference normalized power (dB)", f"{factor}_random_reference_tail_power_db.png"),
        ("dfpc_tail_gain_over_single_mean_db", "DPC gain over mean single node (dB)", f"{factor}_gain_over_single_mean_db.png"),
        ("dfpc_tail_phase_std_deg", "DPC phase std (deg)", f"{factor}_dpc_phase_std_deg.png"),
        ("dfpc_tail_phase_rmse_deg", "DPC phase RMSE (deg)", f"{factor}_dpc_phase_rmse_deg.png"),
        ("random_reference_tail_phase_std_deg", "Random-reference phase std (deg)", f"{factor}_random_reference_phase_std_deg.png"),
        ("no_algorithm_tail_phase_rmse_deg", "No-algorithm phase RMSE (deg)", f"{factor}_no_algorithm_phase_rmse_deg.png"),
        ("dfpc_tail_distance_rmse_m", "Distance RMSE (m)", f"{factor}_dfpc_distance_rmse.png"),
        ("dfpc_tail_uav_rmse_m", "UAV RMSE (m)", f"{factor}_dfpc_uav_rmse.png"),
        ("dfpc_tail_uav_velocity_rmse_mps", "UAV velocity RMSE (m/s)", f"{factor}_dfpc_uav_velocity_rmse.png"),
        ("kf_dfpc_tail_power_db", "UAV+Node-KF DPC normalized power (dB)", f"{factor}_kf_dpc_tail_power_db.png"),
        ("kf_improvement_over_dfpc_db", "KF improvement over DPC (dB)", f"{factor}_kf_improvement_over_dpc_db.png"),
        ("kf_dfpc_distance_rmse_m", "UAV+Node-KF distance RMSE (m)", f"{factor}_kf_dfpc_distance_rmse.png"),
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
                "kf_dfpc_tail_power_db",
                "cluster_dfpc_tail_power_db",
                "cluster_kf_dfpc_tail_power_db",
            ],
            "Tail normalized power (dB)",
            f"{factor}_all_methods_tail_power_db.png",
        ),
        (
            [
                "dfpc_tail_phase_std_deg",
                "kf_dfpc_phase_std_deg",
                "cluster_dfpc_tail_phase_std_deg",
                "cluster_kf_dfpc_tail_phase_std_deg",
            ],
            "Tail residual phase std (deg)",
            f"{factor}_all_methods_phase_std_deg.png",
        ),
        (
            [
                "dfpc_tail_node_rmse_m",
                "kf_dfpc_node_rmse_m",
                "cluster_dfpc_tail_node_rmse_m",
                "cluster_kf_dfpc_tail_node_rmse_m",
            ],
            "Tail node RMSE (m)",
            f"{factor}_all_methods_node_rmse.png",
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
                "dfpc_tail_power_db",
                "kf_dfpc_tail_power_db",
                "Tail normalized power (dB)",
                "frequency_dfpc_vs_kf_tail_power_db.png",
            ),
            (
                "dfpc_final_power_db",
                "kf_dfpc_final_power_db",
                "Final normalized power (dB)",
                "frequency_dfpc_vs_kf_final_power_db.png",
            ),
        ]
        for dfpc_key, kf_key, ylabel, filename in comparison_specs:
            dfpc_values = np.array([float(row[dfpc_key]) for row in group], dtype=np.float64)
            kf_values = np.array([float(row[kf_key]) for row in group], dtype=np.float64)
            plt.figure(figsize=(10.2, 5.8))
            plt.plot(xs, dfpc_values, marker="o", linewidth=2.0, label="DPC")
            plt.plot(xs, kf_values, marker="o", linewidth=2.0, label="UAV+Node-KF DPC")
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
            "convergence_time_s_05db",
            "Time to sustained 0.5 dB steady-state band (s)",
            "connectivity_all_methods_convergence_time.png",
        ),
        (
            "transient_deficit_mean_db",
            "Mean first-half transient deficit (dB)",
            "connectivity_all_methods_transient_deficit.png",
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
    """Plot low/medium/high connectivity time curves for each research method."""
    if not rows:
        return
    connectivities = sorted({float(row["factor_value"]) for row in rows})
    selected = {connectivities[0], connectivities[len(connectivities) // 2], connectivities[-1]}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True)
    for axis, (method, _) in zip(axes.flat, PERFORMANCE_METHODS):
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
            axis.plot(
                [float(row["physical_time_s"]) for row in curve],
                [float(row["normalized_power_db"]) for row in curve],
                linewidth=1.8,
                label=f"p={connectivity:g}",
            )
        axis.set_title(method)
        axis.set_ylabel("normalized power (dB)")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1, 0].set_xlabel("physical time (s)")
    axes[-1, 1].set_xlabel("physical time (s)")
    figure.tight_layout()
    figure.savefig(out_dir / "connectivity_selected_time_curves.png", dpi=220)
    plt.close(figure)
