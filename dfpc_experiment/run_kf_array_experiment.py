#!/usr/bin/env python3
"""Full dual-timescale KF-only experiment with multi-antenna buoys.

This entry point deliberately keeps only the two Kalman-filter branches used
by the modular project:

* UAV+Node-KF DPC
* Cluster UAV+Node-KF DPC

For each branch it compares 1/2/4/8 transmit elements per buoy.  Every buoy's
total transmit power is fixed.  Element phases use spherical propagation
distances, a persistent buoy attitude error, common system phase error, and
zero-mean differential element calibration errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_DIR = PACKAGE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from dfpc_experiment import backend
from dfpc_experiment.cluster import (
    cluster_node_localization,
    communication_graph_clusters,
    rotating_cluster_trajectory_correction,
)
from dfpc_experiment.config import ExperimentConfig, validate_config
from dfpc_experiment.kalman import (
    BatchCVKalman3D,
    copy_velocity_state,
    inject_position_state,
    make_cv3d_filter,
    predict_n_steps,
)
from dfpc_experiment.scenario import (
    advance_truth_one_short_step,
    current_truth,
    init_scene,
    observe_positions,
)
from dfpc_experiment.sim_core import effective_phase_errors, sample_phase_std
from dfpc_experiment.trajectory import (
    init_line_estimator,
    normalize_trajectory_directions,
    predict_positions,
    trajectory_state_from_position_velocity,
    update_local_line_estimates,
)


METHOD_KF = "UAV+Node-KF DPC"
METHOD_CLUSTER_KF = "Cluster UAV+Node-KF DPC"
METHODS = (METHOD_KF, METHOD_CLUSTER_KF)


@dataclass(frozen=True)
class ArraySettings:
    sizes: tuple[int, ...] = (1, 2, 4, 8)
    spacing_lambda: float = 0.5
    yaw_error_std_deg: float = 5.0
    element_phase_error_std_deg: float = 3.0


def ula_offsets(n: int, m: int, spacing_m: float, yaw_rad: np.ndarray) -> np.ndarray:
    along = (np.arange(m, dtype=np.float64) - 0.5 * (m - 1)) * spacing_m
    offsets = np.zeros((n, m, 3), dtype=np.float64)
    offsets[:, :, 0] = np.cos(yaw_rad)[:, None] * along[None, :]
    offsets[:, :, 1] = np.sin(yaw_rad)[:, None] * along[None, :]
    return offsets


def make_persistent_array_errors(
    cfg: ExperimentConfig,
    settings: ArraySettings,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    yaw_true = rng.uniform(-np.pi, np.pi, size=cfg.N)
    yaw_est = yaw_true + rng.normal(
        0.0, np.deg2rad(settings.yaw_error_std_deg), size=cfg.N
    )
    element_errors: dict[int, np.ndarray] = {}
    for m in settings.sizes:
        error = rng.normal(
            0.0,
            np.deg2rad(settings.element_phase_error_std_deg),
            size=(cfg.N, m),
        )
        # Only differential calibration error belongs to the subarray.  The
        # persistent common oscillator/system error is modelled separately.
        error -= np.mean(error, axis=1, keepdims=True)
        element_errors[m] = error
    return yaw_true, yaw_est, element_errors


def evaluate_array_dfpc(
    cfg: ExperimentConfig,
    m: int,
    spacing_m: float,
    k_const: float,
    p_u_est: np.ndarray,
    node_est: np.ndarray,
    p_u_true: np.ndarray,
    node_true: np.ndarray,
    yaw_true: np.ndarray,
    yaw_est: np.ndarray,
    common_phase_error: np.ndarray,
    differential_phase_error: np.ndarray,
) -> dict[str, float]:
    true_elements = node_true[:, None, :] + ula_offsets(cfg.N, m, spacing_m, yaw_true)
    est_elements = node_est[:, None, :] + ula_offsets(cfg.N, m, spacing_m, yaw_est)
    d_true = np.linalg.norm(p_u_true[None, None, :] - true_elements, axis=2)
    d_hat = np.linalg.norm(p_u_est[:, None, :] - est_elements, axis=2)
    amp = np.sqrt(cfg.tx_power / m) / np.maximum(d_true, 1e-6) ** cfg.path_loss_alpha
    residual = effective_phase_errors(
        k_const * d_hat,
        k_const * d_true,
        common_phase_error[:, None] + differential_phase_error,
    )
    field = np.sum(amp * np.exp(1j * residual))
    power = float(np.abs(field) ** 2)
    ideal_power = float(np.sum(amp) ** 2)
    return {
        "power_linear": power,
        "ideal_power_linear": ideal_power,
        "gain_linear": power / max(ideal_power, 1e-30),
        "phase_std_deg": float(np.rad2deg(sample_phase_std(residual.ravel()))),
        "element_distance_rmse_m": float(np.sqrt(np.mean((d_hat - d_true) ** 2))),
        "node_rmse_m": float(np.sqrt(np.mean(np.sum((node_est - node_true) ** 2, axis=1)))),
        "uav_rmse_m": float(np.sqrt(np.mean(np.sum((p_u_est - p_u_true[None, :]) ** 2, axis=1)))),
    }


def empty_trial_metrics(settings: ArraySettings, total_steps: int) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    keys = (
        "power_linear",
        "ideal_power_linear",
        "gain_linear",
        "phase_std_deg",
        "element_distance_rmse_m",
        "node_rmse_m",
        "uav_rmse_m",
    )
    return {
        method: {
            m: {key: np.zeros(total_steps, dtype=np.float64) for key in keys}
            for m in settings.sizes
        }
        for method in METHODS
    }


def run_single_trial(
    cfg: ExperimentConfig,
    settings: ArraySettings,
    seed: int,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    rng_scene = np.random.default_rng(seed)
    rng_obs = np.random.default_rng(seed + 3_000_003)
    rng_cluster = np.random.default_rng(seed + 4_000_003)
    rng_array = np.random.default_rng(seed + 7_000_003)

    wavelength = 3e8 / (cfg.fc_mhz * 1e6)
    k_const = 2.0 * np.pi / wavelength
    spacing_m = settings.spacing_lambda * wavelength
    total_steps = cfg.T_long * cfg.K
    block_duration_s = cfg.K * cfg.Ts

    state = init_scene(cfg, rng_scene)
    labels = communication_graph_clusters(state.W_global, cfg.n_clusters, rng_cluster)
    common_phase_error = rng_scene.normal(
        0.0, np.deg2rad(cfg.system_phase_std_deg), size=cfg.N
    )
    yaw_true, yaw_est, element_errors = make_persistent_array_errors(cfg, settings, rng_array)

    line_estimator = init_line_estimator(cfg.N)
    uav_kf_long: BatchCVKalman3D | None = None
    uav_kf_short: BatchCVKalman3D | None = None
    node_kf: BatchCVKalman3D | None = None
    uav_initial_velocity = state.uav_velocity_true[0].copy()
    metrics = empty_trial_metrics(settings, total_steps)

    for block in range(cfg.T_long):
        if block > 0:
            advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
        time_s = block * block_duration_s
        p_u_true, node_true = current_truth(state)
        uav_obs, node_obs = observe_positions(cfg, p_u_true, node_true, rng_obs)
        line_params = update_local_line_estimates(line_estimator, time_s, uav_obs)

        if uav_kf_long is None:
            initial_velocity = uav_initial_velocity + rng_obs.normal(
                0.0, cfg.uav_kf_velocity_init_std, size=3
            )
            uav_kf_long = make_cv3d_filter(
                uav_obs,
                cfg.Ts,
                cfg.uav_obs_noise,
                cfg.uav_kf_accel_std,
                cfg.uav_kf_initial_velocity_std,
                initial_velocity_xyz=initial_velocity,
            )
            uav_kf_short = make_cv3d_filter(
                uav_obs,
                cfg.Ts,
                cfg.uav_obs_noise,
                cfg.uav_kf_accel_std,
                cfg.uav_kf_initial_velocity_std,
                initial_velocity_xyz=initial_velocity,
                velocity_propagate=True,
            )
        else:
            predict_n_steps(uav_kf_long, cfg.K)

        assert uav_kf_short is not None
        copy_velocity_state(uav_kf_short, uav_kf_long)

        if node_kf is None:
            node_kf = make_cv3d_filter(
                node_obs,
                cfg.Ts,
                cfg.buoy_center_obs_noise,
                cfg.buoy_kf_accel_std,
                cfg.buoy_kf_initial_velocity_std,
                initial_velocity_xyz=state.buoy_wave_mean_velocity,
                velocity_propagate=True,
            )

        for iteration in range(cfg.K):
            sidx = block * cfg.K + iteration
            if iteration > 0:
                advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
                time_s = (block * cfg.K + iteration) * cfg.Ts
                p_u_true, node_true = current_truth(state)
                uav_obs, node_obs = observe_positions(cfg, p_u_true, node_true, rng_obs)
                line_params = update_local_line_estimates(line_estimator, time_s, uav_obs)

            line_params = backend.consensus_linear_accel(
                state.W_global, state.W_global_gpu, line_params, 1
            )
            line_params = normalize_trajectory_directions(line_params)
            uav_dfpc = predict_positions(line_params, time_s, cfg.uav_height)

            if sidx > 0:
                uav_kf_short.predict()
            inject_position_state(uav_kf_short, uav_dfpc, cfg.uav_obs_noise)
            fresh_uav_obs = p_u_true + rng_obs.normal(
                0.0, cfg.uav_obs_noise, size=(cfg.N, 3)
            )
            uav_kf_short.update(fresh_uav_obs)
            uav_state = backend.consensus_linear_accel(
                state.W_global, state.W_global_gpu, uav_kf_short.x.copy(), 1
            )
            uav_kf_short.x = uav_state.copy()
            uav_est_kf = uav_state[:, :3]

            if sidx > 0:
                node_kf.predict()
            fresh_node_obs = node_true + rng_obs.normal(
                0.0, cfg.buoy_center_obs_noise, size=(cfg.N, 3)
            )
            node_kf.update(fresh_node_obs)
            node_est_kf = node_kf.positions
            if cfg.cluster_mode == "localization":
                cluster_node_est = cluster_node_localization(
                    node_est_kf,
                    labels,
                    node_true,
                    rng_cluster,
                    alpha=cfg.cluster_alpha,
                    relative_noise_std=cfg.cluster_relative_noise_std,
                    constraint_type=cfg.cluster_constraint_type,
                    iterations=cfg.cluster_localization_iterations,
                )
                cluster_uav_est_kf = uav_est_kf
                cluster_node_est_for_metrics = cluster_node_est
            else:
                cluster_kf_line_params = trajectory_state_from_position_velocity(
                    uav_est_kf,
                    uav_state[:, 3:],
                    time_s,
                )
                cluster_kf_line_params = rotating_cluster_trajectory_correction(
                    cluster_kf_line_params,
                    labels,
                    rng_cluster,
                    cfg.cluster_alpha,
                    cfg.cluster_trajectory_noise_std,
                )
                cluster_kf_line_params = normalize_trajectory_directions(cluster_kf_line_params)
                cluster_uav_est_kf = predict_positions(cluster_kf_line_params, time_s, cfg.uav_height)
                cluster_node_est_for_metrics = node_est_kf

            estimates = {
                METHOD_KF: uav_est_kf,
                METHOD_CLUSTER_KF: cluster_uav_est_kf,
            }
            node_estimates = {
                METHOD_KF: node_est_kf,
                METHOD_CLUSTER_KF: cluster_node_est_for_metrics,
            }
            for method, uav_est in estimates.items():
                for m in settings.sizes:
                    result = evaluate_array_dfpc(
                        cfg,
                        m,
                        spacing_m,
                        k_const,
                        uav_est,
                        node_estimates[method],
                        p_u_true,
                        node_true,
                        yaw_true,
                        yaw_est,
                        common_phase_error,
                        element_errors[m],
                    )
                    for key, value in result.items():
                        metrics[method][m][key][sidx] = value

        if block == 0:
            predict_n_steps(uav_kf_long, cfg.K - 1)
        uav_kf_long.update(uav_est_kf)
        uav_kf_long.x = backend.consensus_linear_accel(
            state.W_global, state.W_global_gpu, uav_kf_long.x.copy(), 1
        ).copy()

    return metrics


def aggregate_trials(
    trials: list[dict[str, dict[int, dict[str, np.ndarray]]]],
    settings: ArraySettings,
) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    total_steps = next(iter(trials[0][METHOD_KF].values()))["power_linear"].size
    aggregate = empty_trial_metrics(settings, total_steps)
    linear_keys = ("power_linear", "ideal_power_linear", "gain_linear")
    other_keys = (
        "phase_std_deg",
        "element_distance_rmse_m",
        "node_rmse_m",
        "uav_rmse_m",
    )
    for method in METHODS:
        for m in settings.sizes:
            for key in (*linear_keys, *other_keys):
                aggregate[method][m][key] = np.mean(
                    np.stack([trial[method][m][key] for trial in trials]), axis=0
                )
            aggregate[method][m]["norm_db"] = 10.0 * np.log10(
                np.maximum(aggregate[method][m]["gain_linear"], 1e-30)
            )
    for method in METHODS:
        baseline = aggregate[method][1]["power_linear"]
        for m in settings.sizes:
            aggregate[method][m]["power_gain_vs_m1_db"] = 10.0 * np.log10(
                np.maximum(aggregate[method][m]["power_linear"] / baseline, 1e-30)
            )
    return aggregate


def tail_summary(
    metrics: dict[str, dict[int, dict[str, np.ndarray]]],
    cfg: ExperimentConfig,
    settings: ArraySettings,
) -> list[dict[str, Any]]:
    total_steps = metrics[METHOD_KF][1]["power_linear"].size
    tail = slice(int(0.8 * total_steps), None)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        p1_tail = float(np.mean(metrics[method][1]["power_linear"][tail]))
        for m in settings.sizes:
            data = metrics[method][m]
            tail_power = float(np.mean(data["power_linear"][tail]))
            tail_ideal = float(np.mean(data["ideal_power_linear"][tail]))
            rows.append(
                {
                    "method": method,
                    "elements_per_buoy": m,
                    "array_aperture_m": (
                        (m - 1) * settings.spacing_lambda * 3e8 / (cfg.fc_mhz * 1e6)
                    ),
                    "tail_norm_power_db": 10.0 * np.log10(max(tail_power / tail_ideal, 1e-30)),
                    "tail_power_gain_vs_m1_db": 10.0 * np.log10(max(tail_power / p1_tail, 1e-30)),
                    "tail_phase_std_deg": float(np.mean(data["phase_std_deg"][tail])),
                    "tail_element_distance_rmse_m": float(np.mean(data["element_distance_rmse_m"][tail])),
                    "tail_node_rmse_m": float(np.mean(data["node_rmse_m"][tail])),
                    "tail_uav_rmse_m": float(np.mean(data["uav_rmse_m"][tail])),
                }
            )
    return rows


def save_outputs(
    cfg: ExperimentConfig,
    settings: ArraySettings,
    metrics: dict[str, dict[int, dict[str, np.ndarray]]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = tail_summary(metrics, cfg, settings)
    with (out_dir / "kf_array_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "kf_array_config.json").open("w", encoding="utf-8") as f:
        json.dump({"experiment": asdict(cfg), "array": asdict(settings)}, f, indent=2, ensure_ascii=False)

    total_steps = cfg.T_long * cfg.K
    time_s = np.arange(total_steps) * cfg.Ts
    curve_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for m in settings.sizes:
            data = metrics[method][m]
            for idx in range(total_steps):
                curve_rows.append(
                    {
                        "time_step": idx,
                        "physical_time_s": time_s[idx],
                        "method": method,
                        "elements_per_buoy": m,
                        "norm_power_db": data["norm_db"][idx],
                        "power_gain_vs_m1_db": data["power_gain_vs_m1_db"][idx],
                        "phase_std_deg": data["phase_std_deg"][idx],
                        "element_distance_rmse_m": data["element_distance_rmse_m"][idx],
                        "node_rmse_m": data["node_rmse_m"][idx],
                        "uav_rmse_m": data["uav_rmse_m"][idx],
                    }
                )
    with (out_dir / "kf_array_curves.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    colors = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red"}
    for method in METHODS:
        fig, ax = plt.subplots(figsize=(11, 6))
        for m in settings.sizes:
            ax.plot(time_s, metrics[method][m]["norm_db"], color=colors.get(m), label=f"M={m}")
        ax.set_xlabel("Physical time (s)")
        ax.set_ylabel("Normalized coherent power (dB)")
        ax.set_title(f"{cfg.fc_mhz:.0f} MHz: {method}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        stem = "cluster_kf" if method == METHOD_CLUSTER_KF else "kf"
        fig.savefig(out_dir / f"{stem}_normalized_power.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    x = np.arange(len(settings.sizes), dtype=float)
    width = 0.36
    for offset, method in [(-width / 2, METHOD_KF), (width / 2, METHOD_CLUSTER_KF)]:
        method_rows = [row for row in rows if row["method"] == method]
        ax.bar(
            x + offset,
            [row["tail_power_gain_vs_m1_db"] for row in method_rows],
            width,
            label=method,
        )
    ax.plot(x, 10.0 * np.log10(np.asarray(settings.sizes)), "k--", marker="o", label="Ideal 10log10(M)")
    ax.set_xticks(x, [str(m) for m in settings.sizes])
    ax.set_xlabel("Transmit elements per buoy")
    ax.set_ylabel("Tail target-power gain vs M=1 (dB)")
    ax.set_title("Multi-antenna gain with fixed total power per buoy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "kf_array_tail_gain.png", dpi=200)
    plt.close(fig)

    table = []
    for row in rows:
        table.append(
            "| {method} | {elements_per_buoy} | {tail_norm_power_db:.3f} | "
            "{tail_power_gain_vs_m1_db:.3f} | {tail_phase_std_deg:.3f} | "
            "{tail_element_distance_rmse_m:.4f} | {tail_node_rmse_m:.4f} | {tail_uav_rmse_m:.4f} |".format(**row)
        )
    report = f"""# {cfg.fc_mhz:.0f} MHz multi-antenna KF-only DPC experiment

Nodes: {cfg.N}; Monte Carlo trials: {cfg.mc_trials}; physical steps: {cfg.T_long * cfg.K}.
Only the standard UAV+Node Kalman-filter branch and its rotating-cluster
position-correction variant are retained. Total transmit power per buoy is
fixed at {cfg.tx_power:.1f} W for every array size.

ULA spacing is {settings.spacing_lambda:.2f} wavelength. Persistent yaw error
is {settings.yaw_error_std_deg:.1f} deg RMS; differential element phase error
is {settings.element_phase_error_std_deg:.1f} deg RMS. The existing common
system phase error remains {cfg.system_phase_std_deg:.1f} deg RMS.

| method | M | normalized power | power gain vs M=1 | phase std | element-distance RMSE | node RMSE | UAV RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}
"""
    (out_dir / "kf_array_report.md").write_text(report, encoding="utf-8")
    return rows


def cfg_defaults_n_clusters() -> int:
    return ExperimentConfig.n_clusters


def cfg_defaults_cluster_alpha() -> float:
    return ExperimentConfig.cluster_alpha


def cfg_defaults_cluster_trajectory_noise_std() -> float:
    return ExperimentConfig.cluster_trajectory_noise_std


def cfg_defaults_cluster_relative_noise_std() -> float:
    return ExperimentConfig.cluster_relative_noise_std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--fc_mhz", type=float, default=50.0)
    parser.add_argument("--T_long", type=int, default=20)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--array_sizes", type=str, default="1,2,4,8")
    parser.add_argument("--spacing_lambda", type=float, default=0.5)
    parser.add_argument("--yaw_error_std_deg", type=float, default=5.0)
    parser.add_argument("--element_phase_error_std_deg", type=float, default=3.0)
    parser.add_argument("--n_clusters", type=int, default=cfg_defaults_n_clusters())
    parser.add_argument("--cluster_alpha", type=float, default=cfg_defaults_cluster_alpha())
    parser.add_argument(
        "--cluster_trajectory_noise_std",
        type=float,
        default=cfg_defaults_cluster_trajectory_noise_std(),
    )
    parser.add_argument(
        "--cluster_relative_noise_std",
        type=float,
        default=cfg_defaults_cluster_relative_noise_std(),
    )
    parser.add_argument(
        "--cluster_mode",
        choices=["trajectory", "localization", "uav_prior"],
        default="localization",
    )
    parser.add_argument(
        "--cluster_constraint_type",
        choices=["vector", "range"],
        default="vector",
    )
    parser.add_argument(
        "--cluster_localization_iterations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cluster_dpc_prior_weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--cluster_uav_prior_noise_std",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("D:/git/cc/workspace/output/modular_kf_array_50mhz_1000nodes_mc20"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = tuple(sorted({int(value) for value in args.array_sizes.split(",") if value.strip()}))
    if not sizes or sizes[0] != 1 or any(value < 1 for value in sizes):
        raise ValueError("array_sizes must contain 1 and only positive integers")
    cfg = ExperimentConfig(
        N=args.N,
        fc_mhz=args.fc_mhz,
        T_long=args.T_long,
        K=args.K,
        mc_trials=args.mc_trials,
        device=args.device,
        n_clusters=args.n_clusters,
        cluster_alpha=args.cluster_alpha,
        cluster_relative_noise_std=args.cluster_relative_noise_std,
        cluster_trajectory_noise_std=args.cluster_trajectory_noise_std,
        cluster_mode=args.cluster_mode,
        cluster_constraint_type=args.cluster_constraint_type,
        cluster_localization_iterations=args.cluster_localization_iterations,
        cluster_dpc_prior_weight=args.cluster_dpc_prior_weight,
        cluster_uav_prior_noise_std=args.cluster_uav_prior_noise_std,
        out_dir=str(args.out_dir),
    )
    validate_config(cfg)
    settings = ArraySettings(
        sizes=sizes,
        spacing_lambda=args.spacing_lambda,
        yaw_error_std_deg=args.yaw_error_std_deg,
        element_phase_error_std_deg=args.element_phase_error_std_deg,
    )
    print(f"Consensus device: {backend.resolve_device(cfg)}", flush=True)
    trials = []
    for trial in range(cfg.mc_trials):
        seed = cfg.seed + trial * cfg.mc_seed_stride
        print(f"KF-array trial {trial + 1}/{cfg.mc_trials} seed={seed}", flush=True)
        trials.append(run_single_trial(cfg, settings, seed))
    metrics = aggregate_trials(trials, settings)
    rows = save_outputs(cfg, settings, metrics, args.out_dir)
    print(f"KF-array outputs saved to: {args.out_dir.resolve()}", flush=True)
    for row in rows:
        print(
            f"{row['method']} M={row['elements_per_buoy']}: "
            f"norm={row['tail_norm_power_db']:.3f} dB, "
            f"gain-vs-M1={row['tail_power_gain_vs_m1_db']:.3f} dB, "
            f"phase={row['tail_phase_std_deg']:.2f} deg",
            flush=True,
        )


if __name__ == "__main__":
    main()
