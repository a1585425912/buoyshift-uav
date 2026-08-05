#!/usr/bin/env python3
"""Reproduce the four Note3.3 sweeps with one consistent metric definition.

This runner intentionally evaluates only the methods used by the document:

* Random Phase Reference: theta=0; no observation, consensus, or filtering.
* No Algorithm: phase compensation from instantaneous noisy positions.
* DPC: distributed online UAV line estimation plus one consensus step at
  every short-time sample; raw node-position observations are retained.

All methods use an ideal 0/pi polarity choice: effective residual phases are
folded modulo pi into [-pi/2, pi/2) before power and phase metrics are
computed.  Power is averaged in the linear domain before conversion to dB.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_DIR = PACKAGE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from dfpc_experiment import backend
from dfpc_experiment.config import ExperimentConfig
from dfpc_experiment.metrics import evaluate_dfpc, random_phase_reference_metrics
from dfpc_experiment.scenario import (
    advance_truth_one_short_step,
    current_truth,
    geometry_terms,
    init_scene,
    observe_positions,
)
from dfpc_experiment.trajectory import (
    init_line_estimator,
    normalize_trajectory_directions,
    predict_positions,
    update_local_line_estimates,
)


METHODS = ("Random Phase Reference", "No Algorithm", "DPC")
DEFAULT_VALUES = {
    "node_count": "20,30,50,75,100,150,200,300,500,1000,1500,2000",
    "connectivity": "0.0005,0.001,0.002,0.003,0.005,0.008,0.01,0.02,0.03,0.05,0.08,0.10",
    "node_position_noise": "0.1,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6",
    "uav_position_error": "0.1,0.5,1,2,3,5,8,10,15,20,30,40,50,75,100,150,175,200,250,300",
}


def parse_numbers(text: str, integer: bool = False) -> list[float | int]:
    converter = int if integer else float
    return [converter(item.strip()) for item in text.split(",") if item.strip()]


def point_override(factor: str, value: float | int) -> dict[str, float | int]:
    return {
        "node_count": {"N": int(value)},
        "connectivity": {"global_connectivity": float(value)},
        "node_position_noise": {"buoy_center_obs_noise": float(value)},
        "uav_position_error": {"uav_obs_noise": float(value)},
    }[factor]


def run_trial(cfg: ExperimentConfig, seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng_scene = np.random.default_rng(seed)
    rng_obs = np.random.default_rng(seed + 3000003)
    state = init_scene(cfg, rng_scene)
    fc_hz = float(cfg.fc_mhz) * 1e6
    k_const = 2.0 * np.pi / (3e8 / fc_hz)
    total_steps = int(cfg.T_long) * int(cfg.K)
    block_duration_s = float(cfg.K) * float(cfg.Ts)
    eps_trial = rng_scene.normal(
        0.0, np.deg2rad(cfg.system_phase_std_deg), size=cfg.N
    )
    line_estimator = init_line_estimator(cfg.N)
    metrics = {
        method: {
            key: np.zeros(total_steps, dtype=np.float64)
            for key in ("gain_linear", "power_linear", "ideal_power_linear", "single_mean", "phase_std_deg", "phase_rmse_deg")
        }
        for method in METHODS
    }

    for block in range(cfg.T_long):
        if block > 0:
            advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
        time_s = block * block_duration_s
        p_u_true, buoy_true = current_truth(state)
        uav_obs, buoy_obs = observe_positions(cfg, p_u_true, buoy_true, rng_obs)
        line_params = update_local_line_estimates(line_estimator, time_s, uav_obs)

        for iteration in range(cfg.K):
            sidx = block * cfg.K + iteration
            if iteration > 0:
                advance_truth_one_short_step(cfg, state, rng_scene, cfg.Ts)
                time_s = (block * cfg.K + iteration) * float(cfg.Ts)
                p_u_true, buoy_true = current_truth(state)
                uav_obs, buoy_obs = observe_positions(cfg, p_u_true, buoy_true, rng_obs)
                line_params = update_local_line_estimates(line_estimator, time_s, uav_obs)

            line_params = backend.consensus_linear_accel(
                state.W_global, state.W_global_gpu, line_params, 1
            )
            line_params = normalize_trajectory_directions(line_params)
            uav_dfpc = predict_positions(line_params, time_s, cfg.uav_height)
            phi_true, amp, p_ideal, p_single_mean, p_single_best = geometry_terms(
                cfg, p_u_true, buoy_true, k_const
            )
            results = {
                "Random Phase Reference": random_phase_reference_metrics(
                    phi_true, amp, eps_trial, p_ideal, p_single_mean, p_single_best
                ),
                "No Algorithm": evaluate_dfpc(
                    uav_obs, buoy_obs, p_u_true, buoy_true, phi_true, amp,
                    eps_trial, k_const, p_ideal, p_single_mean, p_single_best,
                ),
                "DPC": evaluate_dfpc(
                    uav_dfpc, buoy_obs, p_u_true, buoy_true, phi_true, amp,
                    eps_trial, k_const, p_ideal, p_single_mean, p_single_best,
                ),
            }
            for method, result in results.items():
                metrics[method]["gain_linear"][sidx] = result["gain_linear"]
                metrics[method]["power_linear"][sidx] = result["power_linear"]
                metrics[method]["ideal_power_linear"][sidx] = result["ideal_power_linear"]
                metrics[method]["single_mean"][sidx] = result["single_node_mean_power_linear"]
                metrics[method]["phase_std_deg"][sidx] = result["phase_std_deg"]
                metrics[method]["phase_rmse_deg"][sidx] = result["phase_rmse_deg"]
    return metrics


def run_point(cfg: ExperimentConfig) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    trials = []
    trial_rows: list[dict[str, float | int | str]] = []
    for trial in range(cfg.mc_trials):
        seed = int(cfg.seed) + trial * int(cfg.mc_seed_stride)
        trials.append(run_trial(cfg, seed))
    tail = slice(int(0.8 * cfg.T_long * cfg.K), None)
    summary: dict[str, float] = {}
    for method in METHODS:
        prefix = method.lower().replace(" ", "_")
        gain = np.mean(np.stack([t[method]["gain_linear"] for t in trials]), axis=0)
        power = np.mean(np.stack([t[method]["power_linear"] for t in trials]), axis=0)
        single = np.mean(np.stack([t[method]["single_mean"] for t in trials]), axis=0)
        phase = np.mean(np.stack([t[method]["phase_std_deg"] for t in trials]), axis=0)
        phase_rmse = np.mean(np.stack([t[method]["phase_rmse_deg"] for t in trials]), axis=0)
        summary[f"{prefix}_tail_normalized_power_db"] = float(
            10.0 * np.log10(max(float(np.mean(gain[tail])), 1e-30))
        )
        summary[f"{prefix}_tail_gain_over_single_mean_db"] = float(
            10.0 * np.log10(max(float(np.mean(power[tail]) / np.mean(single[tail])), 1e-30))
        )
        summary[f"{prefix}_tail_phase_std_deg"] = float(np.mean(phase[tail]))
        summary[f"{prefix}_tail_phase_rmse_deg"] = float(np.mean(phase_rmse[tail]))
        for trial_index, trial_metrics in enumerate(trials):
            trial_gain = trial_metrics[method]["gain_linear"]
            trial_phase = trial_metrics[method]["phase_std_deg"]
            trial_phase_rmse = trial_metrics[method]["phase_rmse_deg"]
            trial_rows.append(
                {
                    "trial_index": trial_index,
                    "seed": int(cfg.seed) + trial_index * int(cfg.mc_seed_stride),
                    "method": method,
                    "tail_normalized_power_db": float(10.0 * np.log10(max(float(np.mean(trial_gain[tail])), 1e-30))),
                    "tail_phase_std_deg": float(np.mean(trial_phase[tail])),
                    "tail_phase_rmse_deg": float(np.mean(trial_phase_rmse[tail])),
                }
            )
    summary["dpc_gain_gap_to_ideal_coherent_scaling_db"] = float(
        summary["dpc_tail_gain_over_single_mean_db"] - 20.0 * np.log10(cfg.N)
    )
    return summary, trial_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_factor(rows: list[dict], factor: str, out_dir: Path) -> None:
    xs = np.array([float(row["factor_value"]) for row in rows])
    for suffix, ylabel in (
        ("tail_normalized_power_db", "Normalized power (dB)"),
        ("tail_phase_std_deg", "Residual phase std (deg)"),
        ("tail_phase_rmse_deg", "Residual phase RMSE (deg)"),
    ):
        plt.figure(figsize=(9.6, 5.6))
        for method in METHODS:
            prefix = method.lower().replace(" ", "_")
            ys = np.array([float(row[f"{prefix}_{suffix}"]) for row in rows])
            plt.plot(xs, ys, marker="o", linewidth=2, label=method)
        plt.xlabel(factor.replace("_", " "))
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        if factor == "node_count":
            plt.xscale("log")
        plt.tight_layout()
        plt.savefig(out_dir / f"{factor}_{suffix}.png", dpi=220)
        plt.close()


def write_reference_time_curves(cfg: ExperimentConfig, out_dir: Path) -> None:
    """Save the MC-mean physical-time curves for the document base scene."""
    trials = [
        run_trial(cfg, int(cfg.seed) + trial * int(cfg.mc_seed_stride))
        for trial in range(cfg.mc_trials)
    ]
    rows: list[dict] = []
    step_count = cfg.T_long * cfg.K
    curves: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        gain = np.mean(np.stack([trial[method]["gain_linear"] for trial in trials]), axis=0)
        phase = np.mean(np.stack([trial[method]["phase_std_deg"] for trial in trials]), axis=0)
        phase_rmse = np.mean(np.stack([trial[method]["phase_rmse_deg"] for trial in trials]), axis=0)
        curves[method] = {
            "normalized_power_db": 10.0 * np.log10(np.maximum(gain, 1e-30)),
            "phase_std_deg": phase,
            "phase_rmse_deg": phase_rmse,
        }
    for step in range(step_count):
        for method in METHODS:
            rows.append(
                {
                    "global_step": step,
                    "physical_time_s": step * float(cfg.Ts),
                    "method": method,
                    "normalized_power_db": float(curves[method]["normalized_power_db"][step]),
                    "phase_std_deg": float(curves[method]["phase_std_deg"][step]),
                    "phase_rmse_deg": float(curves[method]["phase_rmse_deg"][step]),
                }
            )
    write_csv(out_dir / "reference_time_curves.csv", rows)
    for key, ylabel in (
        ("normalized_power_db", "Normalized power (dB)"),
        ("phase_std_deg", "Residual phase std (deg)"),
        ("phase_rmse_deg", "Residual phase RMSE (deg)"),
    ):
        plt.figure(figsize=(10.5, 5.8))
        for method in METHODS:
            plt.plot(
                np.arange(step_count) * float(cfg.Ts), curves[method][key],
                linewidth=1.8, label=method,
            )
        plt.xlabel("Physical time (s)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"reference_time_{key}.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", default=",".join(DEFAULT_VALUES))
    parser.add_argument("--mc_trials", type=int, default=20)
    parser.add_argument("--T_long", type=int, default=8)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out_dir", type=Path, required=True)
    for factor, values in DEFAULT_VALUES.items():
        parser.add_argument(f"--{factor}_values", default=values)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = ExperimentConfig(
        N=1000, fc_mhz=20.0, global_connectivity=0.03,
        uav_obs_noise=3.0, buoy_center_obs_noise=1.0,
        mc_trials=args.mc_trials, T_long=args.T_long, K=args.K,
        device=args.device,
    )
    all_rows: list[dict] = []
    for factor in [item.strip() for item in args.factors.split(",") if item.strip()]:
        values = parse_numbers(
            getattr(args, f"{factor}_values"), integer=factor == "node_count"
        )
        rows: list[dict] = []
        diagnostics: list[dict] = []
        for index, value in enumerate(values):
            cfg = replace(base, **point_override(factor, value))
            print(f"[{factor}] {index + 1}/{len(values)} value={value}", flush=True)
            summary, trial_rows = run_point(cfg)
            common = {
                "factor": factor,
                "factor_value": value,
                "N": cfg.N,
                "connectivity": cfg.global_connectivity,
                "uav_position_noise_m": cfg.uav_obs_noise,
                "node_position_noise_m": cfg.buoy_center_obs_noise,
                "frequency_MHz": cfg.fc_mhz,
                "mc_trials": cfg.mc_trials,
                "T_long": cfg.T_long,
                "K": cfg.K,
            }
            rows.append({**common, **summary})
            diagnostics.extend([{**common, **row} for row in trial_rows])
        write_csv(args.out_dir / f"{factor}_summary.csv", rows)
        write_csv(args.out_dir / f"{factor}_trial_diagnostics.csv", diagnostics)
        plot_factor(rows, factor, args.out_dir)
        all_rows.extend(rows)
    write_csv(args.out_dir / "note33_unified_sweeps_summary.csv", all_rows)
    write_reference_time_curves(base, args.out_dir)
    print(f"Saved unified Note3.3 sweeps to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
