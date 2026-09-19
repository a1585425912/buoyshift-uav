#!/usr/bin/env python3
"""Sweep shore-station broadcast accuracy or carrier frequency."""

from __future__ import annotations

import argparse
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

from dfpc_experiment.config import ExperimentConfig, add_common_args
from dfpc_experiment.experiment import run_experiment
from dfpc_experiment.io_utils import write_rows
from dfpc_experiment.metrics import tail_summary
from dfpc_experiment.plots import save_outputs


def parse_values(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def plot_summary(
    rows: list[dict[str, object]],
    out_dir: Path,
    x_key: str,
    xlabel: str,
    title: str,
    file_prefix: str,
) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    specs = [
        ("tail_mean_power_db", "Tail normalized power (dB)", f"{file_prefix}功率.png"),
        ("tail_mean_phase_rmse_deg", "Tail residual phase RMSE (deg)", f"{file_prefix}相位RMSE.png"),
        ("tail_mean_distance_rmse_m", "Tail distance RMSE (m)", f"{file_prefix}距离RMSE.png"),
    ]
    for key, ylabel, filename in specs:
        plt.figure(figsize=(10.2, 5.8))
        for method in methods:
            group = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: float(row[x_key]),
            )
            xs = np.array([float(row[x_key]) for row in group])
            values = [row[key] for row in group]
            if any(value == "" for value in values):
                continue
            ys = np.array([float(value) for value in values])
            plt.plot(xs, ys, marker="o", linewidth=2.0, label=method)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shore-station broadcast accuracy sweep")
    add_common_args(parser)
    parser.add_argument(
        "--broadcast_noise_values",
        default="0,0.05,0.1,0.25,0.5,1.0",
        help="comma-separated 3-D coordinate noise std values shared by UAV and buoy broadcasts",
    )
    parser.add_argument(
        "--frequency_values",
        default="",
        help="optional comma-separated MHz values; requires exactly one broadcast noise value",
    )
    ns = parser.parse_args()
    cfg_keys = ExperimentConfig.__dataclass_fields__.keys()
    cfg = ExperimentConfig(**{key: vars(ns)[key] for key in cfg_keys})
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_values = parse_values(ns.broadcast_noise_values)
    frequency_values = parse_values(ns.frequency_values)
    if frequency_values:
        if len(noise_values) != 1:
            raise ValueError("frequency sweep requires exactly one broadcast noise value")
        points = [(frequency, noise_values[0]) for frequency in frequency_values]
        x_key = "frequency_MHz"
        xlabel = "carrier frequency (MHz)"
        title = "Shore broadcast vs Node-KF: frequency sweep"
        file_prefix = "岸基广播_频率扫描_"
        summary_filename = "岸基广播_频率扫描汇总.csv"
    else:
        points = [(cfg.fc_mhz, noise_std) for noise_std in noise_values]
        x_key = "broadcast_position_noise_m"
        xlabel = "shore broadcast position error std (m)"
        title = "Shore-station broadcast accuracy sweep"
        file_prefix = "岸基广播_定位误差与"
        summary_filename = "岸基广播_定位误差扫描汇总.csv"

    rows: list[dict[str, object]] = []
    for index, (frequency_mhz, noise_std) in enumerate(points):
        point_dir = out_dir / f"point_{index:02d}_{frequency_mhz:g}MHz_{noise_std:g}m"
        point_cfg = replace(
            cfg,
            fc_mhz=frequency_mhz,
            shore_broadcast_enabled=True,
            shore_broadcast_uav_noise=noise_std,
            shore_broadcast_node_noise=noise_std,
            out_dir=str(point_dir),
        )
        print(f"[shore broadcast] {frequency_mhz:g} MHz, {noise_std:g} m", flush=True)
        result = run_experiment(point_cfg)
        save_outputs(result, point_dir)
        for method in result["methods"]:
            rows.append(
                {
                    "broadcast_position_noise_m": noise_std,
                    "frequency_MHz": result["frequency_MHz"],
                    "mc_trials": result["mc_trials"],
                    **tail_summary(result, method),
                    "point_out_dir": str(point_dir),
                }
            )

    write_rows(out_dir / summary_filename, rows)
    plot_summary(rows, out_dir, x_key, xlabel, title, file_prefix)
    print(f"Shore broadcast sweep outputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
