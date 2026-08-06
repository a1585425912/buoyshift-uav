#!/usr/bin/env python3
"""Run high-frequency single points with cluster methods enabled."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(r"D:\git\cc\experiment")
PACKAGE_DIR = EXPERIMENT_DIR / "dfpc_experiment"
FINAL_OUT = EXPERIMENT_DIR / "outputs" / "high_freq_cluster_points"
WORK_ROOT = EXPERIMENT_DIR / "outputs" / "high_freq_cluster_workers"

POINTS = [
    (100.0, "cuda"),
    (200.0, "cuda"),
    (300.0, "cpu"),
    (400.0, "cpu"),
]


def main() -> None:
    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    processes = []
    for freq, device in POINTS:
        point_dir = FINAL_OUT / f"{freq:g}mhz"
        point_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(PACKAGE_DIR / "run_experiment.py"),
            "--N",
            "1000",
            "--fc_mhz",
            f"{freq:g}",
            "--mc_trials",
            "20",
            "--T_long",
            "8",
            "--K",
            "30",
            "--device",
            device,
            "--cluster_mode",
            "node_prior",
            "--uav_kf_prior_mode",
            "dpc_only",
            "--out_dir",
            str(point_dir),
        ]
        stdout = (point_dir / "worker.log").open("w", encoding="utf-8")
        stderr = (point_dir / "worker.err.log").open("w", encoding="utf-8")
        print(f"Starting {freq:g} MHz on {device}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(EXPERIMENT_DIR), stdout=stdout, stderr=stderr)
        processes.append((f"{freq:g} MHz", proc, stdout, stderr))

    failed = []
    for label, proc, stdout, stderr in processes:
        code = proc.wait()
        stdout.close()
        stderr.close()
        print(f"{label} finished with exit code {code}", flush=True)
        if code != 0:
            failed.append(label)

    if failed:
        raise SystemExit(f"Points failed: {', '.join(failed)}")
    print(f"High-frequency cluster points saved under {FINAL_OUT}", flush=True)


if __name__ == "__main__":
    main()
