#!/usr/bin/env python3
"""Compare cluster relative-noise 0.0 vs 0.2 with the prior mechanism."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(r"D:\git\cc\experiment")
PACKAGE_DIR = EXPERIMENT_DIR / "dfpc_experiment"
FINAL_OUT = EXPERIMENT_DIR / "outputs" / "traj_noise_compare_prior"
WORK_ROOT = EXPERIMENT_DIR / "outputs" / "noise_compare_workers"

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from dfpc_experiment.io_utils import write_rows


POINTS = [
    ("20_000", 20.0, 0.0, "cuda", "worker_cuda_0"),
    ("20_020", 20.0, 0.2, "cuda", "worker_cuda_1"),
    ("100_000", 100.0, 0.0, "cpu", "worker_cpu_0"),
    ("100_020", 100.0, 0.2, "cpu", "worker_cpu_1"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge() -> None:
    rows: list[dict[str, str]] = []
    for point_name, freq, noise, _, worker_name in POINTS:
        worker_dir = WORK_ROOT / worker_name
        summary_path = worker_dir / f"dual_timescale_dpc_{freq:g}mhz_summary.csv"
        for row in read_rows(summary_path):
            row = dict(row)
            row["point_name"] = point_name
            row["frequency_MHz"] = str(freq)
            row["cluster_relative_noise_std"] = str(noise)
            rows.append(row)
    write_rows(FINAL_OUT / "noise_compare_summary.csv", rows)
    print(f"Merged noise comparison into {FINAL_OUT}", flush=True)


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    processes = []
    for point_name, freq, noise, device, worker_name in POINTS:
        worker_dir = WORK_ROOT / worker_name
        worker_dir.mkdir(parents=True, exist_ok=True)
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
            "--cluster_relative_noise_std",
            f"{noise:g}",
            "--out_dir",
            str(worker_dir),
        ]
        stdout = (worker_dir / "worker.log").open("w", encoding="utf-8")
        stderr = (worker_dir / "worker.err.log").open("w", encoding="utf-8")
        print(f"Starting {worker_name}: {freq} MHz noise={noise} on {device}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(EXPERIMENT_DIR), stdout=stdout, stderr=stderr)
        processes.append((worker_name, proc, stdout, stderr))

    failed = []
    for worker_name, proc, stdout, stderr in processes:
        code = proc.wait()
        stdout.close()
        stderr.close()
        print(f"{worker_name} finished with exit code {code}", flush=True)
        if code != 0:
            failed.append(worker_name)

    if failed:
        raise SystemExit(f"Workers failed: {', '.join(failed)}")
    merge()


if __name__ == "__main__":
    main()
