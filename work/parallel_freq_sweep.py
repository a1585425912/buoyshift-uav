#!/usr/bin/env python3
"""Run the frequency factor sweep in parallel, then merge outputs."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(r"D:\git\cc\experiment")
PACKAGE_DIR = EXPERIMENT_DIR / "dfpc_experiment"
FINAL_OUT = EXPERIMENT_DIR / "outputs" / "sweep_node_freq_20mc"
WORK_ROOT = EXPERIMENT_DIR / "outputs" / "freq_parallel_workers"

if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from dfpc_experiment.io_utils import write_rows
from dfpc_experiment.sweeps import (
    detect_frequency_rebounds,
    plot_factor,
    write_frequency_rebound_report,
)


WORKERS = [
    ("10,20", "cuda", "worker_cuda_0"),
    ("30,50", "cuda", "worker_cuda_1"),
    ("80,100", "cpu", "worker_cpu_0"),
    ("150", "cpu", "worker_cpu_1"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_frequency_outputs() -> None:
    summaries: list[dict[str, str]] = []
    trial_rows: list[dict[str, str]] = []
    block_rows: list[dict[str, str]] = []
    for _, _, worker_name in WORKERS:
        worker_dir = WORK_ROOT / worker_name
        summaries.extend(read_rows(worker_dir / "frequency_summary.csv"))
        trial_rows.extend(read_rows(worker_dir / "frequency_trial_diagnostics.csv"))
        block_rows.extend(read_rows(worker_dir / "frequency_block_trial_diagnostics.csv"))

    summaries.sort(key=lambda row: float(row["factor_value"]))
    for index, row in enumerate(summaries):
        row["point_index"] = str(index)
        row["point_out_dir"] = str(FINAL_OUT / "frequency" / f"point_{index:02d}")

    FINAL_OUT.mkdir(parents=True, exist_ok=True)
    write_rows(FINAL_OUT / "frequency_summary.csv", summaries)
    write_rows(FINAL_OUT / "frequency_trial_diagnostics.csv", trial_rows)
    write_rows(FINAL_OUT / "frequency_block_trial_diagnostics.csv", block_rows)
    plot_factor(summaries, "frequency", FINAL_OUT)
    events = detect_frequency_rebounds(summaries, trial_rows, 0.5)
    write_rows(FINAL_OUT / "frequency_rebound_events.csv", events)
    write_frequency_rebound_report(events, 0.5, FINAL_OUT)

    node_summaries = read_rows(FINAL_OUT / "node_count_summary.csv")
    all_rows = node_summaries + summaries
    write_rows(FINAL_OUT / "modular_dfpc_factor_sweeps_summary.csv", all_rows)
    print(f"Merged frequency sweep outputs into {FINAL_OUT}", flush=True)


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    processes = []
    for values, device, worker_name in WORKERS:
        worker_dir = WORK_ROOT / worker_name
        worker_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(PACKAGE_DIR / "run_sweeps.py"),
            "--factors",
            "frequency",
            "--frequency_values",
            values,
            "--mc_trials",
            "20",
            "--T_long",
            "8",
            "--K",
            "30",
            "--device",
            device,
            "--out_dir",
            str(worker_dir),
        ]
        stdout = (worker_dir / "worker.log").open("w", encoding="utf-8")
        stderr = (worker_dir / "worker.err.log").open("w", encoding="utf-8")
        print(f"Starting {worker_name}: frequency {values} on {device}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(EXPERIMENT_DIR),
            stdout=stdout,
            stderr=stderr,
        )
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
    merge_frequency_outputs()


if __name__ == "__main__":
    main()
