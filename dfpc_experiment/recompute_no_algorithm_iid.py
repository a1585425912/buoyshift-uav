#!/usr/bin/env python3
"""Re-statistic an existing run with the analytic i.i.d. random-phase baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


THEORETICAL_PHASE_MEAN_DEG = 0.0
THEORETICAL_PHASE_VARIANCE_RAD2 = np.pi**2 / 12.0
THEORETICAL_PHASE_STD_DEG = 90.0 / np.sqrt(3.0)
COHERENCE_FACTOR = 2.0 / np.pi


def recompute(input_csv: Path, output_dir: Path, node_count: int) -> dict[str, float]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, float | int]] = []
    for row in rows:
        ideal = float(row["no_algorithm_ideal_power_linear"])
        single_mean = float(row["no_algorithm_single_node_mean_power_linear"])
        single_best = float(row["no_algorithm_single_node_best_power_linear"])
        sum_amp_sq = node_count * single_mean
        expected_power = sum_amp_sq + COHERENCE_FACTOR**2 * (ideal - sum_amp_sq)
        gain_linear = expected_power / ideal
        gain_over_mean = expected_power / single_mean
        gain_over_best = expected_power / single_best
        out_rows.append(
            {
                "iteration_step": int(row["iteration_step"]),
                "expected_power_linear": expected_power,
                "expected_normalized_gain_linear": gain_linear,
                "expected_normalized_power_db": 10.0 * np.log10(gain_linear),
                "expected_gain_over_single_mean_linear": gain_over_mean,
                "expected_gain_over_single_mean_db": 10.0 * np.log10(gain_over_mean),
                "expected_gain_over_single_best_linear": gain_over_best,
                "expected_gain_over_single_best_db": 10.0 * np.log10(gain_over_best),
                "phase_mean_deg": THEORETICAL_PHASE_MEAN_DEG,
                "phase_variance_rad2": THEORETICAL_PHASE_VARIANCE_RAD2,
                "phase_std_deg": THEORETICAL_PHASE_STD_DEG,
            }
        )

    csv_path = output_dir / "no_algorithm_iid_uniform_iteration_axis.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    tail = out_rows[int(0.8 * len(out_rows)) :]
    summary = {
        "node_count": node_count,
        "iteration_axis_points": len(out_rows),
        "phase_mean_deg": THEORETICAL_PHASE_MEAN_DEG,
        "phase_variance_rad2": THEORETICAL_PHASE_VARIANCE_RAD2,
        "phase_std_deg": THEORETICAL_PHASE_STD_DEG,
        "tail_expected_normalized_power_db": float(
            10.0
            * np.log10(np.mean([r["expected_normalized_gain_linear"] for r in tail]))
        ),
        "tail_expected_gain_over_single_mean_db": float(
            10.0
            * np.log10(
                np.mean([r["expected_gain_over_single_mean_linear"] for r in tail])
            )
        ),
        "tail_expected_gain_over_single_best_db": float(
            10.0
            * np.log10(np.mean([r["expected_gain_over_single_best_linear"] for r in tail]))
        ),
    }
    (output_dir / "no_algorithm_iid_uniform_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = f"""# No-Algorithm i.i.d. Folded-Phase Re-statistics

Assumption: at every snapshot, node residual phases are independent and
identically distributed as `Uniform[-pi/2, pi/2)` after the ideal 0/pi
polarity correction. The result below is analytic, so it contains no
finite-Monte-Carlo fluctuation.

- Nodes: {node_count}
- Aggregated iteration-axis amplitude/path-loss reference points: {len(out_rows)}
- Phase mean: {THEORETICAL_PHASE_MEAN_DEG:.6f} deg
- Phase variance: pi^2/12 = {THEORETICAL_PHASE_VARIANCE_RAD2:.9f} rad^2
- Phase standard deviation: 90/sqrt(3) = {THEORETICAL_PHASE_STD_DEG:.6f} deg
- Tail expected normalized power: {summary['tail_expected_normalized_power_db']:.6f} dB
- Tail expected gain over a mean single node: {summary['tail_expected_gain_over_single_mean_db']:.6f} dB
- Tail expected gain over the best single node: {summary['tail_expected_gain_over_single_best_db']:.6f} dB

For independent folded phases, `E[e^{{j phi}}] = 2/pi`, so
`E[P] = sum_i a_i^2 + (2/pi)^2 * ((sum_i a_i)^2 - sum_i a_i^2)`.
The expected gain over mean single-node power is therefore not exactly `N`;
ideal coherent power is `(sum_i a_i)^2`.
"""
    (output_dir / "no_algorithm_iid_uniform_report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--node_count", type=int, default=1000)
    args = parser.parse_args()
    summary = recompute(args.input_csv, args.output_dir, args.node_count)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
