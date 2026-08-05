#!/usr/bin/env python3
"""Run modular DPC factor sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_DIR = PACKAGE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from dfpc_experiment.config import ExperimentConfig, add_common_args
from dfpc_experiment.io_utils import write_rows
from dfpc_experiment.sweeps import run_factor


DEFAULT_VALUES = {
    "node_count": "20,30,50,75,100,150,200,300,500,1000,1500,2000",
    "connectivity": "0.0005,0.001,0.002,0.003,0.005,0.008,0.01,0.02,0.03,0.05,0.08,0.10",
    "frequency": "10,20,30,50,80,100,150",
    "node_position_noise": "0.1,0.3,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,6.0",
    "uav_position_error": "0.1,0.5,1,2,3,5,8,10,20,50,100,150,175,200",
    "wave_speed": "0.1,0.5,1,2,3,5",
}


def parse_args() -> tuple[ExperimentConfig, list[str], dict[str, str], float]:
    parser = argparse.ArgumentParser(description="Modular DPC factor sweeps")
    add_common_args(parser)
    parser.add_argument("--factors", type=str, default="node_count,connectivity,frequency,node_position_noise,uav_position_error,wave_speed")
    parser.add_argument(
        "--rebound_threshold_db",
        type=float,
        default=0.5,
        help="adjacent-frequency power increase that triggers a diagnostic event",
    )
    for factor, default in DEFAULT_VALUES.items():
        parser.add_argument(f"--{factor}_values", type=str, default=default)
    ns = parser.parse_args()
    factors = [x.strip() for x in ns.factors.split(",") if x.strip()]
    values = {factor: getattr(ns, f"{factor}_values") for factor in DEFAULT_VALUES}
    ns_dict = vars(ns)
    ns_dict["enable_cluster"] = not bool(ns_dict.pop("disable_cluster", False))
    cfg_keys = ExperimentConfig.__dataclass_fields__.keys()
    cfg = ExperimentConfig(**{key: ns_dict[key] for key in cfg_keys})
    return cfg, factors, values, float(ns.rebound_threshold_db)


def main() -> None:
    cfg, factors, values, rebound_threshold_db = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for factor in factors:
        rows = run_factor(cfg, factor, values[factor], out_dir, rebound_threshold_db)
        all_rows.extend(rows)
    write_rows(out_dir / "modular_dfpc_factor_sweeps_summary.csv", all_rows)
    print(f"Modular factor sweep outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
