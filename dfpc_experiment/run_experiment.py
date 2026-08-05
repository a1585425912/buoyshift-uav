#!/usr/bin/env python3
"""Run one modular dual-timescale DPC experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_DIR = PACKAGE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from dfpc_experiment.config import parse_experiment_args
from dfpc_experiment.experiment import run_experiment
from dfpc_experiment.plots import save_outputs


def main() -> None:
    cfg = parse_experiment_args()
    out_dir = Path(cfg.out_dir)
    res = run_experiment(cfg)
    save_outputs(res, out_dir)
    print(f"Modular DPC outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
