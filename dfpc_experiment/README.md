# UAV DPC Modular Project

This is a self-contained modular DPC project. It does not import files from
the old experiment folder; the required shared simulation functions were
migrated into `sim_core.py`.

## Modules

- `config.py`: all experiment parameters and CLI options.
- `scenario.py`: node/UAV deployment, motion model, and observation model.
- `trajectory.py`: each buoy uses its continuous observation history to estimate the current 3-D UAV `[position, velocity]` state and uncertainty.
- `dpc.py`: communication-matrix state fusion, spatial convergence tracking, phase output, and the update-then-communicate KF-DPC step.
- `kalman.py`: batched 3-D constant-velocity Kalman filters; UAV and buoy filters initialize at step 0 and remain recursive.
- `metrics.py`: DPC power, phase error, RMSE, and single-node gain metrics.
- `experiment.py`: single-trial and Monte Carlo experiment loops.
- `plots.py`: CSV, NPZ, PNG, and Markdown outputs.
- `sweeps.py`: factor-sweep helpers.
- `debug_tools.py`: step-level debug snapshots.
- `sim_core.py`: migrated minimal simulation utilities.
- `run_experiment.py`: entry point for one experiment.
- `run_sweeps.py`: entry point for sensitivity sweeps.

## Run one experiment

`Ts` controls buoy integration, and each long block advances exactly `K` short
physical steps, so the effective block duration is `K*Ts`. `TL` is kept only for
CLI backward compatibility and is not used by the experiment loop. Each buoy
center moves continuously with a fixed current speed and direction. At every
iteration, separated from the next one by `Ts`, a small random displacement is sampled; a configurable fraction
accumulates into the center and the remainder is the instantaneous offset.

```powershell
python uav_dfpc_modular_project/dfpc_experiment/run_experiment.py --mc_trials 20 --T_long 8 --K 30 --device cuda --out_dir uav_dfpc_modular_project/outputs/out_single
```

The experiment retains only the linear 3-D constant-velocity Kalman filter.
`K` controls only the buoy-motion/output block length. The UAV and node filters
are initialized once at physical step 0. At every later `Ts`, each UAV filter
predicts from its own previous posterior and updates from the node's current
trajectory-state observation; only then does `W` fuse the UAV posterior states
and covariances. Node states are distinct physical targets and are therefore
not averaged across buoys. Trajectory statistics and both KF states retain
history across long-block boundaries. Convergence is simultaneous network
agreement in UAV position and velocity, not state change between iterations.
Outputs compare the pure-random `No Algorithm` baseline with the two retained
algorithms: DPC and KF-DPC.

## Run with debug snapshots

```powershell
python uav_dfpc_modular_project/dfpc_experiment/run_experiment.py --mc_trials 1 --T_long 2 --K 5 --debug --debug_blocks 0,-1 --debug_iterations 0,-1 --debug_max_nodes 20 --out_dir uav_dfpc_modular_project/outputs/out_debug
```

Debug outputs:

- `debug/debug_step_summary.csv`: one row per captured block/iteration.
- `debug/debug_node_samples.csv`: per-node samples with true/estimated distance and residual phase.

## Run selected sweeps

```powershell
python uav_dfpc_modular_project/dfpc_experiment/run_sweeps.py --factors node_count,connectivity --mc_trials 20 --T_long 8 --K 30 --device cuda --out_dir uav_dfpc_modular_project/outputs/out_sweeps
```
