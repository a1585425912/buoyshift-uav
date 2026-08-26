# UAV DPC Modular Project

This is a self-contained modular DPC project. It does not import files from
the old experiment folder; the required shared simulation functions were
migrated into `sim_core.py`.

## Modules

- `config.py`: all experiment parameters and CLI options.
- `scenario.py`: node/UAV deployment, motion model, and observation model.
- `trajectory.py`: each buoy independently estimates a 3-D UAV line and its OLS uncertainty; DPC reaches consensus on the line and predicts the full current `[position, velocity]` state.
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
CLI backward compatibility and is not used by the experiment loop. Buoy
displacement follows a Gaussian drift-diffusion model with configurable non-zero
mean speed (5 m/s by default).

```powershell
python uav_dfpc_modular_project/dfpc_experiment/run_experiment.py --mc_trials 20 --T_long 8 --K 30 --device cuda --out_dir uav_dfpc_modular_project/outputs/out_single
```

The experiment retains only the linear 3-D constant-velocity Kalman filter.
`K` controls only the buoy-motion/output block length.  The UAV filter is
initialized from the step-0 position with an uncertain velocity prior.  At
step 1 and every later `Ts`, it predicts from the previous posterior and
consumes only the newly arrived position observation.  Predicted states and
observations are consensused separately, and the update uses the effective
covariance of the consensused observation.  UAV trajectory statistics and KF
state both retain history across long-block boundaries.
Outputs compare the no-algorithm baseline, ordinary DPC, clustered DPC,
UAV+node Kalman DPC, and its clustered position-correction variant.

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
