# Frequency rebound diagnostics

Detection threshold: 0.5 dB between adjacent frequencies.

| method | statistic | frequency (MHz) | rebound | trial rebound fraction | phase std change | distance RMSE change | dominant seed | diagnosis |
|---|---|---:|---:|---:|---:|---:|---:|---|
No rebound exceeded the configured threshold.

Diagnostic files:

- `frequency_rebound_events.csv`: detected aggregate rebound events.
- `frequency_trial_diagnostics.csv`: per-frequency, per-trial final/tail statistics.
- `frequency_block_trial_diagnostics.csv`: per-frequency, per-trial, per-block final statistics.
- `frequency/point_XX/debug/*.csv`: per-node 3-D errors when the sweep is run with `--debug`.

Interpretation: if distance RMSE is unchanged but power and phase change, the rebound is caused by
carrier-phase wrapping/coherent summation. If only a small fraction of trials rebound, it is Monte Carlo
variance. If most trials rebound together with a lower distance RMSE, investigate the estimator state.
