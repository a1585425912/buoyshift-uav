# 400 MHz Modular Dual-Timescale DPC Experiment

This is the modular DPC experiment.

Monte Carlo trials: 20. Global consensus device: `cpu`.
Physical duration: 11.950 s; block duration K*Ts=1.5 s; Ts=0.05 s.
Buoy displacement over dt: N(mean speed=5 m/s at 0 deg, diffusion=0.5 m/sqrt(s)).
Each buoy independently estimates the UAV trajectory. DPC reaches consensus on
`[bx,vx,by,vy,bz,vz]` and unit flight direction `[dx,dy,dz]`.
UAV+Node-KF DPC filters both UAV and buoy states as `[x,y,z,vx,vy,vz]` and uses 3-D ranges.
The only filtering branch retained is the linear 3-D constant-velocity Kalman filter.
Power is averaged as linear gain before conversion to dB. Positive power change versus DPC is better.
Every method uses an ideal 0/pi polarity choice. Effective residual phases are folded modulo pi into [-90, 90) deg; they are not clipped.
`Random Phase Reference` transmits with theta=0 and uses no observations. `No Algorithm` uses each node's instantaneous noisy UAV/node positions without consensus or filtering.

| method | tail normalized power | power change vs DPC | gain over mean single node | tail phase std | tail phase RMSE | tail node RMSE | tail distance RMSE | tail UAV RMSE | tail UAV velocity RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Phase Reference | -3.92 dB | -0.01 dB | 55.21 dB | 51.98 deg | 51.98 deg |  |  |  |  |
| No Algorithm | -3.91 dB | -0.00 dB | 55.22 dB | 51.96 deg | 51.96 deg | 1.73170556782207 | 3.166429097167006 | 5.195682979992946 |  |
| DPC | -3.91 dB | +0.00 dB | 55.22 dB | 51.96 deg | 51.96 deg | 1.73170556782207 | 1.002694910276606 | 0.15401986434156836 | 0.024896226443692396 |
| Cluster DPC | -3.90 dB | +0.02 dB | 55.24 dB | 51.96 deg | 51.96 deg | 0.4272493113555152 | 0.28868447823970694 | 0.15401986434156836 | 0.024896226443692396 |
| UAV+Node-KF DPC | -3.92 dB | -0.00 dB | 55.22 dB | 51.97 deg | 51.96 deg | 0.5394002293612955 | 0.3429441585971347 | 0.032937151200700676 | 0.009192909704661724 |
| Cluster UAV+Node-KF DPC | -3.88 dB | +0.03 dB | 55.26 dB | 51.88 deg | 51.88 deg | 0.4272493113555152 | 0.2749618977423503 | 0.032937151200700676 | 0.009192909704661724 |
