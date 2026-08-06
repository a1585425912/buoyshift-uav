# 20 MHz Modular Dual-Timescale DPC Experiment

This is the modular DPC experiment.

Monte Carlo trials: 20. Global consensus device: `cuda`.
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
| Random Phase Reference | -3.92 dB | -3.14 dB | 55.22 dB | 51.99 deg | 52.00 deg |  |  |  |  |
| No Algorithm | -3.75 dB | -2.98 dB | 55.38 dB | 51.04 deg | 51.04 deg | 1.73170556782207 | 3.166429097167006 | 5.195682979992946 |  |
| DPC | -0.78 dB | +0.00 dB | 58.36 dB | 24.24 deg | 24.24 deg | 1.73170556782207 | 1.0026949151065279 | 0.15401987809422582 | 0.024896228682008248 |
| Cluster DPC | -0.07 dB | +0.71 dB | 59.07 dB | 7.55 deg | 7.55 deg | 0.4272493113555152 | 0.288684481056592 | 0.15401987809422582 | 0.024896228682008248 |
| UAV+Node-KF DPC | -0.09 dB | +0.68 dB | 59.04 dB | 8.75 deg | 8.76 deg | 0.5394002293612955 | 0.3429442062984889 | 0.03292906492147588 | 0.009192765287405176 |
| Cluster UAV+Node-KF DPC | -0.06 dB | +0.71 dB | 59.07 dB | 7.24 deg | 7.25 deg | 0.4272493113555152 | 0.27496218152192536 | 0.03292906492147588 | 0.009192765287405176 |
