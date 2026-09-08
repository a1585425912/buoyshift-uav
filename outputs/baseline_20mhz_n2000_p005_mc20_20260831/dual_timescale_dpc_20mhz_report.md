# 20 MHz Modular Dual-Timescale DPC Experiment

This is the modular DPC experiment.

Monte Carlo trials: 20. Global consensus device: `cuda`.
Physical duration: 29.950 s; block duration K*Ts=1.5 s; Ts=0.05 s.
Buoy centers move at 0.3 m/s in the 0 deg direction. At every time step, a random displacement with std=0.03 m per horizontal axis is sampled; 0.05 of it accumulates into the center and the remainder is the instantaneous offset.
Each buoy independently estimates the UAV trajectory. DPC reaches consensus on
`[bx,vx,by,vy,bz,vz]` and unit flight direction `[dx,dy,dz]`.
UAV+Node-KF DPC filters both UAV and buoy states as `[x,y,z,vx,vy,vz]` and uses 3-D ranges.
The only filtering branch retained is the linear 3-D constant-velocity Kalman filter.
    Power is averaged as linear gain before conversion to dB. Positive power change versus DPC is better.
    PNG curves use a causal EMA with span 11 and show the raw trace faintly; CSV and NPZ remain unsmoothed.
    K is only an output grouping on one continuous physical timeline, so no modulo-K "iteration-axis" figures are generated.
Every method uses an ideal 0/pi polarity choice. Effective residual phases are folded modulo pi into [-90, 90) deg; they are not clipped.
`No Algorithm` is the pure-random reference: it transmits with theta=0 and uses no observations, consensus, or filtering.

| method | tail normalized power | power change vs DPC | gain over mean single node | tail phase std | tail phase RMSE | tail node RMSE | tail distance RMSE | tail UAV RMSE | tail UAV velocity RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Algorithm | -3.92 dB | -3.15 dB | 62.10 dB | 51.98 deg | 51.98 deg |  |  |  |  |
| DPC | -0.77 dB | +0.00 dB | 65.25 dB | 24.20 deg | 24.20 deg | 1.7323022454528403 | 1.0007038617169735 | 0.05508146471862178 | 0.003562934780015813 |
| KF-DPC | -0.02 dB | +0.76 dB | 66.00 dB | 3.52 deg | 3.52 deg | 0.13161100372499418 | 0.07717995626444309 | 0.011941427044967324 | 0.001474899185434283 |
