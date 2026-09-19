# 40 MHz Modular Dual-Timescale DPC Experiment

This is the modular DPC experiment.

Monte Carlo trials: 10. Global consensus device: `cuda`.
Physical duration: 29.950 s; block duration K*Ts=1.5 s; Ts=0.05 s.
Warm-up: 20 shared observation step(s) before formal k=0; warm-up samples are not plotted or included in reported metrics.
Buoy centers move at 0.8 m/s in the 55 deg direction. At every iteration, separated by Ts=0.05 s, a random displacement with std=0.08 m per horizontal axis is sampled; 0.08 of it accumulates into the center and the remainder is the instantaneous offset.
Each buoy independently estimates the UAV trajectory. DPC reaches consensus on
`[bx,vx,by,vy,bz,vz]` and unit flight direction `[dx,dy,dz]`.
UAV+Node-KF DPC filters both UAV and buoy states as `[x,y,z,vx,vy,vz]` and uses 3-D ranges.
The only filtering branch retained is the linear 3-D constant-velocity Kalman filter.
    Power is averaged as linear gain before conversion to dB. Positive power change versus DPC is better.
    PNG curves use a causal EMA with span 21 and show the raw trace faintly; CSV and NPZ remain unsmoothed.
    K is only an output grouping on one continuous physical timeline, so no modulo-K "iteration-axis" figures are generated.
Every method uses an ideal 0/pi polarity choice. Effective residual phases are folded modulo pi into [-90, 90) deg; they are not clipped.
`No Algorithm` is the pure-random reference: it transmits with theta=0 and uses no observations, consensus, or filtering.

| method | tail normalized power | power change vs DPC | gain over mean single node | tail phase std | tail phase RMSE | tail node RMSE | tail distance RMSE | tail UAV RMSE | tail UAV velocity RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Algorithm | -3.92 dB | -1.26 dB | 50.06 dB | 52.01 deg | 52.01 deg |  |  |  |  |
| DPC | -2.66 dB | +0.00 dB | 51.32 dB | 43.95 deg | 43.95 deg | 1.7331225394346874 | 1.0101125085997635 | 0.2476126207332031 | 0.015365968287234669 |
| Node-KF DPC | -0.13 dB | +2.53 dB | 53.85 dB | 9.79 deg | 9.78 deg | 0.16199909901719575 | 0.1767140520774759 | 0.2476126207332031 | 0.015365968287234669 |
| KF-DPC | -0.07 dB | +2.59 dB | 53.91 dB | 7.07 deg | 7.09 deg | 0.16199909901719575 | 0.10661951775003016 | 0.03592930935738078 | 0.005791172758003543 |
| Shore Broadcast | -0.03 dB | +2.62 dB | 53.95 dB | 4.91 deg | 4.91 deg | 0.0 | 0.0 | 0.0 |  |
