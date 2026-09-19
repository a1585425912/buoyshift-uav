# 50 MHz Modular Dual-Timescale DPC Experiment

This is the modular DPC experiment.

Monte Carlo trials: 20. Global consensus device: `cuda`.
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
| No Algorithm | -3.92 dB | -0.60 dB | 62.10 dB | 51.94 deg | 51.94 deg |  |  |  |  |
| DPC | -3.31 dB | +0.00 dB | 62.71 dB | 48.40 deg | 48.40 deg | 1.7322730720829116 | 1.0033266909736538 | 0.13898139739854568 | 0.008655840424421912 |
| KF-DPC | -0.09 dB | +3.23 dB | 65.94 dB | 8.04 deg | 8.05 deg | 0.16204126839822658 | 0.10504290669423186 | 0.019705503524146666 | 0.0032077458113168628 |
