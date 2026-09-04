# Experiment 26 - does the Exp-24 winner survive a harder path?

20-step / 0.4 s horizon, 20 ms real-time budget, same Q/R/Qf as Exp 24 on every path.

## Lemniscate (baseline)

| controller | rms_pos_err_mm | max_pos_err_mm | ctrl_energy | lat_median_ms | lat_p95_ms | lat_cold_ms |
| --- | --- | --- | --- | --- | --- | --- |
| Sampling MPC (CEM) | 202.2 | 439.1 | 0.02982 | 28.53 | 34.82 | 27.1 |
| iLQR / RTI-NMPC | 1.336 | 2.404 | 0.004082 | 16.59 | 21.82 | 33.78 |

## Lissajous 3:2

| controller | rms_pos_err_mm | max_pos_err_mm | ctrl_energy | lat_median_ms | lat_p95_ms | lat_cold_ms |
| --- | --- | --- | --- | --- | --- | --- |
| Sampling MPC (CEM) | 172.8 | 314.6 | 0.01925 | 29.74 | 35.35 | 30.85 |
| iLQR / RTI-NMPC | 5.413 | 9.907 | 0.001339 | 16.56 | 21.42 | 62.05 |

## Spiral

| controller | rms_pos_err_mm | max_pos_err_mm | ctrl_energy | lat_median_ms | lat_p95_ms | lat_cold_ms |
| --- | --- | --- | --- | --- | --- | --- |
| Sampling MPC (CEM) | 147.6 | 248.8 | 0.02589 | 30.55 | 39.9 | 28.69 |
| iLQR / RTI-NMPC | 0.1754 | 0.3839 | 0.0003826 | 16.43 | 21.24 | 31.75 |

