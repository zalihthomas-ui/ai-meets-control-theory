# Experiment 24 - iLQR / RTI-NMPC vs Sampling MPC (CEM)

## Task 1 - cart-pole swing-up (online receding horizon)

horizon 60 steps (1.20 s), |F| <= 20 N, 4 s run

| controller | time_to_upright_s | held_upright | final_angle_deg | peak_force_N | rms_force_N | lat_median_ms | lat_p95_ms | lat_cold_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sampling MPC (CEM) | 1.6 | True | 1.594 | 17.2 | 6.887 | 76.13 | 82.4 | 76.44 |
| iLQR / RTI-NMPC | 1.12 | True | 0.3878 | 20 | 7.386 | 23.82 | 26.34 | 623.6 |

## Task 2 - quadrotor figure-8 tracking (Crazyflie 2.0)

horizon 20 steps, 12 s, real-time budget 20 ms/step

| controller | rms_pos_err_mm | max_pos_err_mm | ctrl_energy | lat_median_ms | lat_p95_ms | lat_cold_ms |
| --- | --- | --- | --- | --- | --- | --- |
| Sampling MPC (CEM) | 202.2 | 439.1 | 0.02982 | 22.08 | 24.16 | 24.11 |
| iLQR / RTI-NMPC | 1.336 | 2.404 | 0.004082 | 11.29 | 13.02 | 23.07 |

_median / p95 latency exclude the cold first solve, reported separately as `lat_cold_ms`._

