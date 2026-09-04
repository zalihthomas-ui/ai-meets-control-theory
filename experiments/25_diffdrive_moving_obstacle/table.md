# Experiment 25 - a moving obstacle on the differential-drive path

3 obstacles (2 static + 1 moving, crosses the path near t~18 s), 37 s run, 50 ms step.

| controller | rms_err_mm | max_err_mm | completion_pct | collision_steps | ctrl_energy | lat_median_ms | lat_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure pursuit (blind) | 69.05 | 107.8 | 100 | 44 | 1.118 | 0.0246 | 0.0368 |
| Path LQR (blind) | 96.86 | 230 | 100 | 82 | 1.442 | 0.0551 | 0.06712 |
| Sampling MPC (CEM, obstacle-aware) | 138.9 | 258 | 100 | 36 | 8.246 | 112.3 | 146.4 |
| iLQR / RTI-NMPC (obstacle-aware) | 57.63 | 229.1 | 100 | 69 | 1.292 | 111.3 | 141.7 |

