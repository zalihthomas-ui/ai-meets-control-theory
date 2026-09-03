# Experiment 21 - the grand bake-off [FULL]

| controller | rms_mm | robust_rms_mm | energy | slew | obstacle_steps | sat_frac | mean_latency_ms | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LQR + flatness | 99.9 | 102 | 0.483 | 378 | 28 | 0.724 | 0.0472 | **0.0** |
| Linear MPC (preview) | 243 | 242 | 0.00343 | 0.0639 | 20 | 0 | 11.5 | **0.0** |
| Sampling MPC (learned) | 316 | 275 | 0.0334 | 14.3 | 0 | 0 | 38.4 | **8.0** |
| Imitation policy (BC+PPO) | 41.4 | 35.7 | 0.00402 | 0.058 | 46 | 0 | 0.0853 | **0.0** |
| Imitation + MPC shield | 317 | 283 | 0.0316 | 12.8 | 0 | 0 | 38.4 | **7.9** |
