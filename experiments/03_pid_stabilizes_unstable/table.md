# Experiment 03 - PID vs unstable second-order plant

| controller | rise_time | settling_time | peak_overshoot_pct | steady_state_error | iae | itae | control_energy | peak_control | saturation_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P-only | 0.275 | 10 | 9.541e+09 | 6.037e+07 | 4.771e+07 | 4.532e+08 | 984.3 | 10 | 97.62 |
| PD | 0.389 | 10 | 28.19 | 0.2812 | 2.811 | 13.63 | 316.3 | 10 | 1.595 |
| PID (no AW) | 0.334 | 6.162 | 53.08 | 3.898e-05 | 0.9274 | 1.043 | 262.2 | 10 | 5.545 |
| PID + AW | 0.362 | 6.161 | 39.43 | 3.895e-05 | 0.7979 | 0.876 | 245.1 | 10 | 1.605 |
