# Experiment 03 - PID vs unstable second-order plant

| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ | RMSE | Energy $E_u$ | Peak $u_{max}$ | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P-only** | 0.275 | 10 | 9.54e+09 | 6.04e+07 | 1.51e+07 | 984 | 10 | 97.6 | Diverged |
| **PD** | 0.389 | 10 | 28.2 | 0.281 | 0.302 | 316 | 10 | 1.59 | Marginal |
| **PID (no AW)** | 0.334 | 6.16 | 53.1 | 3.9e-05 | 0.214 | 262 | 10 | 5.54 | Stable |
| **PID + AW** | 0.362 | 6.16 | 39.4 | 3.89e-05 | 0.192 | 245 | 10 | 1.61 | Stable |

- **Precision:** PID + AW - lowest RMSE (0.192).
- **Energy:** PID + AW - least control effort (245).
- **Diverged:** P-only.
