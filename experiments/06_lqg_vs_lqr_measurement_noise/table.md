# Experiment 06 - LQG vs full-state LQR under measurement noise

| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ | RMSE | Energy $E_u$ | Peak $u_{max}$ | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LQR (full state, clean)** | 0 | 0.942 | 0 | 9.29e-05 | 0.015 | 1.49 | 6.97 | 0 | Stable |
| **LQG** | 0 | 2.07 | 0 | 0.00118 | 0.0249 | 2.21 | 3.28 | 0 | Stable |
| **Luenberger + K** | 0 | 0.496 | 0 | 0.000961 | 0.0165 | 16.2 | 15.1 | 0 | Stable |
| **LQG (overconfident)** | 0 | 0.828 | 0 | 0.000925 | 0.0215 | 2.97 | 4.8 | 0 | Stable |

- **Precision:** LQR (full state, clean) - lowest RMSE (0.015).
- **Energy:** LQR (full state, clean) - least control effort (1.49).

