# Experiment 04 - Cart-pole: LQR vs pole placement vs PID

| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ | RMSE | Energy $E_u$ | Peak $u_{max}$ | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pole placement** | 0 | 1.05 | 0 | 9.87e-05 | 0.0171 | 2.34 | 8.71 | 0 | Stable |
| **LQR (balanced)** | 0 | 1.05 | 0 | 9.87e-05 | 0.0171 | 2.34 | 8.71 | 0 | Stable |
| **LQR (soft)** | 0 | 1.7 | 0 | 0.00193 | 0.0223 | 0.959 | 3.08 | 0 | Stable |
| **PID (angle only)** | 0 | 0.202 | 0 | 4.28e-22 | 0.011 | 8.76 | 18 | 0 | Stable |

- **Precision:** PID (angle only) - lowest RMSE (0.011).
- **Energy:** LQR (soft) - least control effort (0.959).


## Peak cart excursion (rail limit +/- 2.4 m)

- **Pole placement**: peak |cart x| = 0.162 m (on rail)
- **LQR (balanced)**: peak |cart x| = 0.162 m (on rail)
- **LQR (soft)**: peak |cart x| = 0.321 m (on rail)
- **PID (angle only)**: peak |cart x| = 0.823 m (on rail)

## Designed feedback gains  K = [x, xdot, theta, thetadot]

- **Pole placement**: [-10.000, -12.870, -87.137, -23.223]
- **LQR (balanced)**: [-10.000, -12.871, -87.138, -23.223]
- **LQR (soft)**: [-1.000, -2.047, -30.846, -7.868]
