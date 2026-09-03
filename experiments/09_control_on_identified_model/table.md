# Experiment 09 - control on an identified model

| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ | RMSE | Energy $E_u$ | Peak $u_{max}$ | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LQR (true model)** | 0 | 1.05 | 0 | 9.87e-05 | 0.0171 | 2.34 | 8.71 | 0 | Stable |
| **LQR (identified, 300 steps)** | 0 | 6 | 0 | 59.4 | 30.2 | 1.59e+03 | 20 | 56.7 | Marginal |
| **LQR (identified, 1500 steps)** | 0 | 4.36 | 0 | 0.00618 | 0.0375 | 2.74 | 3.31 | 0 | Stable |
| **LQR (identified, 12000 steps)** | 0 | 2.57 | 0 | 0.000641 | 0.0254 | 1.41 | 4.34 | 0 | Stable |

- **Precision:** LQR (true model) - lowest RMSE (0.0171).
- **Energy:** LQR (identified, 12000 steps) - least control effort (1.41).


## Identification vs data length (closed-loop LS, sensor noise std [0.003, 0.006, 0.003, 0.006], 6 seeds, medians)

| data [steps] | A rel-Fro err | K_id error (rel) | closed-loop RMSE(theta) | stable runs |
| --- | --- | --- | --- | --- |
| 300 (1 s) | 5.86e+00 | 0.992 | 26.7697 | 1/6 |
| 1500 (3 s) | 4.19e-01 | 0.527 | 0.0336 | 4/6 |
| 12000 (24 s) | 2.17e-01 | 0.493 | 0.0246 | 6/6 |
