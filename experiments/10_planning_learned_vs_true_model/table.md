# Experiment 10 - planning with a learned model vs the true model

| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ | RMSE | Energy $E_u$ | Peak $u_{max}$ | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LQR (true model)** | 0 | 1.26 | 0 | 0.000229 | 0.0374 | 8.42 | 17.4 | 0 | Stable |
| **SamplingMPC (true model)** | 0 | 4.44 | 0 | 0.00243 | 0.0448 | 10.8 | 6.59 | 0 | Stable |
| **SamplingMPC (learned model)** | 0 | 3.56 | 0 | 0.0014 | 0.0462 | 10.2 | 6 | 0 | Stable |

- **Precision:** LQR (true model) - lowest RMSE (0.0374).
- **Energy:** LQR (true model) - least control effort (8.42).


## Learned model

- residual MLP `[5, 64, 64, 4]`, 4804 params, trained on 4000 steps (80 s); final train MSE 1.96e-04
- held-out prediction error: 1-step 4.45e-04, 30-step 2.50e-02
