| Controller | Rise $t_r$ [s] | Settling $t_s$ [s] | Overshoot $M_p$ [%] | Steady error $e_{ss}$ [cm] | RMSE [cm] | Energy $E_u$ [$\text{N}^2\cdot\text{m}^2\cdot\text{s}$] | Peak Torque $|\tau|_{\max}$ [N$\cdot$m] | Saturation [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Cascade PID | 0.284 | 5.000 | 10.5 | 5.51e-01 | 5.81 | 0.0243 | 1.500 | 0.0 | Stable |
| PFL (Nonlinear) | 0.302 | 2.404 | 18.3 | 2.51e-03 | 5.88 | 0.0249 | 1.500 | 0.1 | Stable |
| Multivariable LQR | 0.576 | 1.492 | 1.3 | 1.85e-05 | 6.46 | 0.0187 | 0.966 | 0.0 | Stable |
| Linear MPC | 0.576 | 1.494 | 1.3 | 1.83e-05 | 6.47 | 0.0184 | 0.784 | 0.0 | Stable |
