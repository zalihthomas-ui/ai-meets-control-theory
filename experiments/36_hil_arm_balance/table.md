| Controller | Delay Margin $\tau_{\max}$ [ms] | Poke Settling $t_s$ [s] | Max Tip $|\theta_1 - \pi/2|_{\max}$ [deg] | Wind Bias $e_{\text{wind}}$ [deg] | Post-Wind $e_{ss}$ [deg] | Energy $E_u$ [$\text{N}^2\cdot\text{m}^2\cdot\text{s}$] | Peak Torque $|\tau|_{\max}$ [$\text{N}\cdot\text{m}$] | Slew Sat [%] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LQR (Stiff) | 24 | 1.026 | 7.73 | 6.43 | 0.348 | 8.57 | 2.94 | 0.0 | **Stable** |
| LQR + Integral (LQI) | 24 | 0.790 | 6.91 | 5.76 | 1.014 | 7.79 | 2.75 | 0.0 | **Stable** |
| LQR (Soft) | 16 | 1.500 | 48.81 | 28.96 | 42.298 | 87.69 | 7.43 | 0.0 | **Stable** |
