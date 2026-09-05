# Experiment 27 - double lane change on the dynamic bicycle model

## Part A - nominal (25 m/s, 3.5 m shift over ~70 m, linear tire, 8 s)

| controller | rms_err_mm | max_err_mm | final_err_mm | peak_delta_deg | ctrl_energy | status |
| --- | --- | --- | --- | --- | --- | --- |
| Stanley | 371.4 | 845.6 | 114.6 | 1.767 | 0.01138 | OK |
| LQR | 96.06 | 187.1 | 0.05542 | 2.973 | 0.02903 | OK |
| Kinematic MPC | 52.53 | 117.8 | 0.2261 | 3.215 | 0.02785 | OK |
| RL (PPO) | 84.31 | 166.9 | 2.707 | 2.914 | 0.01489 | OK |

## Part B - aggressive (6 m sharpness, Pacejka tire, mu=0.6, controllers unaware of the swap)

| controller | rms_err_mm | max_err_mm | final_err_mm | peak_delta_deg | ctrl_energy | status |
| --- | --- | --- | --- | --- | --- | --- |
| Stanley | 733.7 | 1973 | 32.65 | 12.37 | 0.5062 | OK |
| LQR | 767.8 | 1712 | 314.6 | 30 | 7.897 | OK |
| Kinematic MPC | 1326 | 2633 | 1382 | 30 | 16.01 | OK |
| RL (PPO) | 5223 | 1.138e+04 | 1.138e+04 | 30 | 2.682 | OK |

