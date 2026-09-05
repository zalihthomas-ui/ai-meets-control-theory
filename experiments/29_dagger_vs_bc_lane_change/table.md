# Experiment 29 - DAgger vs behaviour cloning on the Exp-27 Part-B lane change

aggressive double lane change (6 m sharpness), Pacejka tyre mu=0.6, 25 m/s.

| controller | rms_err_mm | max_err_mm | final_err_mm | peak_delta_deg | ctrl_energy | status |
| --- | --- | --- | --- | --- | --- | --- |
| Stanley | 733.7 | 1973 | 32.65 | 12.37 | 0.5062 | OK |
| LQR | 767.8 | 1712 | 314.6 | 30 | 7.897 | OK |
| Kinematic MPC | 1326 | 2633 | 1382 | 30 | 16.01 | OK |
| Plain BC | 6022 | 1.243e+04 | 1.221e+04 | 30 | 2.102 | OK |
| DAgger | 768.8 | 1708 | 315.9 | 30 | 7.86 | OK |

