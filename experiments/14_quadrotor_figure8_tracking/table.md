# Experiment 14 - quadrotor figure-8 tracking (Crazyflie 2.0)

| controller | rms_pos_err_mm | max_pos_err_mm | rms_pitch_deg | ctrl_energy | peak_thrust_N | thrust_sat_pct |
| --- | --- | --- | --- | --- | --- | --- |
| LQR + flatness feedforward | 43.5 | 112 | 4.29 | 0.0329 | 0.6 | 0.7 |
| LQR feedback only | 51 | 109 | 4.35 | 0.0379 | 0.6 | 0.733 |
| Linear MPC (single setpoint) | 142 | 208 | 4.3 | 0.0264 | 0.493 | 0.1 |
| Linear MPC (preview) | 47.9 | 134 | 4.52 | 0.0326 | 0.6 | 0.566 |
