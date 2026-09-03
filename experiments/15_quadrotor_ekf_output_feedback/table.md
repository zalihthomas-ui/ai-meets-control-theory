# Experiment 15 - output-feedback quadrotor tracking (EKF)

Noisy [x, z, theta] + gyro; velocities unmeasured. sigma = [0.003, 0.003, 0.005235987755982988, 0.01] (m, m, rad, rad/s).

| controller | rms_pos_err_mm | max_pos_err_mm | rms_pitch_deg | ctrl_energy | rms_vel_est_err |
| --- | --- | --- | --- | --- | --- |
| LQR (full state) | 9.04 | 49.0 | 1.256 | 0.0024 | -- |
| LQR + EKF | 9.51 | 50.58 | 1.248 | 0.0026 | 0.0078 |
| LQR + finite-diff vel | 18.32 | 47.4 | 1.888 | 0.3578 | 0.3316 |
