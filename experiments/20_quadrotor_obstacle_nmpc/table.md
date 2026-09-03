# Experiment 20 - obstacle-aware nonlinear MPC (quadrotor)

Learned grey-box model 20-step prediction error: 0.001

| controller | rms_pos_err_mm | min_clearance_mm | steps_in_keepout | ctrl_energy |
| --- | --- | --- | --- | --- |
| LQR + flatness feedforward | 86.26 | -85.8 | 26 | 0.4804 |
| SamplingMPC (true model) | 234.4 | 10.88 | 0 | 0.03291 |
| SamplingMPC (learned model) | 254.6 | 26.06 | 0 | 0.0343 |
