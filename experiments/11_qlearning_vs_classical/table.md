# Experiment 11 - tabular Q-learning vs classical control (pendulum)

Torque |u| <= 4. `train_samples` = env steps consumed to learn the policy (0 for the model-based classical laws). `n_params` = size of the resulting controller.

| controller | task | min_err_rad | held_upright | t_upright_s | final_err_rad | control_energy | train_samples | n_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| energy-shaping + LQR | swing-up | 0.0 | True | 4.0 | 0.0 | 45.9 | 0 | 3 |
| tabular Q-learning | swing-up | 0.02 | False | 9.1 | 1.479 | 101.2 | 450000 | 61875 |
| LQR | balance | 0.0 | True | 0.85 | 0.0 | 11.7 | 0 | 2 |
| tabular Q-learning | balance | 0.047 | False | 3.45 | 1.527 | 40.4 | 180000 | 51975 |
