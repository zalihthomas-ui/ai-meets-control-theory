# Experiment 12 - shielded tabular Q-learning (pendulum)

Torque |u| <= 4. Shield hands off RL -> classical finisher when |wrap(theta-pi)| <= 1.0 rad. `shield_active_frac` = fraction of steps the classical fallback drove.

| controller | min_err_rad | final_err_rad | held_upright | t_upright_s | control_energy | shield_active_frac |
| --- | --- | --- | --- | --- | --- | --- |
| tabular Q-learning (raw) | 0.028 | 1.406 | False | 10.3 | 105.4 | 0.0 |
| shielded (RL + classical finisher) | 0.0 | 0.0 | True | 9.3 | 68.0 | 0.437 |
