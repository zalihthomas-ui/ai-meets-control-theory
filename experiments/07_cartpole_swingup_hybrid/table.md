# Experiment 07 - cart-pole swing-up + hybrid handoff

From hanging (theta0 = pi), |F| <= 20 N. `t_capture` = first switch to LQR balance; `t_settle` = |wrap(theta)| < 2 deg thereafter.

| controller | t_capture_s | t_settle_s | n_switches | control_energy | peak_force_N | cart_excursion_m | balanced |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k_E = 6 | 6.496 | 7.452 | 1 | 52.28 | 12.23 | 1.012 | True |
| k_E = 10 | 2.852 | 4.058 | 1 | 87.28 | 20.0 | 1.536 | True |
| k_E = 14 | 1.686 | 3.344 | 1 | 107.24 | 19.27 | 1.94 | True |
