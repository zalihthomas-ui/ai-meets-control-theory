# Experiment 32 - direct collocation vs shooting vs sampling (offline)

## Task A - minimum-effort cart-pole swing-up (hard TERMINAL constraint)

`min integral(u^2) dt` s.t. the dynamics, `x(0)=[0,0,pi,0]`, `x(T)=[0,0,0,0]`, `|u|<=20` N, `T=2.0` s. `term_err_rolled` re-integrates the returned `u` through the true dynamics (FOH for collocation, ZOH for iLQR/CEM).

| planner | effort | term_err_planned | term_err_rolled | max_dyn_drift | peak_u | box_ok | knots | solve_ms | converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct collocation (HS) | 74.95 | 0 | 0.0008135 | 0.0008135 | 12.92 | yes | 41 | 711 | yes |
| iLQR / single shooting | 65.03 | 0.2527 | 0.2527 | 3.473e-05 | 12.69 | yes | 101 | 1390 | yes |
| CEM / sampling | 88.02 | 0.6492 | 0.6493 | 1.3e-05 | 13.11 | yes | 101 | 8269 | yes |

Collocation HS defect norm: 2.13e-12.

## Task B - planar point mass around a keep-out disk (hard PATH constraint)

`min integral(||a||^2) dt` s.t. the double integrator, `x(0)=[-2.0, 0.0]`, `x(T)=[2.0, 0.0]`, stay outside the disk centred `(0,0)` radius `0.70`, `|a|<=3`, `T=4.0` s. `min_clear` = min distance to the disk minus its radius; **negative = inside the keep-out zone**.

| planner | effort | term_err_planned | min_clear_planned | min_clear_rolled | path_len | peak_a | box_ok | knots | solve_ms | converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct collocation (HS) | 4.481 | 1.25e-22 | +0.00 mm | -0.00 mm | 4.294 | 1.477 | yes | 41 | 508 | yes |
| iLQR / single shooting | 3.953 | 0.1378 | -0.28 mm | -0.28 mm | 4.284 | 1.441 | yes | 81 | 1001 | yes |
| CEM / sampling | 4.181 | 0.145 | -0.08 mm | -0.08 mm | 4.297 | 1.432 | yes | 81 | 4552 | yes |

Collocation HS defect norm: 2.78e-16.

