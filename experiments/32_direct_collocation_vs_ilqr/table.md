# Experiment 32 - direct collocation vs shooting vs sampling (offline)

minimum-effort cart-pole swing-up: `min integral(u^2) dt` s.t. the dynamics, `x(0)=[0,0,pi,0]`, `x(T)=[0,0,0,0]`, `|u|<=20` N, `T=2.0` s.

`term_err_planned` = ||x(T) - goal|| from the planner's own knots; `term_err_rolled` = same after re-integrating the returned `u` through the true dynamics (fine RK4, first-order hold). `max_dyn_drift` = worst knot mismatch between the plan and that re-integration.

| planner | effort | term_err_planned | term_err_rolled | max_dyn_drift | peak_u_N | box_ok | knots | solve_ms | converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct collocation (HS) | 74.95 | 0 | 0.0008135 | 0.0008135 | 12.92 | yes | 41 | 710 | yes |
| iLQR / single shooting | 65.03 | 0.2527 | 0.2527 | 3.473e-05 | 12.69 | yes | 101 | 1412 | yes |
| CEM / sampling | 88.02 | 0.6492 | 0.6493 | 1.3e-05 | 13.11 | yes | 101 | 8547 | yes |

Collocation Hermite-Simpson defect norm at the solution: 2.13e-12 (the NLP equalities are satisfied to solver tolerance; the residual drift is the inter-knot quadrature error).

