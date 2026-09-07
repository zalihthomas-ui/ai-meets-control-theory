# Experiment 37 - tube MPC vs nominal MPC under a bounded disturbance

mass-spring-damper (m=1, k=1, c=4), slot `|x| <= 0.40` m, persistent force push `|w_v| = 0.07`, setpoint `x = 0.38` (near the wall).

mРPI box half-widths `z` (built at dt = 0.05): `[0.0700, 0.2825]`  (series terms 140, contraction ~0.817).
Tightened position box: `|x_nom| <= 0.330` (wall 0.40 minus tube width 0.0700).
Tightened input box: `|u_nom| <= 3.72` (of 6).

| controller | max_pos | box_violation | violates | steps_over | x_ss | peak_u | rms_u |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal LinearMPC | 0.407 | 0.007 | **YES** | 184 | 0.405 | 6.000 | 1.214 |
| TubeMPC | 0.400 | 0.000 | no | 0 | 0.400 | 3.725 | 1.127 |

Nominal MPC sits 6.5 mm outside the slot for 184 of 200 steps; tube MPC never leaves it. The cost: the tube's nominal setpoint is 0.330 m, 50 mm short of the 0.38 m the nominal controller aims for.

