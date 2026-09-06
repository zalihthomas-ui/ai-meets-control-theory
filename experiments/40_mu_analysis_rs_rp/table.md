# Experiment 40 - mu analysis: structured robust stability / performance

Exp-35 plant `G = 12/((s+1)(s+3))`, a moderate LQG (nominal crossover ~5 rad/s). Output-multiplicative uncertainty `Delta = blkdiag(delta_r * 0.45 loop gain, Delta_c * (R(s)-1))`, `omega_r = 15` rad/s. Both enter at the plant output, so `M = -T [w_g  W_m ; w_g  W_m]` is rank one and `mu = max_w |T|(w_g + |W_m|)` exactly.

`aimct.robust.mu` solver peak vs that analytic value: max relative error **3.7e-16** across the sweep.

## Part (a) - as the resonance grows (zeta_r down)

LQG nominal single-loop margins (on `G` alone): **GM 14.0 dB, PM 66.5 deg** - identical on every row below.

| checked one at a time | crosses 1 at zeta_r |
| --- | --- |
| gain error alone  `w_g ||T||_inf` | n/a (never - `0.41` and falling) |
| resonance alone   `||W_m T||_inf` | **0.059** |
| nominal mode `Delta_c = 1` destabilises `G*R` | **n/a** |

| checked *together* | crosses 1 at zeta_r |
| --- | --- |
| structured `mu(delta_r, Delta_c)` | **0.062** |

Neither uncertainty on its own takes the loop down anywhere in this range - each single-loop check, and the nominal `G*R` pole, stays on the safe side until `zeta_r ~ 0.059` or below. `mu` says the *combination* is already unsafe at `zeta_r ~ 0.062`: there is a `||Delta|| <= 1` mixing the 45% gain error with a fraction of the resonance that destabilises the loop. The gain and phase margins, blind to structure, never move.

| zeta_r | mu (analytic) | mu (solver ub) | mu (solver lb) | ||W_m T|| | w_g||T|| | RS margin 1/mu | nom `G*R` pole | GM dB | PM deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.654 | 0.654 | 0.653 | 0.322 | 0.410 | 1.530 | -2.531 | 14.0 | 66.5 |
| 0.383 | 0.645 | 0.645 | 0.645 | 0.315 | 0.410 | 1.551 | -2.607 | 14.0 | 66.5 |
| 0.367 | 0.636 | 0.636 | 0.636 | 0.309 | 0.410 | 1.572 | -2.689 | 14.0 | 66.5 |
| 0.350 | 0.627 | 0.627 | 0.627 | 0.303 | 0.410 | 1.594 | -2.775 | 14.0 | 66.5 |
| 0.333 | 0.619 | 0.619 | 0.619 | 0.298 | 0.410 | 1.616 | -2.868 | 14.0 | 66.5 |
| 0.316 | 0.610 | 0.610 | 0.610 | 0.293 | 0.410 | 1.640 | -2.966 | 14.0 | 66.5 |
| 0.300 | 0.601 | 0.601 | 0.601 | 0.289 | 0.410 | 1.663 | -3.071 | 14.0 | 66.5 |
| 0.283 | 0.593 | 0.593 | 0.593 | 0.288 | 0.410 | 1.688 | -3.183 | 14.0 | 66.5 |
| 0.266 | 0.584 | 0.584 | 0.583 | 0.291 | 0.410 | 1.713 | -3.302 | 14.0 | 66.5 |
| 0.249 | 0.575 | 0.575 | 0.575 | 0.297 | 0.410 | 1.738 | -3.427 | 14.0 | 66.5 |
| 0.233 | 0.567 | 0.567 | 0.567 | 0.307 | 0.410 | 1.763 | -3.559 | 14.0 | 66.5 |
| 0.216 | 0.559 | 0.559 | 0.558 | 0.320 | 0.410 | 1.789 | -3.697 | 14.0 | 66.5 |
| 0.199 | 0.551 | 0.551 | 0.550 | 0.336 | 0.410 | 1.815 | -3.840 | 14.0 | 66.5 |
| 0.182 | 0.543 | 0.543 | 0.542 | 0.358 | 0.410 | 1.841 | -3.860 | 14.0 | 66.5 |
| 0.166 | 0.536 | 0.536 | 0.535 | 0.384 | 0.410 | 1.867 | -3.547 | 14.0 | 66.5 |
| 0.149 | 0.528 | 0.528 | 0.528 | 0.419 | 0.410 | 1.892 | -3.229 | 14.0 | 66.5 |
| 0.132 | 0.522 | 0.522 | 0.522 | 0.463 | 0.410 | 1.917 | -2.906 | 14.0 | 66.5 |
| 0.115 | 0.578 | 0.578 | 0.578 | 0.522 | 0.410 | 1.729 | -2.580 | 14.0 | 66.5 |
| 0.099 | 0.656 | 0.656 | 0.656 | 0.600 | 0.410 | 1.524 | -2.253 | 14.0 | 66.5 |
| 0.082 | 0.770 | 0.770 | 0.769 | 0.717 | 0.410 | 1.298 | -1.926 | 14.0 | 66.5 |
| 0.065 | 0.946 | 0.946 | 0.946 | 0.893 | 0.410 | 1.057 | -1.600 | 14.0 | 66.5 |
| 0.048 | 1.240 | 1.240 | 1.240 | 1.187 | 0.410 | 0.806 | -1.277 | 14.0 | 66.5 |
| 0.032 | 1.818 | 1.818 | 1.817 | 1.765 | 0.410 | 0.550 | -0.956 | 14.0 | 66.5 |
| 0.015 | 3.340 | 3.340 | 3.338 | 3.287 | 0.410 | 0.299 | -0.639 | 14.0 | 66.5 |

## Part (b) - constant-D D-K on the Exp-35 H-infinity design

Same structure at a severe `zeta_r = 0.04` (where plain `mixsyn` peak mu is above 1), 2 constant-D round(s):

| design | peak mu | RS margin | ||S||_inf | gamma |
| --- | --- | --- | --- | --- |
| plain mixsyn (Exp 35) | 1.010 | 0.990 | 1.01 | 0.981 |
| + constant-D D-K | 0.995 | 1.005 | 1.00 | 0.983 |

